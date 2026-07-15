"""
Entrena y evalúa el agente PVRP-RL una vez por cada semilla dada y agrega
los resultados (media ± desviación de costo y gap, tasa de factibilidad),
ya que el rendimiento de PPO depende de la semilla de inicialización.

Entrada: una Instance, una lista de semillas, un PPOConfig base y,
opcionalmente, el costo de la BKS y un directorio donde guardar cada modelo.
Salida: un MultiSeedResult con una SeedRun por semilla y las estadísticas
agregadas (gap_mean/std/min/max, cost_mean/std, feasibility_rate).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import numpy as np

from src.agent.evaluate import evaluate_deterministic
from src.agent.policy_config import PPOConfig
from src.agent.train import train_agent
from src.data.instance import Instance


# Resultado de una única corrida (una semilla): costo, factibilidad, gap y tiempo de entrenamiento.
@dataclass
class SeedRun:
    seed: int
    cost: float
    feasible: bool
    num_routes: int
    gap_pct: Optional[float]
    train_time: float


# Resultado agregado sobre todas las semillas de una instancia.
@dataclass
class MultiSeedResult:
    instance_name: str
    n_seeds: int
    bks_cost: Optional[float]
    runs: list[SeedRun] = field(default_factory=list)

    @property
    def feasible_runs(self) -> list[SeedRun]:
        return [r for r in self.runs if r.feasible]

    @property
    def feasibility_rate(self) -> float:
        if not self.runs:
            return 0.0
        return len(self.feasible_runs) / len(self.runs)

    def _gaps(self) -> np.ndarray:
        gaps = [r.gap_pct for r in self.feasible_runs if r.gap_pct is not None]
        return np.array(gaps) if gaps else np.array([])

    def _costs(self) -> np.ndarray:
        costs = [r.cost for r in self.feasible_runs]
        return np.array(costs) if costs else np.array([])

    @property
    def gap_mean(self) -> Optional[float]:
        g = self._gaps()
        return float(g.mean()) if g.size else None

    @property
    def gap_std(self) -> Optional[float]:
        g = self._gaps()
        return float(g.std()) if g.size else None

    @property
    def gap_min(self) -> Optional[float]:
        g = self._gaps()
        return float(g.min()) if g.size else None

    @property
    def gap_max(self) -> Optional[float]:
        g = self._gaps()
        return float(g.max()) if g.size else None

    @property
    def cost_mean(self) -> Optional[float]:
        c = self._costs()
        return float(c.mean()) if c.size else None

    @property
    def cost_std(self) -> Optional[float]:
        c = self._costs()
        return float(c.std()) if c.size else None


# Entrena y evalúa el agente una vez por cada semilla, devolviendo el resultado agregado.
def run_multi_seed(
    instance: Instance,
    seeds: list[int],
    base_config: PPOConfig,
    bks_cost: Optional[float] = None,
    models_dir: Optional[Path] = None,
    verbose: bool = True,
) -> MultiSeedResult:
    result = MultiSeedResult(
        instance_name=instance.name,
        n_seeds=len(seeds),
        bks_cost=bks_cost,
    )

    for i, seed in enumerate(seeds, start=1):
        if verbose:
            print(f"\n[{i}/{len(seeds)}] Entrenando semilla {seed}...")

        config = replace(base_config, seed=seed)

        save_path = None
        if models_dir is not None:
            save_path = Path(models_dir) / f"ppo_{instance.name}_seed{seed}"

        start = time.time()
        model = train_agent(instance, config=config, save_path=save_path)
        train_time = time.time() - start

        ev = evaluate_deterministic(model, instance, bks_cost=bks_cost)

        run = SeedRun(
            seed=seed,
            cost=ev.cost,
            feasible=ev.feasible,
            num_routes=ev.num_routes,
            gap_pct=ev.gap_pct,
            train_time=train_time,
        )
        result.runs.append(run)

        if verbose:
            gap = f"{ev.gap_pct:+.1f}%" if ev.gap_pct is not None else "N/A"
            feas = "factible" if ev.feasible else "INFACTIBLE"
            print(f"    costo={ev.cost:.2f}  gap={gap}  {feas}  "
                  f"({train_time/60:.1f} min)")

    return result


# Imprime un resumen legible del análisis multi-semilla.
def print_multi_seed(result: MultiSeedResult) -> None:
    print(f"\n{'='*64}")
    print(f"  ANÁLISIS MULTI-SEMILLA — instancia {result.instance_name}")
    print(f"  ({result.n_seeds} semillas)")
    print(f"{'='*64}")
    if result.bks_cost is not None:
        print(f"  BKS (referencia): {result.bks_cost:.2f}")
    print(f"  {'-'*60}")
    print(f"  {'Semilla':>8} {'Costo':>10} {'Gap':>9} {'Factible':>10} {'Tiempo':>9}")
    print(f"  {'-'*60}")
    for r in result.runs:
        gap = f"{r.gap_pct:+.1f}%" if r.gap_pct is not None else "N/A"
        feas = "sí" if r.feasible else "NO"
        print(f"  {r.seed:>8} {r.cost:>10.2f} {gap:>9} {feas:>10} "
              f"{r.train_time/60:>7.1f}m")
    print(f"  {'-'*60}")
    print(f"  Tasa de factibilidad: {result.feasibility_rate*100:.0f}% "
          f"({len(result.feasible_runs)}/{len(result.runs)})")
    if result.gap_mean is not None:
        print(f"  Gap medio:   {result.gap_mean:+.1f}% ± {result.gap_std:.1f}%")
        print(f"  Gap mínimo:  {result.gap_min:+.1f}%   "
              f"Gap máximo: {result.gap_max:+.1f}%")
        print(f"  Costo medio: {result.cost_mean:.2f} ± {result.cost_std:.2f}")
    print(f"{'='*64}\n")
