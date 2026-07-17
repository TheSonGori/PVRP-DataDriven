"""
Entrena (o carga) y evalúa el agente PVRP-RL una vez por cada semilla dada y
agrega los resultados (media ± desviación de costo y gap, tasa de
factibilidad), ya que el rendimiento de PPO depende de la semilla de
inicialización.

Permite además evaluar los modelos ya guardados sin reentrenarlos
(load_models=True) y aplicar una ablación de búsqueda local sobre las
soluciones construidas (local_search), para descomponer el gap del enfoque
constructivo puro. Ver src/agent/evaluate.py.

Entrada: una Instance, una lista de semillas, un PPOConfig base y,
opcionalmente, el costo de la BKS, un directorio de modelos, el modo de carga
y el nivel de búsqueda local.
Salida: un MultiSeedResult con una SeedRun por semilla y las estadísticas
agregadas (gap_mean/std/min/max, cost_mean/std, feasibility_rate).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import numpy as np

from sb3_contrib import MaskablePPO

from src.agent.evaluate import evaluate_deterministic
from src.agent.policy_config import PPOConfig
from src.agent.train import train_agent
from src.data.instance import Instance


# Grados de libertad usados al calcular las desviaciones estándar.
#
# 0 = desviación poblacional (np.std por defecto). Es el valor histórico de
# este módulo y el que produjo los números reportados en la memoria para el
# agente RL.
#
# ATENCIÓN: scripts/vns_multi_seed.py usa statistics.stdev, que es la
# desviación muestral (ddof=1). Con 5 semillas ambas difieren en un factor
# sqrt(5/4) = 1.118, de modo que las desviaciones del RL y las del VNS no son
# hoy directamente comparables. Unificar en ddof=1 es lo estándar, pero
# cambiaría los valores del RL ya reportados (por ejemplo, p01 pasaría de
# ±6.5 a ±7.3). La decisión se deja explícita aquí en lugar de quedar
# escondida en un valor por defecto de numpy.
STD_DDOF = 0


# Resultado de una única corrida (una semilla): costo, factibilidad, gap y tiempo.
@dataclass
class SeedRun:
    seed: int
    cost: float
    feasible: bool
    num_routes: int
    gap_pct: Optional[float]
    train_time: float
    cost_before_ls: Optional[float] = None
    gap_before_ls: Optional[float] = None
    ls_time: float = 0.0


# Resultado agregado sobre todas las semillas de una instancia.
@dataclass
class MultiSeedResult:
    instance_name: str
    n_seeds: int
    bks_cost: Optional[float]
    runs: list[SeedRun] = field(default_factory=list)
    local_search: str = "none"
    loaded_models: bool = False

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

    def _gaps_before(self) -> np.ndarray:
        gaps = [
            r.gap_before_ls for r in self.feasible_runs
            if r.gap_before_ls is not None
        ]
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
        return float(g.std(ddof=STD_DDOF)) if g.size else None

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
        return float(c.std(ddof=STD_DDOF)) if c.size else None

    # Gap medio ANTES de aplicar la búsqueda local (None si no se aplicó).
    @property
    def gap_before_mean(self) -> Optional[float]:
        g = self._gaps_before()
        return float(g.mean()) if g.size else None

    @property
    def gap_before_std(self) -> Optional[float]:
        g = self._gaps_before()
        return float(g.std(ddof=STD_DDOF)) if g.size else None

    # Puntos de gap que la búsqueda local recupera, en promedio.
    @property
    def gap_recovered(self) -> Optional[float]:
        if self.gap_before_mean is None or self.gap_mean is None:
            return None
        return self.gap_before_mean - self.gap_mean

    # Tiempo medio de la búsqueda local por corrida.
    @property
    def ls_time_mean(self) -> Optional[float]:
        times = [r.ls_time for r in self.feasible_runs]
        return float(np.mean(times)) if times else None


# Ruta del modelo guardado de una instancia y semilla dadas.
def model_path(models_dir: Path, instance_name: str, seed: int) -> Path:
    return Path(models_dir) / f"ppo_{instance_name}_seed{seed}"


# Entrena (o carga) y evalúa el agente una vez por cada semilla.
#
# Con load_models=True no se entrena nada: se cargan los modelos guardados en
# models_dir, lo que permite repetir evaluaciones y ablaciones sobre exactamente
# los mismos agentes que produjeron los resultados reportados.
def run_multi_seed(
    instance: Instance,
    seeds: list[int],
    base_config: PPOConfig,
    bks_cost: Optional[float] = None,
    models_dir: Optional[Path] = None,
    load_models: bool = False,
    local_search: str = "none",
    verbose: bool = True,
) -> MultiSeedResult:
    if load_models and models_dir is None:
        raise ValueError(
            "load_models=True requiere models_dir con los modelos guardados."
        )

    result = MultiSeedResult(
        instance_name=instance.name,
        n_seeds=len(seeds),
        bks_cost=bks_cost,
        local_search=local_search,
        loaded_models=load_models,
    )

    for i, seed in enumerate(seeds, start=1):
        path = model_path(models_dir, instance.name, seed) if models_dir else None

        if load_models:
            if not Path(str(path) + ".zip").exists():
                raise FileNotFoundError(
                    f"No existe el modelo {path}.zip. Entrena primero con "
                    f"--save-models o quita --load-models."
                )
            if verbose:
                print(f"\n[{i}/{len(seeds)}] Cargando semilla {seed}...")
            model = MaskablePPO.load(str(path))
            train_time = 0.0
        else:
            if verbose:
                print(f"\n[{i}/{len(seeds)}] Entrenando semilla {seed}...")
            config = replace(base_config, seed=seed)
            start = time.time()
            model = train_agent(instance, config=config, save_path=path)
            train_time = time.time() - start

        ev = evaluate_deterministic(
            model, instance, bks_cost=bks_cost, local_search=local_search
        )

        run = SeedRun(
            seed=seed,
            cost=ev.cost,
            feasible=ev.feasible,
            num_routes=ev.num_routes,
            gap_pct=ev.gap_pct,
            train_time=train_time,
            cost_before_ls=ev.cost_before_ls,
            gap_before_ls=ev.gap_before_ls,
            ls_time=ev.ls_time,
        )
        result.runs.append(run)

        if verbose:
            gap = f"{ev.gap_pct:+.1f}%" if ev.gap_pct is not None else "N/A"
            feas = "factible" if ev.feasible else "INFACTIBLE"
            extra = ""
            if ev.gap_before_ls is not None:
                extra = (f"  [antes {ev.gap_before_ls:+.1f}%, "
                         f"recupera {ev.gap_recovered:.1f} pts "
                         f"en {ev.ls_time:.3f}s]")
            tiempo = "" if load_models else f"  ({train_time/60:.1f} min)"
            print(f"    costo={ev.cost:.2f}  gap={gap}  {feas}{tiempo}{extra}")

    return result


# Imprime un resumen legible del análisis multi-semilla.
def print_multi_seed(result: MultiSeedResult) -> None:
    con_ls = result.local_search != "none"

    print(f"\n{'='*78}")
    print(f"  ANÁLISIS MULTI-SEMILLA — instancia {result.instance_name}")
    modo = "modelos cargados" if result.loaded_models else "modelos entrenados"
    ls = f", búsqueda local: {result.local_search}" if con_ls else ""
    print(f"  ({result.n_seeds} semillas, {modo}{ls})")
    print(f"{'='*78}")
    if result.bks_cost is not None:
        print(f"  BKS (referencia): {result.bks_cost:.2f}")
    print(f"  {'-'*74}")

    if con_ls:
        print(f"  {'Semilla':>8} {'Gap antes':>11} {'Gap después':>13} "
              f"{'Recupera':>10} {'t búsq.':>9} {'Factible':>10}")
        print(f"  {'-'*74}")
        for r in result.runs:
            antes = f"{r.gap_before_ls:+.1f}%" if r.gap_before_ls is not None else "N/A"
            desp = f"{r.gap_pct:+.1f}%" if r.gap_pct is not None else "N/A"
            rec = (f"{r.gap_before_ls - r.gap_pct:.1f}"
                   if r.gap_before_ls is not None and r.gap_pct is not None else "N/A")
            print(f"  {r.seed:>8} {antes:>11} {desp:>13} {rec:>10} "
                  f"{r.ls_time:>8.3f}s {('sí' if r.feasible else 'NO'):>10}")
    else:
        print(f"  {'Semilla':>8} {'Costo':>10} {'Gap':>9} {'Factible':>10} {'Tiempo':>9}")
        print(f"  {'-'*74}")
        for r in result.runs:
            gap = f"{r.gap_pct:+.1f}%" if r.gap_pct is not None else "N/A"
            feas = "sí" if r.feasible else "NO"
            print(f"  {r.seed:>8} {r.cost:>10.2f} {gap:>9} {feas:>10} "
                  f"{r.train_time/60:>7.1f}m")

    print(f"  {'-'*74}")
    print(f"  Tasa de factibilidad: {result.feasibility_rate*100:.0f}% "
          f"({len(result.feasible_runs)}/{len(result.runs)})")
    if result.gap_mean is not None:
        if con_ls:
            print(f"  Gap ANTES:   {result.gap_before_mean:+.1f}% ± "
                  f"{result.gap_before_std:.1f}%")
            print(f"  Gap DESPUÉS: {result.gap_mean:+.1f}% ± {result.gap_std:.1f}%")
            print(f"  Recupera:    {result.gap_recovered:.1f} puntos "
                  f"en {result.ls_time_mean:.3f} s promedio")
        else:
            print(f"  Gap medio:   {result.gap_mean:+.1f}% ± {result.gap_std:.1f}%")
        print(f"  Gap mínimo:  {result.gap_min:+.1f}%   "
              f"Gap máximo: {result.gap_max:+.1f}%")
        print(f"  Costo medio: {result.cost_mean:.2f} ± {result.cost_std:.2f}")
    print(f"{'='*78}\n")
