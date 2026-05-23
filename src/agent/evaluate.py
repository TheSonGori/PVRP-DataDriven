"""
Evaluación sistemática del agente PVRP-RL.

Este módulo centraliza toda la lógica de evaluación, separándola del
entrenamiento. Provee tres niveles de evaluación:

    1. evaluate_deterministic : una corrida con política determinística
       (argmax sobre la política). Reproducible; es el "resultado oficial"
       del agente.

    2. evaluate_stochastic    : N corridas muestreando de la política. Como
       PPO aprende una política estocástica, muestrear varias veces y quedarse
       con la mejor suele mejorar el resultado. Reporta media, desviación,
       mejor y peor.

    3. compare_methods        : ejecuta Greedy, VNS y el agente RL sobre la
       misma instancia y arma una tabla comparativa homogénea, incluyendo
       el gap respecto a la BKS cuando está disponible.

Estas funciones alimentan las tablas y figuras de los Capítulos 4 y 5.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.agent.train import build_env
from src.baselines.greedy import greedy_solve
from src.baselines.vns import vns_solve
from src.data.instance import Instance
from src.data.solution_loader import load_solution
from src.utils.solution import Solution


@dataclass
class EvalResult:
    """Resultado de evaluar un método sobre una instancia."""
    method: str
    cost: float
    feasible: bool
    num_routes: int
    elapsed_time: float
    gap_pct: Optional[float] = None  # respecto a la BKS, si se conoce

    def __str__(self) -> str:
        gap = f"{self.gap_pct:+.1f}%" if self.gap_pct is not None else "N/A"
        feas = "sí" if self.feasible else "NO"
        return (
            f"{self.method:<22} costo={self.cost:>9.2f}  gap={gap:>7}  "
            f"factible={feas}  rutas={self.num_routes}  t={self.elapsed_time:.3f}s"
        )


@dataclass
class StochasticEvalResult:
    """Resultado de evaluar el agente con N muestreos estocásticos."""
    method: str
    best_cost: float
    mean_cost: float
    std_cost: float
    worst_cost: float
    feasibility_rate: float
    n_runs: int
    elapsed_time: float
    best_solution: Optional[Solution] = None
    gap_pct: Optional[float] = None

    def __str__(self) -> str:
        gap = f"{self.gap_pct:+.1f}%" if self.gap_pct is not None else "N/A"
        return (
            f"{self.method:<22} mejor={self.best_cost:>9.2f}  gap={gap:>7}  "
            f"media={self.mean_cost:.2f}±{self.std_cost:.2f}  "
            f"factib={self.feasibility_rate*100:.0f}%  n={self.n_runs}"
        )


def _run_episode(model, env, deterministic: bool, max_steps: int = 5000) -> Solution:
    """Ejecuta un episodio completo y retorna la solución construida."""
    obs, _ = env.reset()
    terminated = False
    steps = 0
    while not terminated and steps < max_steps:
        mask = env.action_masks()
        action, _ = model.predict(
            obs, action_masks=mask, deterministic=deterministic
        )
        obs, _, terminated, truncated, _ = env.step(int(action))
        steps += 1
    return env.unwrapped.get_solution()


def _bks_cost(instance: Instance, data_dir: Path) -> Optional[float]:
    """Carga el costo de la BKS si el archivo .res existe."""
    bks_path = data_dir / f"{instance.name}.res"
    if bks_path.exists():
        try:
            return load_solution(bks_path).reported_cost
        except Exception:
            return None
    return None


def evaluate_deterministic(
    model,
    instance: Instance,
    seed: int = 42,
    bks_cost: Optional[float] = None,
) -> EvalResult:
    """
    Evalúa el agente con política determinística (una sola corrida).

    Args:
        model: Modelo MaskablePPO entrenado.
        instance: Instancia del PVRP.
        seed: Semilla del entorno.
        bks_cost: Costo de la BKS para calcular gap (opcional).

    Returns:
        Un `EvalResult`.
    """
    env = build_env(instance, seed=seed)
    start = time.time()
    sol = _run_episode(model, env, deterministic=True)
    elapsed = time.time() - start

    cost = sol.total_cost(instance)
    feasible, _ = sol.is_feasible(instance)
    gap = ((cost - bks_cost) / bks_cost * 100) if bks_cost else None

    return EvalResult(
        method="RL (determinístico)",
        cost=cost,
        feasible=feasible,
        num_routes=len(sol.routes),
        elapsed_time=elapsed,
        gap_pct=gap,
    )


def evaluate_stochastic(
    model,
    instance: Instance,
    n_runs: int = 20,
    base_seed: int = 1000,
    bks_cost: Optional[float] = None,
) -> StochasticEvalResult:
    """
    Evalúa el agente muestreando la política N veces, quedándose con la mejor
    solución factible.

    Como PPO aprende una política estocástica, muestrear varias trayectorias
    y elegir la mejor es una estrategia estándar que suele superar a la
    evaluación determinística sin costo de reentrenamiento.

    Args:
        model: Modelo MaskablePPO entrenado.
        instance: Instancia del PVRP.
        n_runs: Número de muestreos.
        base_seed: Semilla base (cada corrida usa base_seed + i).
        bks_cost: Costo de la BKS para calcular gap (opcional).

    Returns:
        Un `StochasticEvalResult` con estadísticas agregadas.
    """
    costs: list[float] = []
    feasibles: list[bool] = []
    best_cost = float("inf")
    best_solution: Optional[Solution] = None

    start = time.time()
    for i in range(n_runs):
        env = build_env(instance, seed=base_seed + i)
        sol = _run_episode(model, env, deterministic=False)
        cost = sol.total_cost(instance)
        feasible, _ = sol.is_feasible(instance)

        costs.append(cost)
        feasibles.append(feasible)

        # Nos quedamos con la mejor solución FACTIBLE.
        if feasible and cost < best_cost:
            best_cost = cost
            best_solution = sol
    elapsed = time.time() - start

    # Si ninguna corrida fue factible, reportamos el mejor costo igualmente.
    if best_solution is None:
        best_cost = min(costs)

    costs_arr = np.array(costs)
    gap = ((best_cost - bks_cost) / bks_cost * 100) if bks_cost else None

    return StochasticEvalResult(
        method="RL (estocástico)",
        best_cost=best_cost,
        mean_cost=float(costs_arr.mean()),
        std_cost=float(costs_arr.std()),
        worst_cost=float(costs_arr.max()),
        feasibility_rate=float(np.mean(feasibles)),
        n_runs=n_runs,
        elapsed_time=elapsed,
        best_solution=best_solution,
        gap_pct=gap,
    )


def evaluate_greedy(
    instance: Instance, bks_cost: Optional[float] = None
) -> EvalResult:
    """Evalúa la heurística Greedy sobre la instancia."""
    start = time.time()
    sol = greedy_solve(instance)
    elapsed = time.time() - start
    cost = sol.total_cost(instance)
    feasible, _ = sol.is_feasible(instance)
    gap = ((cost - bks_cost) / bks_cost * 100) if bks_cost else None
    return EvalResult(
        method="Greedy",
        cost=cost,
        feasible=feasible,
        num_routes=len(sol.routes),
        elapsed_time=elapsed,
        gap_pct=gap,
    )


def evaluate_vns(
    instance: Instance,
    max_iterations: int = 100,
    seed: int = 42,
    bks_cost: Optional[float] = None,
) -> EvalResult:
    """Evalúa el VNS sobre la instancia."""
    start = time.time()
    result = vns_solve(instance, max_iterations=max_iterations, seed=seed)
    elapsed = time.time() - start
    sol = result.solution
    cost = sol.total_cost(instance)
    feasible, _ = sol.is_feasible(instance)
    gap = ((cost - bks_cost) / bks_cost * 100) if bks_cost else None
    return EvalResult(
        method="VNS",
        cost=cost,
        feasible=feasible,
        num_routes=len(sol.routes),
        elapsed_time=elapsed,
        gap_pct=gap,
    )


def compare_methods(
    model,
    instance: Instance,
    data_dir: Path,
    n_stochastic_runs: int = 20,
    vns_iterations: int = 100,
) -> dict:
    """
    Ejecuta los tres métodos sobre la misma instancia y arma una comparación.

    Args:
        model: Agente RL entrenado.
        instance: Instancia del PVRP.
        data_dir: Carpeta con los archivos .res (para la BKS).
        n_stochastic_runs: Muestreos para la evaluación estocástica del RL.
        vns_iterations: Iteraciones del VNS.

    Returns:
        Diccionario con los resultados de cada método y la BKS.
    """
    bks = _bks_cost(instance, data_dir)

    results = {
        "bks": bks,
        "greedy": evaluate_greedy(instance, bks_cost=bks),
        "vns": evaluate_vns(instance, max_iterations=vns_iterations, bks_cost=bks),
        "rl_deterministic": evaluate_deterministic(model, instance, bks_cost=bks),
        "rl_stochastic": evaluate_stochastic(
            model, instance, n_runs=n_stochastic_runs, bks_cost=bks
        ),
    }
    return results


def print_comparison(results: dict, instance_name: str) -> None:
    """Imprime una tabla comparativa legible de los resultados."""
    print(f"\n{'='*70}")
    print(f"  COMPARACIÓN DE MÉTODOS — instancia {instance_name}")
    print(f"{'='*70}")
    if results["bks"] is not None:
        print(f"  BKS (referencia): {results['bks']:.2f}")
    print(f"{'-'*70}")
    print(f"  {results['greedy']}")
    print(f"  {results['vns']}")
    print(f"  {results['rl_deterministic']}")
    print(f"  {results['rl_stochastic']}")
    print(f"{'='*70}\n")
