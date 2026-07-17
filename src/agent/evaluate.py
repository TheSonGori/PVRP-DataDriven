"""
Evaluación del agente PVRP-RL: corrida determinística (política argmax),
corrida estocástica (N muestreos, se conserva la mejor solución factible), y
comparación homogénea entre Greedy, VNS y el agente RL sobre la misma
instancia, incluyendo el gap contra la BKS cuando existe el .res.

Ofrece además una ablación de búsqueda local sobre las soluciones del agente
(apply_local_search): permite descomponer el gap del enfoque constructivo puro
en el error de ordenamiento dentro de cada ruta y el error de estructura
(asignación de clientes a días y a vehículos). Ver LOCAL_SEARCH_LEVELS.

Entrada: un modelo MaskablePPO entrenado y una Instance (o, para
compare_methods, además un directorio con archivos .res).
Salida: EvalResult / StochasticEvalResult con costo, factibilidad, rutas,
tiempo y gap; o un dict con los resultados de los tres métodos.
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
from src.baselines.vns_operators import (
    NEIGHBORHOOD_OPERATORS,
    penalized_cost,
    two_opt_within_route,
)
from src.data.instance import Instance
from src.data.solution_loader import load_solution
from src.utils.distance import build_distance_matrix, build_id_to_index_map
from src.utils.solution import Solution


# Configuración del VNS de referencia. Debe coincidir con la reportada en la
# memoria y con la de scripts/vns_multi_seed.py: cualquier divergencia hace que
# el repositorio produzca números distintos a los del documento.
VNS_ITERATIONS = 8000
VNS_K_MAX = 6
VNS_SEED = 0


# Niveles de búsqueda local aplicables a una solución del agente.
#
#   "2opt"      Solo 2-opt intra-ruta: reordena la secuencia de visitas de cada
#               ruta sin mover ningún cliente de ruta ni de día. Aísla el error
#               de ordenamiento, es decir, los cruces de aristas.
#   "intra_day" Agrega los operadores que mueven clientes entre rutas de un
#               mismo día. Aísla, además, el error de asignación a vehículos.
#
# Deliberadamente ninguno incluye relocate_between_days: la asignación de días
# es lo que la propuesta afirma que emerge de la construcción secuencial, de
# modo que reoptimizarla haría que la ablación dejara de medir al agente. Lo
# que quede de gap tras "intra_day" es atribuible a esa asignación.
LOCAL_SEARCH_LEVELS = {
    "none": None,
    "2opt": [("2-opt", two_opt_within_route)],
    "intra_day": [t for t in NEIGHBORHOOD_OPERATORS if t[0] != "relocate_days"],
}


# Resultado de evaluar un método (una corrida) sobre una instancia.
@dataclass
class EvalResult:
    method: str
    cost: float
    feasible: bool
    num_routes: int
    elapsed_time: float
    gap_pct: Optional[float] = None
    local_search: str = "none"
    ls_time: float = 0.0
    cost_before_ls: Optional[float] = None
    gap_before_ls: Optional[float] = None

    # Puntos de gap recuperados por la búsqueda local, si se aplicó.
    @property
    def gap_recovered(self) -> Optional[float]:
        if self.gap_before_ls is None or self.gap_pct is None:
            return None
        return self.gap_before_ls - self.gap_pct

    def __str__(self) -> str:
        gap = f"{self.gap_pct:+.1f}%" if self.gap_pct is not None else "N/A"
        feas = "sí" if self.feasible else "NO"
        base = (
            f"{self.method:<22} costo={self.cost:>9.2f}  gap={gap:>7}  "
            f"factible={feas}  rutas={self.num_routes}  t={self.elapsed_time:.3f}s"
        )
        if self.local_search != "none" and self.gap_recovered is not None:
            base += (
                f"  [antes: {self.gap_before_ls:+.1f}%, "
                f"recupera {self.gap_recovered:.1f} pts en {self.ls_time:.3f}s]"
            )
        return base


# Resultado de evaluar el agente con N muestreos estocásticos de la política.
@dataclass
class StochasticEvalResult:
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


# Aplica búsqueda local a una solución hasta alcanzar un óptimo local.
#
# El peso de penalización se fija en el máximo del artículo (1000), de modo que
# ningún movimiento que exceda la capacidad resulte aceptable: la búsqueda
# preserva la factibilidad de la solución que recibe. La solución del agente no
# se modifica: se devuelve una nueva.
#
# Entrada: una Solution, la Instance y el nivel de búsqueda local.
# Salida: una nueva Solution de costo menor o igual.
def apply_local_search(
    solution: Solution,
    instance: Instance,
    level: str = "2opt",
    weight: float = 1000.0,
) -> Solution:
    if level not in LOCAL_SEARCH_LEVELS:
        raise ValueError(
            f"Nivel de búsqueda local desconocido: {level!r}. "
            f"Opciones: {sorted(LOCAL_SEARCH_LEVELS)}"
        )
    operators = LOCAL_SEARCH_LEVELS[level]
    if operators is None:
        return solution

    matrix = build_distance_matrix(instance)
    id_to_idx = build_id_to_index_map(instance)
    demands = {c.id: c.demand for c in instance.customers}
    demands[0] = 0.0

    current = solution
    current_f = penalized_cost(current, instance, demands, weight)
    while True:
        improved = False
        for _, op_fn in operators:
            candidate = op_fn(current, instance, matrix, id_to_idx, demands, weight)
            if candidate is None:
                continue
            candidate_f = penalized_cost(candidate, instance, demands, weight)
            if candidate_f < current_f - 1e-9:
                current = candidate
                current_f = candidate_f
                improved = True
                break
        if not improved:
            return current


# Ejecuta un episodio completo con el modelo dado y devuelve la solución construida.
def _run_episode(model, env, deterministic: bool, max_steps: int = 5000) -> Solution:
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


# Carga el costo de la BKS desde el .res correspondiente, si existe.
def _bks_cost(instance: Instance, data_dir: Path) -> Optional[float]:
    bks_path = data_dir / f"{instance.name}.res"
    if bks_path.exists():
        try:
            return load_solution(bks_path).reported_cost
        except Exception:
            return None
    return None


# Evalúa el agente con política determinística (una sola corrida, resultado oficial).
#
# Si local_search es distinto de "none", se aplica el nivel indicado sobre la
# solución construida y se reportan el costo antes y después, más el tiempo de
# la búsqueda local por separado. El tiempo de inferencia del agente
# (elapsed_time) no incluye ese refinamiento.
def evaluate_deterministic(
    model,
    instance: Instance,
    seed: int = 42,
    bks_cost: Optional[float] = None,
    local_search: str = "none",
) -> EvalResult:
    env = build_env(instance, seed=seed)
    start = time.time()
    sol = _run_episode(model, env, deterministic=True)
    elapsed = time.time() - start

    cost_before = sol.total_cost(instance)
    gap_before = ((cost_before - bks_cost) / bks_cost * 100) if bks_cost else None

    ls_time = 0.0
    if local_search != "none":
        t0 = time.time()
        sol = apply_local_search(sol, instance, level=local_search)
        ls_time = time.time() - t0

    cost = sol.total_cost(instance)
    feasible, _ = sol.is_feasible(instance)
    gap = ((cost - bks_cost) / bks_cost * 100) if bks_cost else None

    method = (
        "RL (determinístico)" if local_search == "none"
        else f"RL + {local_search}"
    )

    return EvalResult(
        method=method,
        cost=cost,
        feasible=feasible,
        num_routes=len(sol.routes),
        elapsed_time=elapsed,
        gap_pct=gap,
        local_search=local_search,
        ls_time=ls_time,
        cost_before_ls=cost_before if local_search != "none" else None,
        gap_before_ls=gap_before if local_search != "none" else None,
    )


# Evalúa el agente muestreando la política N veces y se queda con la mejor solución factible.
def evaluate_stochastic(
    model,
    instance: Instance,
    n_runs: int = 20,
    base_seed: int = 1000,
    bks_cost: Optional[float] = None,
) -> StochasticEvalResult:
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

        if feasible and cost < best_cost:
            best_cost = cost
            best_solution = sol
    elapsed = time.time() - start

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


# Evalúa la heurística Greedy sobre la instancia.
def evaluate_greedy(
    instance: Instance, bks_cost: Optional[float] = None
) -> EvalResult:
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


# Evalúa el VNS sobre la instancia, con la configuración reportada en la memoria.
def evaluate_vns(
    instance: Instance,
    max_iterations: int = VNS_ITERATIONS,
    k_max: int = VNS_K_MAX,
    seed: int = VNS_SEED,
    bks_cost: Optional[float] = None,
) -> EvalResult:
    start = time.time()
    result = vns_solve(
        instance, max_iterations=max_iterations, k_max=k_max, seed=seed
    )
    elapsed = time.time() - start
    sol = result.solution
    cost = sol.total_cost(instance)
    feasible, _ = sol.is_feasible(instance)
    gap = ((cost - bks_cost) / bks_cost * 100) if bks_cost else None
    return EvalResult(
        method=f"VNS ({max_iterations} iter)",
        cost=cost,
        feasible=feasible,
        num_routes=len(sol.routes),
        elapsed_time=elapsed,
        gap_pct=gap,
    )


# Ejecuta Greedy, VNS y el agente RL (determinístico + estocástico) sobre la misma
# instancia. Con local_search distinto de "none" agrega la variante refinada del
# agente bajo la clave "rl_local_search".
def compare_methods(
    model,
    instance: Instance,
    data_dir: Path,
    n_stochastic_runs: int = 20,
    vns_iterations: int = VNS_ITERATIONS,
    local_search: str = "none",
) -> dict:
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
    if local_search != "none":
        results["rl_local_search"] = evaluate_deterministic(
            model, instance, bks_cost=bks, local_search=local_search
        )
    return results


# Imprime una tabla comparativa legible de los resultados de compare_methods.
def print_comparison(results: dict, instance_name: str) -> None:
    print(f"\n{'='*70}")
    print(f"  COMPARACIÓN DE MÉTODOS — instancia {instance_name}")
    print(f"{'='*70}")
    if results["bks"] is not None:
        print(f"  BKS (referencia): {results['bks']:.2f}")
    print(f"{'-'*70}")
    print(f"  {results['greedy']}")
    print(f"  {results['vns']}")
    print(f"  {results['rl_deterministic']}")
    if "rl_local_search" in results:
        print(f"  {results['rl_local_search']}")
    print(f"  {results['rl_stochastic']}")
    print(f"{'='*70}\n")
