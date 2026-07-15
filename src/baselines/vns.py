"""
Variable Neighborhood Search (VNS) general para el PVRP: parte de una
solución (Greedy si no se provee otra), aplica búsqueda local con los
operadores de vecindad hasta un óptimo local, y cuando no mejora perturba la
solución (shaking) con intensidad creciente k, reiniciando k al encontrar
mejora.

Entrada: una Instance (src/data/instance.py), tope de iteraciones/tiempo,
k_max, una Solution inicial opcional y una semilla.
Salida: un VNSResult con la mejor Solution encontrada y estadísticas de la
ejecución (costo inicial/final, mejora, iteraciones, llamadas a operadores).
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass, field
from typing import Optional

from src.baselines.greedy import greedy_solve
from src.baselines.vns_operators import NEIGHBORHOOD_OPERATORS
from src.baselines.vns_shaking import shake
from src.data.instance import Instance
from src.utils.distance import build_distance_matrix, build_id_to_index_map
from src.utils.solution import Solution


# Resultado de una ejecución del VNS: solución final y estadísticas de la búsqueda.
@dataclass
class VNSResult:
    solution: Solution
    initial_cost: float
    final_cost: float
    improvement: float
    improvement_pct: float
    iterations: int
    local_search_calls: int = 0
    shaking_calls: int = 0
    operator_hits: dict = field(default_factory=dict)
    elapsed_time: float = 0.0


# Aplica los operadores de vecindad repetidamente hasta no encontrar más mejoras (óptimo local).
def _local_search(
    solution: Solution,
    instance: Instance,
    matrix,
    id_to_idx,
    operator_hits: dict,
) -> Solution:
    current = solution
    current_cost = current.total_cost(instance)
    while True:
        improved = False
        for op_name, op_fn in NEIGHBORHOOD_OPERATORS:
            candidate = op_fn(current, instance, matrix, id_to_idx)
            if candidate is None:
                continue
            candidate_cost = candidate.total_cost(instance)
            if candidate_cost < current_cost - 1e-9:
                current = candidate
                current_cost = candidate_cost
                operator_hits[op_name] = operator_hits.get(op_name, 0) + 1
                improved = True
                break
        if not improved:
            return current


# Ejecuta el VNS completo (búsqueda local + shaking) sobre una instancia del PVRP.
def vns_solve(
    instance: Instance,
    max_iterations: int = 100,
    time_limit: Optional[float] = None,
    k_max: int = 3,
    initial_solution: Optional[Solution] = None,
    seed: int = 0,
    verbose: bool = False,
) -> VNSResult:
    matrix = build_distance_matrix(instance)
    id_to_idx = build_id_to_index_map(instance)
    rng = random.Random(seed)

    if initial_solution is None:
        current = greedy_solve(instance)
    else:
        current = initial_solution

    operator_hits: dict = {}
    current = _local_search(current, instance, matrix, id_to_idx, operator_hits)
    current_cost = current.total_cost(instance)
    initial_cost = current_cost

    best = current
    best_cost = current_cost

    if verbose:
        print(f"[VNS] Costo inicial (Greedy+LS): {initial_cost:.2f}")

    k = 1
    iterations = 0
    local_search_calls = 1
    shaking_calls = 0
    start_time = time.time()

    while iterations < max_iterations:
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break

        perturbed = shake(best, instance, k=k, rng=rng)
        shaking_calls += 1

        local_opt = _local_search(perturbed, instance, matrix, id_to_idx, operator_hits)
        local_search_calls += 1
        local_cost = local_opt.total_cost(instance)

        if local_cost < best_cost - 1e-9:
            best = local_opt
            best_cost = local_cost
            k = 1
            if verbose:
                print(f"[VNS] iter {iterations:4d}  costo={best_cost:.2f}  (k=1)")
        else:
            k = k + 1 if k < k_max else 1

        iterations += 1

    elapsed = time.time() - start_time
    improvement = initial_cost - best_cost
    pct = (improvement / initial_cost * 100) if initial_cost > 0 else 0.0

    return VNSResult(
        solution=best,
        initial_cost=initial_cost,
        final_cost=best_cost,
        improvement=improvement,
        improvement_pct=pct,
        iterations=iterations,
        local_search_calls=local_search_calls,
        shaking_calls=shaking_calls,
        operator_hits=operator_hits,
        elapsed_time=elapsed,
    )
