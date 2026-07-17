"""
Variable Neighborhood Search (VNS) para el PVRP, siguiendo el esquema de
Hemmelmayr, Doerner y Hartl (2009).

Parte de una solución inicial (Greedy si no se provee otra), aplica búsqueda
local hasta un óptimo local, y en cada iteración perturba la solución
incumbente (shaking) con intensidad creciente k, reiniciando k cuando la
solución perturbada es aceptada.

La búsqueda admite soluciones infactibles por capacidad. El objetivo que guía
la búsqueda es el costo penalizado

    f(x) = distancia(x) + w * exceso_de_capacidad(x)

con w ajustado dinámicamente: se multiplica por `penalty_factor` cuando la
incumbente es infactible y se divide por el mismo valor cuando es factible,
manteniéndose acotado en [penalty_min, penalty_max]. La aceptación de
soluciones peores sigue una regla tipo Simulated Annealing sobre f, con
temperatura que decrece linealmente hasta cero a lo largo de la ejecución.
La mejor solución factible encontrada se registra aparte y es la que se
devuelve.

Correspondencia con los parámetros del artículo original y desviaciones
declaradas de esta implementación:

  * penalty_max = penalty_min inicial = 1000, penalty_min = 10 y
    penalty_factor = 1.001 son los valores del artículo (Sección 4.1.2).
  * initial_temperature = 7.0 es el valor del artículo para las instancias
    p01-p26, que incluyen a las evaluadas en este trabajo.
  * El artículo enfría la temperatura cada 1000 iteraciones sobre un
    presupuesto de 10^7 iteraciones. Con los presupuestos utilizados aquí ese
    esquema no aplica, de modo que la temperatura decrece de forma lineal y
    continua sobre el presupuesto efectivo. Es una desviación declarada.
  * El artículo penaliza además el exceso de duración de ruta. Esta
    implementación penaliza únicamente el exceso de capacidad.
  * El número de vehículos por día se respeta de forma estructural, no
    mediante penalización.

Entrada: una Instance (src/data/instance.py), tope de iteraciones/tiempo,
k_max, una Solution inicial opcional, una semilla y los parámetros de
penalización y temperatura.
Salida: un VNSResult con la mejor Solution factible encontrada y estadísticas
de la ejecución.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from src.baselines.greedy import greedy_solve
from src.baselines.vns_operators import (
    NEIGHBORHOOD_OPERATORS,
    penalized_cost,
    total_capacity_excess,
)
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
    accepted_moves: int = 0
    sa_accepted_moves: int = 0
    infeasible_iterations: int = 0
    final_penalty_weight: float = 0.0


# Aplica los operadores de vecindad repetidamente hasta no encontrar más mejoras
# del costo penalizado (óptimo local respecto al objetivo vigente).
def _local_search(
    solution: Solution,
    instance: Instance,
    matrix,
    id_to_idx,
    demands: Dict[int, float],
    weight: float,
    operator_hits: dict,
) -> Solution:
    current = solution
    current_f = penalized_cost(current, instance, demands, weight)
    while True:
        improved = False
        for op_name, op_fn in NEIGHBORHOOD_OPERATORS:
            candidate = op_fn(current, instance, matrix, id_to_idx, demands, weight)
            if candidate is None:
                continue
            candidate_f = penalized_cost(candidate, instance, demands, weight)
            if candidate_f < current_f - 1e-9:
                current = candidate
                current_f = candidate_f
                operator_hits[op_name] = operator_hits.get(op_name, 0) + 1
                improved = True
                break
        if not improved:
            return current


# Probabilidad de aceptar una solución peor, según la regla de Simulated Annealing.
def _accept_worse(delta: float, temperature: float, rng: random.Random) -> bool:
    if temperature <= 1e-12:
        return False
    exponent = -delta / temperature
    if exponent < -700.0:
        return False
    return rng.random() < math.exp(exponent)


# Ejecuta el VNS completo (búsqueda local + shaking) sobre una instancia del PVRP.
def vns_solve(
    instance: Instance,
    max_iterations: int = 250,
    time_limit: Optional[float] = None,
    k_max: int = 6,
    initial_solution: Optional[Solution] = None,
    seed: int = 0,
    penalty_init: float = 1000.0,
    penalty_min: float = 10.0,
    penalty_max: float = 1000.0,
    penalty_factor: float = 1.001,
    initial_temperature: float = 7.0,
    verbose: bool = False,
) -> VNSResult:
    matrix = build_distance_matrix(instance)
    id_to_idx = build_id_to_index_map(instance)
    rng = random.Random(seed)

    demands: Dict[int, float] = {c.id: c.demand for c in instance.customers}
    demands[0] = 0.0

    weight = penalty_init
    operator_hits: dict = {}

    if initial_solution is None:
        current = greedy_solve(instance)
    else:
        current = initial_solution

    current = _local_search(
        current, instance, matrix, id_to_idx, demands, weight, operator_hits
    )
    initial_cost = current.total_cost(instance)

    # La mejor solución factible se registra por separado: la incumbente puede
    # ser infactible durante la búsqueda, pero lo que se reporta no.
    best = current
    best_cost = initial_cost
    if total_capacity_excess(current, instance, demands) > 1e-9:
        best_cost = float("inf")

    if verbose:
        print(f"[VNS] Costo inicial (Greedy+LS): {initial_cost:.2f}")

    k = 1
    iterations = 0
    local_search_calls = 1
    shaking_calls = 0
    accepted_moves = 0
    sa_accepted_moves = 0
    infeasible_iterations = 0
    start_time = time.time()

    while iterations < max_iterations:
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break

        temperature = initial_temperature * (1.0 - iterations / max_iterations)

        perturbed = shake(
            current, instance, k=k, rng=rng, matrix=matrix, id_to_idx=id_to_idx
        )
        shaking_calls += 1

        candidate = _local_search(
            perturbed, instance, matrix, id_to_idx, demands, weight, operator_hits
        )
        local_search_calls += 1

        current_f = penalized_cost(current, instance, demands, weight)
        candidate_f = penalized_cost(candidate, instance, demands, weight)
        delta = candidate_f - current_f

        if delta < -1e-9:
            current = candidate
            accepted_moves += 1
            k = 1
        elif _accept_worse(delta, temperature, rng):
            current = candidate
            accepted_moves += 1
            sa_accepted_moves += 1
            k = 1
        else:
            k = k + 1 if k < k_max else 1

        # Registro de la mejor solución factible vista hasta ahora. El exceso de
        # capacidad es la única violación posible (el shaking preserva patrones
        # válidos y la flota por construcción), pero se confirma con el
        # validador completo antes de aceptar un nuevo mejor.
        excess = total_capacity_excess(current, instance, demands)
        if excess > 1e-9:
            infeasible_iterations += 1
            weight = min(penalty_max, weight * penalty_factor)
        else:
            weight = max(penalty_min, weight / penalty_factor)
            candidate_cost = current.total_cost(instance)
            if candidate_cost < best_cost - 1e-9:
                feasible, _ = current.is_feasible(instance)
                if feasible:
                    best = current
                    best_cost = candidate_cost
                    if verbose:
                        print(f"[VNS] iter {iterations:5d}  costo={best_cost:.2f}  "
                              f"w={weight:.1f}  T={temperature:.3f}")

        iterations += 1

    elapsed = time.time() - start_time

    if best_cost == float("inf"):
        # Ninguna solución factible durante la búsqueda: se devuelve el Greedy.
        best = greedy_solve(instance)
        best_cost = best.total_cost(instance)

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
        accepted_moves=accepted_moves,
        sa_accepted_moves=sa_accepted_moves,
        infeasible_iterations=infeasible_iterations,
        final_penalty_weight=weight,
    )
