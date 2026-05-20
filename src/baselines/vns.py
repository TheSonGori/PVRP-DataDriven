"""
Variable Neighborhood Search (VNS) para el PVRP — versión completa.

Esta implementación sigue la estructura clásica de **General VNS**
[Hansen & Mladenović, 2001; Hemmelmayr et al., 2009]:

    1. Construir una solución inicial con la heurística Greedy.
    2. Bucle principal:
        a. Búsqueda local: aplicar operadores de vecindad hasta óptimo local.
        b. Si el óptimo local mejora la mejor solución, aceptarla y resetear k=1.
        c. Si no mejora, aumentar k (más perturbación) y aplicar shaking.
        d. Repetir hasta agotar iteraciones o tiempo.

El parámetro `k` controla la **intensidad de la perturbación**: se incrementa
cuando la búsqueda se estanca y se reinicia al encontrar mejora. Esta
alternancia intensificación–diversificación es la esencia de VNS.

Referencias:
    - Hansen, P., & Mladenović, N. (2001). Variable neighborhood search.
    - Hemmelmayr et al. (2009). A variable neighborhood search heuristic
      for periodic routing problems.
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


@dataclass
class VNSResult:
    """Resultado de una ejecución del VNS."""
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


def _local_search(
    solution: Solution,
    instance: Instance,
    matrix,
    id_to_idx,
    operator_hits: dict,
) -> Solution:
    """
    Aplica iterativamente todos los operadores de vecindad hasta no
    encontrar mejora (óptimo local).
    """
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
                break  # reiniciar desde el primer operador
        if not improved:
            return current


def vns_solve(
    instance: Instance,
    max_iterations: int = 100,
    time_limit: Optional[float] = None,
    k_max: int = 3,
    initial_solution: Optional[Solution] = None,
    seed: int = 0,
    verbose: bool = False,
) -> VNSResult:
    """
    Resuelve una instancia del PVRP con VNS general (búsqueda local + shaking).

    Args:
        instance: Instancia del PVRP.
        max_iterations: Tope superior de iteraciones del bucle externo.
        time_limit: Tiempo máximo en segundos (None = sin límite).
        k_max: Nivel máximo de perturbación. Se incrementa cuando no hay
            mejora y se reinicia a 1 al encontrar una.
        initial_solution: Solución inicial. Si es None, se genera con Greedy.
        seed: Semilla aleatoria para reproducibilidad del shaking.
        verbose: Si True, imprime progreso por consola.

    Returns:
        Un `VNSResult` con la solución y estadísticas.
    """
    matrix = build_distance_matrix(instance)
    id_to_idx = build_id_to_index_map(instance)
    rng = random.Random(seed)

    if initial_solution is None:
        current = greedy_solve(instance)
    else:
        current = initial_solution

    # Búsqueda local inicial sobre el punto de partida.
    operator_hits: dict = {}
    current = _local_search(current, instance, matrix, id_to_idx, operator_hits)
    current_cost = current.total_cost(instance)
    initial_cost = current_cost  # registramos el costo POST-LS sobre Greedy

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

        # 1. Shaking: perturbar la mejor solución actual con intensidad k.
        perturbed = shake(best, instance, k=k, rng=rng)
        shaking_calls += 1

        # 2. Búsqueda local desde el punto perturbado.
        local_opt = _local_search(perturbed, instance, matrix, id_to_idx, operator_hits)
        local_search_calls += 1
        local_cost = local_opt.total_cost(instance)

        # 3. Aceptación: si mejora la mejor global, aceptar y reiniciar k.
        if local_cost < best_cost - 1e-9:
            best = local_opt
            best_cost = local_cost
            k = 1
            if verbose:
                print(f"[VNS] iter {iterations:4d}  costo={best_cost:.2f}  (k=1)")
        else:
            # No mejoró: aumentar perturbación.
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
