"""
Heurística constructiva Greedy para el PVRP.

Algoritmo
---------

Esta implementación sigue la estrategia clásica del "vecino más cercano"
adaptada al horizonte temporal del PVRP:

    1. Asignación de patrones (fase temporal):
       Para cada cliente, se selecciona uno de sus patrones permitidos
       siguiendo una regla determinística simple (el patrón con menor
       valor numérico, lo que tiende a distribuir las visitas
       homogéneamente en el horizonte).

    2. Construcción de rutas diarias (fase espacial):
       Para cada día d ∈ T:
            a. Mientras existan clientes pendientes para el día d:
                - Lanzar un vehículo desde el depósito.
                - Repetir hasta que no quepa ningún cliente más en el vehículo:
                    · Elegir el cliente pendiente más cercano que QUEPA en
                      la capacidad restante.
                · Cerrar ruta retornando al depósito.

Esta heurística es **miope**: cada decisión optimiza un criterio local
(distancia inmediata) sin considerar el impacto global. Sirve como
**línea base** para evaluar métodos más sofisticados (VNS, RL) en los
Capítulos 4 y 5 de la memoria.

Complejidad: O(N^2 * T) en el peor caso.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.data.instance import Customer, Instance
from src.utils.distance import build_distance_matrix, build_id_to_index_map
from src.utils.solution import Route, Solution


def _assign_patterns(instance: Instance) -> Dict[int, Tuple[int, ...]]:
    """
    Asigna a cada cliente uno de sus patrones permitidos balanceando la
    carga total entre los días del horizonte.

    Estrategia (greedy de balanceo):

        1. Ordenar los clientes de mayor a menor demanda total (demanda × frecuencia).
           Los clientes "pesados" se asignan primero para evitar quedar atrapados
           más tarde en days saturados.
        2. Para cada cliente, recorrer sus patrones permitidos y elegir aquel
           cuya suma de carga en los días involucrados sea la menor.
        3. Actualizar la carga acumulada por día.

    Esta heurística produce asignaciones bastante uniformes y evita el caso
    degenerado en que todos los clientes terminan asignados al mismo día.

    Returns:
        Diccionario {customer_id: pattern}.
    """
    # Carga acumulada por día (se actualiza durante la asignación).
    day_load: Dict[int, float] = {d: 0.0 for d in range(1, instance.horizon + 1)}
    assignment: Dict[int, Tuple[int, ...]] = {}

    # Ordenar clientes por demanda total descendente.
    sorted_customers = sorted(
        instance.customers,
        key=lambda c: c.demand * c.frequency,
        reverse=True,
    )

    for customer in sorted_customers:
        # Para cada patrón permitido, calcular el "costo" de asignar:
        # = máxima carga resultante en cualquiera de los días involucrados.
        # Minimizar este máximo distribuye uniformemente.
        best_pattern = None
        best_score = float("inf")
        for pattern in customer.allowed_patterns:
            max_load_after = max(
                day_load[day] + customer.demand for day in pattern
            )
            if max_load_after < best_score:
                best_score = max_load_after
                best_pattern = pattern

        assignment[customer.id] = best_pattern
        for day in best_pattern:
            day_load[day] += customer.demand

    return assignment


def _customers_per_day(
    instance: Instance, patterns: Dict[int, Tuple[int, ...]]
) -> Dict[int, List[int]]:
    """
    Construye, para cada día del horizonte, la lista de clientes que deben
    ser visitados ese día según la asignación de patrones.
    """
    per_day: Dict[int, List[int]] = {d: [] for d in range(1, instance.horizon + 1)}
    for c_id, pattern in patterns.items():
        for day in pattern:
            per_day[day].append(c_id)
    return per_day


def _nearest_feasible_customer(
    current_idx: int,
    pending: Set[int],
    remaining_capacity: float,
    instance: Instance,
    distance_matrix: np.ndarray,
    id_to_idx: Dict[int, int],
) -> Optional[int]:
    """
    Busca el cliente pendiente más cercano cuya demanda no exceda la
    capacidad restante del vehículo.

    Returns:
        ID del cliente seleccionado, o `None` si ninguno cabe.
    """
    best_id: Optional[int] = None
    best_dist = float("inf")

    for c_id in pending:
        customer = instance.get_customer(c_id)
        if customer.demand > remaining_capacity:
            continue
        c_idx = id_to_idx[c_id]
        d = distance_matrix[current_idx, c_idx]
        if d < best_dist:
            best_dist = d
            best_id = c_id

    return best_id


def greedy_solve(instance: Instance) -> Solution:
    """
    Resuelve el PVRP mediante la heurística Greedy del vecino más cercano.

    El algoritmo procede en tres fases:

        1. Asignación de patrones balanceada (`_assign_patterns`).
        2. Construcción de rutas día a día con vecino más cercano.
        3. Fase de reparación: si algún cliente quedó sin servir en su día
           asignado por falta de capacidad, se intenta reubicarlo en otro
           día compatible con sus patrones permitidos.

    Para instancias **muy apretadas** (uso de capacidad cercano al 100%),
    se prueban varias estrategias de ordenamiento de los clientes en la
    fase 2 y se devuelve la mejor solución factible encontrada. Si ninguna
    estrategia produce solución factible, se devuelve la de menor costo
    (que la `is_feasible` reportará como infactible).

    Args:
        instance: Instancia del PVRP.

    Returns:
        Una `Solution`. Idealmente factible; si no, la mejor encontrada.
    """
    # Conjunto de estrategias a probar (variar el orden de los clientes
    # pendientes en cada día cambia la estructura del empaque y a veces
    # libera capacidad para clientes "difíciles").
    strategies = ["nearest", "demand_desc", "demand_asc", "far_first"]

    best_feasible: Optional[Solution] = None
    best_feasible_cost = float("inf")
    best_overall: Optional[Solution] = None
    best_overall_cost = float("inf")

    for strategy in strategies:
        sol = _greedy_single_pass(instance, strategy=strategy)
        cost = sol.total_cost(instance)
        feasible, _ = sol.is_feasible(instance)

        if feasible and cost < best_feasible_cost:
            best_feasible = sol
            best_feasible_cost = cost
        if cost < best_overall_cost:
            best_overall = sol
            best_overall_cost = cost

    return best_feasible if best_feasible is not None else best_overall


def _greedy_single_pass(
    instance: Instance, strategy: str = "nearest"
) -> Solution:
    """
    Una pasada del algoritmo Greedy con una estrategia de ordenamiento dada.

    Args:
        instance: Instancia del PVRP.
        strategy: Estrategia de elección del próximo cliente dentro del día:
            - "nearest"     : el cliente más cercano que quepa.
            - "demand_desc" : el cliente de mayor demanda que quepa.
            - "demand_asc"  : el cliente de menor demanda que quepa.
            - "far_first"   : el más lejano que quepa (libera capacidad para
                              clientes pequeños después).
    """
    distance_matrix = build_distance_matrix(instance)
    id_to_idx = build_id_to_index_map(instance)

    patterns = _assign_patterns(instance)
    pending_by_day = _customers_per_day(instance, patterns)

    solution = Solution()
    day_used_capacity: Dict[int, float] = {
        d: 0.0 for d in range(1, instance.horizon + 1)
    }
    unassigned: List[int] = []

    for day in range(1, instance.horizon + 1):
        pending: Set[int] = set(pending_by_day[day])
        vehicle_id = 1

        while pending and vehicle_id <= instance.num_vehicles:
            route_nodes: List[int] = [0]
            current_idx = 0
            remaining_capacity = instance.capacity

            while True:
                next_id = _select_next_customer(
                    current_idx, pending, remaining_capacity,
                    instance, distance_matrix, id_to_idx, strategy,
                )
                if next_id is None:
                    break

                customer = instance.get_customer(next_id)
                route_nodes.append(next_id)
                current_idx = id_to_idx[next_id]
                remaining_capacity -= customer.demand
                day_used_capacity[day] += customer.demand
                pending.remove(next_id)

            route_nodes.append(0)
            if len(route_nodes) > 2:
                solution.add_route(Route(
                    day=day, vehicle_id=vehicle_id, nodes=route_nodes,
                ))
            vehicle_id += 1

        unassigned.extend(pending)

    if unassigned:
        solution = _repair_unassigned(
            solution, unassigned, patterns, day_used_capacity,
            instance, distance_matrix, id_to_idx,
        )

    return solution


def _select_next_customer(
    current_idx: int,
    pending: Set[int],
    remaining_capacity: float,
    instance: Instance,
    distance_matrix: np.ndarray,
    id_to_idx: Dict[int, int],
    strategy: str,
) -> Optional[int]:
    """Selecciona el siguiente cliente según la estrategia indicada."""
    # Filtrar candidatos factibles (que quepan en capacidad).
    candidates = [
        c_id for c_id in pending
        if instance.get_customer(c_id).demand <= remaining_capacity
    ]
    if not candidates:
        return None

    if strategy == "nearest":
        return min(
            candidates,
            key=lambda c: distance_matrix[current_idx, id_to_idx[c]],
        )
    elif strategy == "demand_desc":
        return max(candidates, key=lambda c: instance.get_customer(c).demand)
    elif strategy == "demand_asc":
        return min(candidates, key=lambda c: instance.get_customer(c).demand)
    elif strategy == "far_first":
        return max(
            candidates,
            key=lambda c: distance_matrix[current_idx, id_to_idx[c]],
        )
    else:
        raise ValueError(f"Estrategia desconocida: {strategy}")


def _repair_unassigned(
    solution: Solution,
    unassigned: List[int],
    patterns: Dict[int, Tuple[int, ...]],
    day_used_capacity: Dict[int, float],
    instance: Instance,
    distance_matrix: np.ndarray,
    id_to_idx: Dict[int, int],
) -> Solution:
    """
    Intenta reubicar clientes que no entraron en su día asignado.

    Estrategia: para cada cliente sin servir, busca otro día permitido por
    sus patrones donde haya capacidad sobrante y un vehículo disponible.
    Si encuentra uno, lo inserta abriendo una nueva ruta en ese día.

    Esta fase es una salvaguarda; en instancias bien dimensionadas no se
    activa o procesa muy pocos clientes.
    """
    # Contar cuántos vehículos están ya usados por día.
    vehicles_used: Dict[int, int] = {
        d: 0 for d in range(1, instance.horizon + 1)
    }
    for r in solution.routes:
        vehicles_used[r.day] = max(vehicles_used[r.day], r.vehicle_id)

    for c_id in unassigned:
        customer = instance.get_customer(c_id)

        # Buscar un día compatible con capacidad y vehículo disponibles.
        # Considerar todos los patrones permitidos, no solo el asignado.
        candidate_days = set()
        for pattern in customer.allowed_patterns:
            candidate_days.update(pattern)

        # Ordenar candidatos por menor uso de capacidad (más espacio libre).
        sorted_days = sorted(
            candidate_days,
            key=lambda d: day_used_capacity[d],
        )

        placed = False
        for day in sorted_days:
            total_cap = instance.capacity * instance.num_vehicles
            if (
                day_used_capacity[day] + customer.demand <= total_cap
                and vehicles_used[day] < instance.num_vehicles
            ):
                # Abrir una nueva ruta solo para este cliente.
                new_vehicle_id = vehicles_used[day] + 1
                solution.add_route(Route(
                    day=day,
                    vehicle_id=new_vehicle_id,
                    nodes=[0, c_id, 0],
                ))
                day_used_capacity[day] += customer.demand
                vehicles_used[day] = new_vehicle_id
                placed = True
                break

        # Si tampoco se pudo reubicar, la instancia es infactible para
        # esta heurística (lo reportará is_feasible).

    return solution
