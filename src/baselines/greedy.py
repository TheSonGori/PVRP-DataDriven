"""
Heurística constructiva Greedy (vecino más cercano) para el PVRP: primero
asigna a cada cliente un patrón de visita balanceando la carga por día, luego
construye rutas diarias visitando siempre al cliente factible más conveniente
según la estrategia elegida, y por último intenta reubicar a los clientes que
no cupieron en su día asignado. Sirve como línea base frente a VNS y RL.

Entrada: una Instance (src/data/instance.py).
Salida: una Solution (src/utils/solution.py), idealmente factible; si ninguna
estrategia produce una solución factible, se devuelve la de menor costo.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.data.instance import Customer, Instance
from src.utils.distance import build_distance_matrix, build_id_to_index_map
from src.utils.solution import Route, Solution


# Asigna a cada cliente el patrón permitido que mejor balancea la carga entre días.
def _assign_patterns(instance: Instance) -> Dict[int, Tuple[int, ...]]:
    day_load: Dict[int, float] = {d: 0.0 for d in range(1, instance.horizon + 1)}
    assignment: Dict[int, Tuple[int, ...]] = {}

    sorted_customers = sorted(
        instance.customers,
        key=lambda c: c.demand * c.frequency,
        reverse=True,
    )

    for customer in sorted_customers:
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


# Lista, para cada día, los clientes que deben visitarse según los patrones asignados.
def _customers_per_day(
    instance: Instance, patterns: Dict[int, Tuple[int, ...]]
) -> Dict[int, List[int]]:
    per_day: Dict[int, List[int]] = {d: [] for d in range(1, instance.horizon + 1)}
    for c_id, pattern in patterns.items():
        for day in pattern:
            per_day[day].append(c_id)
    return per_day


# Cliente pendiente más cercano cuya demanda cabe en la capacidad restante.
def _nearest_feasible_customer(
    current_idx: int,
    pending: Set[int],
    remaining_capacity: float,
    instance: Instance,
    distance_matrix: np.ndarray,
    id_to_idx: Dict[int, int],
) -> Optional[int]:
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


# Prueba varias estrategias de construcción y devuelve la mejor solución factible encontrada.
def greedy_solve(instance: Instance) -> Solution:
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


# Ejecuta una pasada completa del Greedy con la estrategia de selección indicada.
def _greedy_single_pass(
    instance: Instance, strategy: str = "nearest"
) -> Solution:
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


# Elige el siguiente cliente a visitar según la estrategia (nearest/demand_desc/demand_asc/far_first).
def _select_next_customer(
    current_idx: int,
    pending: Set[int],
    remaining_capacity: float,
    instance: Instance,
    distance_matrix: np.ndarray,
    id_to_idx: Dict[int, int],
    strategy: str,
) -> Optional[int]:
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


# Intenta reubicar en otro día compatible a los clientes que no entraron en su día asignado.
def _repair_unassigned(
    solution: Solution,
    unassigned: List[int],
    patterns: Dict[int, Tuple[int, ...]],
    day_used_capacity: Dict[int, float],
    instance: Instance,
    distance_matrix: np.ndarray,
    id_to_idx: Dict[int, int],
) -> Solution:
    vehicles_used: Dict[int, int] = {
        d: 0 for d in range(1, instance.horizon + 1)
    }
    for r in solution.routes:
        vehicles_used[r.day] = max(vehicles_used[r.day], r.vehicle_id)

    for c_id in unassigned:
        customer = instance.get_customer(c_id)

        candidate_days = set()
        for pattern in customer.allowed_patterns:
            candidate_days.update(pattern)

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

    return solution
