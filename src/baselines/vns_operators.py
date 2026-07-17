"""
Operadores de vecindad usados por la búsqueda local del VNS. Cada uno devuelve
una nueva Solution con la primera mejora encontrada, o None si no hay ninguna.

Los cuatro primeros operadores actúan dentro de un mismo día y no alteran la
asignación de patrones. El quinto, relocate_between_days, sí la altera: es el
único de la búsqueda local que explora la dimensión periódica del problema, y
el único que hace algo cuando la instancia tiene un solo vehículo por día.

A diferencia de una búsqueda local que solo admite movimientos factibles, la
mejora se evalúa sobre el objetivo penalizado

    f(x) = distancia(x) + w * exceso_de_capacidad(x)

donde w es el peso de penalización que el VNS ajusta dinámicamente. Esto
permite atravesar soluciones infactibles durante la búsqueda, siguiendo el
esquema de penalizaciones de Hemmelmayr et al. (2009). La capacidad deja de
ser una barrera dura en los operadores: un movimiento que la excede se acepta
solo si el ahorro en distancia supera al costo penalizado del exceso.

Entrada: una Solution actual, la Instance, la matriz de distancias, el mapa
id->índice, el mapa id->demanda y el peso de penalización vigente.
Salida: una nueva Solution con menor costo penalizado, o None si el operador
no encuentra ningún movimiento mejorante.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional

import numpy as np

from src.data.instance import Instance
from src.utils.solution import Route, Solution


# Costo de una ruta dada como lista de IDs (incluye los depósitos 0).
def _route_cost_from_nodes(
    nodes: List[int],
    matrix: np.ndarray,
    id_to_idx: dict,
) -> float:
    cost = 0.0
    for i in range(len(nodes) - 1):
        cost += matrix[id_to_idx[nodes[i]], id_to_idx[nodes[i + 1]]]
    return cost


# Demanda total de los clientes en una ruta.
def _route_load(nodes: List[int], demands: Dict[int, float]) -> float:
    return sum(demands[c] for c in nodes if c != 0)


# Exceso de capacidad de una ruta: cuánta carga sobra por sobre Q.
def _excess(load: float, capacity: float) -> float:
    return load - capacity if load > capacity else 0.0


# Exceso de capacidad total de una solución, sumado sobre todas sus rutas.
def total_capacity_excess(
    solution: Solution,
    instance: Instance,
    demands: Dict[int, float],
) -> float:
    return sum(
        _excess(_route_load(r.nodes, demands), instance.capacity)
        for r in solution.routes
    )


# Objetivo penalizado: distancia recorrida más el exceso de capacidad ponderado.
def penalized_cost(
    solution: Solution,
    instance: Instance,
    demands: Dict[int, float],
    weight: float,
) -> float:
    return (
        solution.total_cost(instance)
        + weight * total_capacity_excess(solution, instance, demands)
    )


# 2-opt: invierte segmentos contiguos de cada ruta buscando reducir su distancia.
# No altera las cargas, de modo que el exceso de capacidad no cambia.
def two_opt_within_route(
    solution: Solution,
    instance: Instance,
    matrix: np.ndarray,
    id_to_idx: dict,
    demands: Dict[int, float],
    weight: float,
) -> Optional[Solution]:
    for r_idx, route in enumerate(solution.routes):
        nodes = route.nodes
        n = len(nodes)
        if n < 5:
            continue

        original_cost = _route_cost_from_nodes(nodes, matrix, id_to_idx)

        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                new_nodes = nodes[:i] + nodes[i:j + 1][::-1] + nodes[j + 1:]
                new_cost = _route_cost_from_nodes(new_nodes, matrix, id_to_idx)
                if new_cost < original_cost - 1e-9:
                    new_sol = copy.deepcopy(solution)
                    new_sol.routes[r_idx] = Route(
                        day=route.day,
                        vehicle_id=route.vehicle_id,
                        nodes=new_nodes,
                    )
                    return new_sol
    return None


# Intercambia dos clientes dentro de la misma ruta.
# No altera las cargas, de modo que el exceso de capacidad no cambia.
def swap_within_route(
    solution: Solution,
    instance: Instance,
    matrix: np.ndarray,
    id_to_idx: dict,
    demands: Dict[int, float],
    weight: float,
) -> Optional[Solution]:
    for r_idx, route in enumerate(solution.routes):
        nodes = route.nodes
        n = len(nodes)
        if n < 5:
            continue

        original_cost = _route_cost_from_nodes(nodes, matrix, id_to_idx)

        for i in range(1, n - 1):
            for j in range(i + 1, n - 1):
                new_nodes = list(nodes)
                new_nodes[i], new_nodes[j] = new_nodes[j], new_nodes[i]
                new_cost = _route_cost_from_nodes(new_nodes, matrix, id_to_idx)
                if new_cost < original_cost - 1e-9:
                    new_sol = copy.deepcopy(solution)
                    new_sol.routes[r_idx] = Route(
                        day=route.day,
                        vehicle_id=route.vehicle_id,
                        nodes=new_nodes,
                    )
                    return new_sol
    return None


# Mueve un cliente de una ruta a otra del mismo día. La capacidad no se impone
# como barrera dura: el exceso entra al objetivo penalizado.
def relocate_between_routes(
    solution: Solution,
    instance: Instance,
    matrix: np.ndarray,
    id_to_idx: dict,
    demands: Dict[int, float],
    weight: float,
) -> Optional[Solution]:
    capacity = instance.capacity
    days = sorted(set(r.day for r in solution.routes))

    for day in days:
        routes_day_idx = [
            i for i, r in enumerate(solution.routes) if r.day == day
        ]
        if len(routes_day_idx) < 2:
            continue

        for src_idx in routes_day_idx:
            src = solution.routes[src_idx]
            src_nodes = src.nodes
            src_cost = _route_cost_from_nodes(src_nodes, matrix, id_to_idx)
            src_load = _route_load(src_nodes, demands)
            src_excess = _excess(src_load, capacity)

            for pos_src in range(1, len(src_nodes) - 1):
                customer_id = src_nodes[pos_src]
                customer_demand = demands[customer_id]

                new_src_nodes = src_nodes[:pos_src] + src_nodes[pos_src + 1:]
                new_src_cost = _route_cost_from_nodes(new_src_nodes, matrix, id_to_idx)
                delta_src_excess = (
                    _excess(src_load - customer_demand, capacity) - src_excess
                )

                for dst_idx in routes_day_idx:
                    if dst_idx == src_idx:
                        continue
                    dst = solution.routes[dst_idx]
                    dst_cost = _route_cost_from_nodes(dst.nodes, matrix, id_to_idx)
                    dst_load = _route_load(dst.nodes, demands)
                    delta_dst_excess = (
                        _excess(dst_load + customer_demand, capacity)
                        - _excess(dst_load, capacity)
                    )
                    delta_penalty = weight * (delta_src_excess + delta_dst_excess)

                    for pos_dst in range(1, len(dst.nodes)):
                        new_dst_nodes = (
                            dst.nodes[:pos_dst]
                            + [customer_id]
                            + dst.nodes[pos_dst:]
                        )
                        new_dst_cost = _route_cost_from_nodes(new_dst_nodes, matrix, id_to_idx)
                        delta_distance = (
                            (new_src_cost + new_dst_cost) - (src_cost + dst_cost)
                        )
                        if delta_distance + delta_penalty < -1e-9:
                            new_sol = copy.deepcopy(solution)
                            new_sol.routes[src_idx] = Route(
                                day=src.day,
                                vehicle_id=src.vehicle_id,
                                nodes=new_src_nodes,
                            )
                            new_sol.routes[dst_idx] = Route(
                                day=dst.day,
                                vehicle_id=dst.vehicle_id,
                                nodes=new_dst_nodes,
                            )
                            new_sol.routes = [
                                r for r in new_sol.routes
                                if len(r.nodes) > 2
                            ]
                            return new_sol
    return None


# Intercambia un cliente de una ruta con un cliente de otra ruta del mismo día.
# La capacidad no se impone como barrera dura: el exceso entra al objetivo.
def swap_between_routes(
    solution: Solution,
    instance: Instance,
    matrix: np.ndarray,
    id_to_idx: dict,
    demands: Dict[int, float],
    weight: float,
) -> Optional[Solution]:
    capacity = instance.capacity
    days = sorted(set(r.day for r in solution.routes))

    for day in days:
        routes_day_idx = [
            i for i, r in enumerate(solution.routes) if r.day == day
        ]
        if len(routes_day_idx) < 2:
            continue

        for i1 in range(len(routes_day_idx)):
            for i2 in range(i1 + 1, len(routes_day_idx)):
                idx_a, idx_b = routes_day_idx[i1], routes_day_idx[i2]
                route_a = solution.routes[idx_a]
                route_b = solution.routes[idx_b]
                cost_a = _route_cost_from_nodes(route_a.nodes, matrix, id_to_idx)
                cost_b = _route_cost_from_nodes(route_b.nodes, matrix, id_to_idx)
                load_a = _route_load(route_a.nodes, demands)
                load_b = _route_load(route_b.nodes, demands)
                excess_a = _excess(load_a, capacity)
                excess_b = _excess(load_b, capacity)

                for pa in range(1, len(route_a.nodes) - 1):
                    for pb in range(1, len(route_b.nodes) - 1):
                        c_a = route_a.nodes[pa]
                        c_b = route_b.nodes[pb]
                        d_a = demands[c_a]
                        d_b = demands[c_b]

                        delta_penalty = weight * (
                            _excess(load_a - d_a + d_b, capacity) - excess_a
                            + _excess(load_b - d_b + d_a, capacity) - excess_b
                        )

                        new_a = list(route_a.nodes)
                        new_b = list(route_b.nodes)
                        new_a[pa], new_b[pb] = c_b, c_a

                        new_cost_a = _route_cost_from_nodes(new_a, matrix, id_to_idx)
                        new_cost_b = _route_cost_from_nodes(new_b, matrix, id_to_idx)
                        delta_distance = (
                            (new_cost_a + new_cost_b) - (cost_a + cost_b)
                        )
                        if delta_distance + delta_penalty < -1e-9:
                            new_sol = copy.deepcopy(solution)
                            new_sol.routes[idx_a] = Route(
                                day=route_a.day,
                                vehicle_id=route_a.vehicle_id,
                                nodes=new_a,
                            )
                            new_sol.routes[idx_b] = Route(
                                day=route_b.day,
                                vehicle_id=route_b.vehicle_id,
                                nodes=new_b,
                            )
                            return new_sol
    return None


# Ubicación de cada cliente en la solución: id -> lista de (día, índice de ruta,
# posición en la ruta). Se construye de una pasada para no recorrer todas las
# rutas por cada cliente.
def _visit_map(solution: Solution) -> Dict[int, List[tuple]]:
    visits: Dict[int, List[tuple]] = {}
    for r_idx, route in enumerate(solution.routes):
        for pos, node in enumerate(route.nodes):
            if node != 0:
                visits.setdefault(node, []).append((route.day, r_idx, pos))
    return visits


# Variación del objetivo penalizado al sacar al cliente de la posición indicada.
def _removal_delta(
    solution: Solution,
    route_idx: int,
    pos: int,
    matrix: np.ndarray,
    id_to_idx: dict,
    demands: Dict[int, float],
    capacity: float,
    weight: float,
) -> float:
    nodes = solution.routes[route_idx].nodes
    c = nodes[pos]
    prev, nxt = nodes[pos - 1], nodes[pos + 1]
    delta_distance = (
        matrix[id_to_idx[prev], id_to_idx[nxt]]
        - matrix[id_to_idx[prev], id_to_idx[c]]
        - matrix[id_to_idx[c], id_to_idx[nxt]]
    )
    load = _route_load(nodes, demands)
    delta_excess = _excess(load - demands[c], capacity) - _excess(load, capacity)
    return delta_distance + weight * delta_excess


# Mejor inserción del cliente en un día, evaluada sobre el objetivo penalizado.
# Devuelve (delta, route_idx, pos), donde route_idx None indica abrir una ruta
# nueva, o None si el día no admite ninguna inserción.
def _best_insertion(
    solution: Solution,
    instance: Instance,
    day: int,
    customer_id: int,
    matrix: np.ndarray,
    id_to_idx: dict,
    demands: Dict[int, float],
    weight: float,
) -> Optional[tuple]:
    capacity = instance.capacity
    demand = demands[customer_id]
    best_delta = float("inf")
    best_route_idx: Optional[int] = None
    best_pos: Optional[int] = None
    found = False

    routes_day_idx = [i for i, r in enumerate(solution.routes) if r.day == day]

    for r_idx in routes_day_idx:
        nodes = solution.routes[r_idx].nodes
        load = _route_load(nodes, demands)
        penalty = weight * (
            _excess(load + demand, capacity) - _excess(load, capacity)
        )
        for pos in range(1, len(nodes)):
            prev, nxt = nodes[pos - 1], nodes[pos]
            delta = (
                matrix[id_to_idx[prev], id_to_idx[customer_id]]
                + matrix[id_to_idx[customer_id], id_to_idx[nxt]]
                - matrix[id_to_idx[prev], id_to_idx[nxt]]
                + penalty
            )
            if delta < best_delta:
                best_delta, best_route_idx, best_pos = delta, r_idx, pos
                found = True

    if len(routes_day_idx) < instance.num_vehicles:
        delta = (
            2.0 * matrix[id_to_idx[0], id_to_idx[customer_id]]
            + weight * _excess(demand, capacity)
        )
        if delta < best_delta:
            best_delta, best_route_idx, best_pos = delta, None, None
            found = True

    return (best_delta, best_route_idx, best_pos) if found else None


# Reubica a un cliente entre días: prueba cambiar su combinación de visitas por
# otra admisible, sacándolo de los días que abandona e insertándolo en los
# nuevos, y acepta el primer cambio que mejore el objetivo penalizado.
#
# Es el análogo determinista y sistemático del cambio de combinación que el
# shaking aplica al azar. A diferencia de relocate_between_routes y
# swap_between_routes, que solo operan entre rutas de un mismo día y por tanto
# no hacen nada cuando hay un único vehículo diario, este operador es el único
# de la búsqueda local que explora la dimensión periódica del problema.
def relocate_between_days(
    solution: Solution,
    instance: Instance,
    matrix: np.ndarray,
    id_to_idx: dict,
    demands: Dict[int, float],
    weight: float,
) -> Optional[Solution]:
    capacity = instance.capacity
    visits = _visit_map(solution)

    for customer in instance.customers:
        if len(customer.allowed_patterns) < 2:
            continue
        located = visits.get(customer.id)
        if not located:
            continue

        current_pattern = tuple(sorted(day for day, _, _ in located))
        pos_by_day = {day: (r_idx, pos) for day, r_idx, pos in located}

        for new_pattern in customer.allowed_patterns:
            if new_pattern == current_pattern:
                continue

            days_remove = set(current_pattern) - set(new_pattern)
            days_add = set(new_pattern) - set(current_pattern)
            if not days_remove and not days_add:
                continue

            # Los días que se abandonan y los que se agregan son disjuntos, de
            # modo que las variaciones son independientes y se suman.
            delta = 0.0
            viable = True

            for day in days_remove:
                if day not in pos_by_day:
                    viable = False
                    break
                r_idx, pos = pos_by_day[day]
                delta += _removal_delta(
                    solution, r_idx, pos, matrix, id_to_idx,
                    demands, capacity, weight,
                )
            if not viable:
                continue

            for day in sorted(days_add):
                best = _best_insertion(
                    solution, instance, day, customer.id,
                    matrix, id_to_idx, demands, weight,
                )
                if best is None:
                    viable = False
                    break
                delta += best[0]
            if not viable:
                continue

            if delta < -1e-9:
                new_sol = copy.deepcopy(solution)
                for day in days_remove:
                    _remove_customer_from_day(new_sol, customer.id, day)
                for day in sorted(days_add):
                    _apply_best_insertion(
                        new_sol, instance, day, customer.id,
                        matrix, id_to_idx, demands, weight,
                    )
                return new_sol

    return None


# Saca a un cliente de todas las rutas de un día; elimina las rutas que quedan vacías.
def _remove_customer_from_day(solution: Solution, customer_id: int, day: int) -> None:
    for route in solution.routes:
        if route.day == day and customer_id in route.nodes:
            route.nodes = [n for n in route.nodes if n != customer_id]
    solution.routes = [r for r in solution.routes if len(r.nodes) > 2]


# Inserta al cliente en el día indicado, en la posición elegida por _best_insertion.
def _apply_best_insertion(
    solution: Solution,
    instance: Instance,
    day: int,
    customer_id: int,
    matrix: np.ndarray,
    id_to_idx: dict,
    demands: Dict[int, float],
    weight: float,
) -> None:
    best = _best_insertion(
        solution, instance, day, customer_id, matrix, id_to_idx, demands, weight
    )
    if best is None:
        return
    _, route_idx, pos = best
    if route_idx is None:
        used = {r.vehicle_id for r in solution.routes if r.day == day}
        libres = [v for v in range(1, instance.num_vehicles + 1) if v not in used]
        if libres:
            solution.add_route(
                Route(day=day, vehicle_id=libres[0], nodes=[0, customer_id, 0])
            )
        return
    route = solution.routes[route_idx]
    route.nodes = route.nodes[:pos] + [customer_id] + route.nodes[pos:]


# Orden de aplicación de los operadores en la búsqueda local del VNS.
NEIGHBORHOOD_OPERATORS = [
    ("2-opt", two_opt_within_route),
    ("swap_within", swap_within_route),
    ("relocate", relocate_between_routes),
    ("swap_between", swap_between_routes),
    ("relocate_days", relocate_between_days),
]
