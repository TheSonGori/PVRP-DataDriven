"""
Operadores de vecindad para el VNS aplicado al PVRP.

Cada operador toma una `Solution` y produce una nueva `Solution` (o `None`
si no hay mejora). Los operadores actúan **dentro de un mismo día**, es
decir, no modifican la asignación de patrones; solo redistribuyen visitas
entre rutas del mismo día.

Operadores implementados (estructuras de vecindad):

    1. swap_within_route     : intercambia dos clientes dentro de la misma ruta.
    2. swap_between_routes   : intercambia un cliente de una ruta con uno de otra
                               (mismo día).
    3. relocate_between_routes: mueve un cliente de una ruta a otra.
    4. two_opt_within_route  : invierte un segmento contiguo de una ruta.

Cada operador retorna una nueva solución mejorada si encuentra un movimiento
que reduce el costo total respetando la capacidad de los vehículos, o `None`
si no hay mejora factible.
"""

from __future__ import annotations

import copy
from typing import List, Optional

import numpy as np

from src.data.instance import Instance
from src.utils.solution import Route, Solution


def _route_cost_from_nodes(
    nodes: List[int],
    matrix: np.ndarray,
    id_to_idx: dict,
) -> float:
    """Costo de una ruta dada como lista de IDs (incluye depósitos 0)."""
    cost = 0.0
    for i in range(len(nodes) - 1):
        cost += matrix[id_to_idx[nodes[i]], id_to_idx[nodes[i + 1]]]
    return cost


def _route_load(nodes: List[int], instance: Instance) -> float:
    """Demanda total de los clientes en una ruta."""
    return sum(
        instance.get_customer(c).demand for c in nodes if c != 0
    )


# =============================================================================
#  Operador 1: 2-opt dentro de una ruta
# =============================================================================

def two_opt_within_route(
    solution: Solution,
    instance: Instance,
    matrix: np.ndarray,
    id_to_idx: dict,
) -> Optional[Solution]:
    """
    Aplica 2-opt a cada ruta: invierte segmentos contiguos buscando reducción
    de distancia. Es el operador clásico para mejorar el orden de visita
    dentro de una ruta.

    Retorna la primera mejora encontrada (estrategia *first improvement*).
    """
    for r_idx, route in enumerate(solution.routes):
        nodes = route.nodes
        n = len(nodes)
        if n < 5:
            # 2-opt requiere al menos 3 clientes para tener efecto
            # (rutas como [0, A, B, 0] no se mejoran).
            continue

        original_cost = _route_cost_from_nodes(nodes, matrix, id_to_idx)

        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                # Invertir el segmento nodes[i:j+1]
                new_nodes = nodes[:i] + nodes[i:j + 1][::-1] + nodes[j + 1:]
                new_cost = _route_cost_from_nodes(new_nodes, matrix, id_to_idx)
                if new_cost < original_cost - 1e-9:
                    # Mejora encontrada: construir nueva solución
                    new_sol = copy.deepcopy(solution)
                    new_sol.routes[r_idx] = Route(
                        day=route.day,
                        vehicle_id=route.vehicle_id,
                        nodes=new_nodes,
                    )
                    return new_sol
    return None


# =============================================================================
#  Operador 2: swap dentro de una ruta
# =============================================================================

def swap_within_route(
    solution: Solution,
    instance: Instance,
    matrix: np.ndarray,
    id_to_idx: dict,
) -> Optional[Solution]:
    """
    Intercambia dos clientes dentro de la misma ruta. Es un movimiento más
    local que 2-opt y a veces escapa de óptimos locales del 2-opt.
    """
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


# =============================================================================
#  Operador 3: relocate entre rutas (mismo día)
# =============================================================================

def relocate_between_routes(
    solution: Solution,
    instance: Instance,
    matrix: np.ndarray,
    id_to_idx: dict,
) -> Optional[Solution]:
    """
    Mueve un cliente de una ruta a otra (del mismo día), verificando que la
    capacidad no se viole. Útil para reequilibrar cargas entre vehículos.
    """
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

            for pos_src in range(1, len(src_nodes) - 1):
                customer_id = src_nodes[pos_src]
                customer_demand = instance.get_customer(customer_id).demand

                # Probar insertar en cada otra ruta del mismo día
                for dst_idx in routes_day_idx:
                    if dst_idx == src_idx:
                        continue
                    dst = solution.routes[dst_idx]
                    if _route_load(dst.nodes, instance) + customer_demand > instance.capacity:
                        continue

                    dst_cost = _route_cost_from_nodes(dst.nodes, matrix, id_to_idx)

                    # Probar todas las posiciones de inserción
                    for pos_dst in range(1, len(dst.nodes)):
                        new_src_nodes = src_nodes[:pos_src] + src_nodes[pos_src + 1:]
                        new_dst_nodes = (
                            dst.nodes[:pos_dst]
                            + [customer_id]
                            + dst.nodes[pos_dst:]
                        )
                        new_src_cost = _route_cost_from_nodes(new_src_nodes, matrix, id_to_idx)
                        new_dst_cost = _route_cost_from_nodes(new_dst_nodes, matrix, id_to_idx)
                        delta = (new_src_cost + new_dst_cost) - (src_cost + dst_cost)
                        if delta < -1e-9:
                            # Crear nueva solución; eliminar rutas vacías.
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
                            # Si la ruta origen quedó vacía ([0, 0]), eliminarla.
                            new_sol.routes = [
                                r for r in new_sol.routes
                                if len(r.nodes) > 2
                            ]
                            return new_sol
    return None


def swap_between_routes(
    solution: Solution,
    instance: Instance,
    matrix: np.ndarray,
    id_to_idx: dict,
) -> Optional[Solution]:
    """
    Intercambia un cliente de una ruta con un cliente de otra ruta del mismo día.

    Más potente que `relocate` porque considera intercambios bidireccionales y
    a menudo encuentra mejoras que `relocate` y `2-opt` no detectan.
    """
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
                load_a = _route_load(route_a.nodes, instance)
                load_b = _route_load(route_b.nodes, instance)

                for pa in range(1, len(route_a.nodes) - 1):
                    for pb in range(1, len(route_b.nodes) - 1):
                        c_a = route_a.nodes[pa]
                        c_b = route_b.nodes[pb]
                        d_a = instance.get_customer(c_a).demand
                        d_b = instance.get_customer(c_b).demand

                        # Verificar capacidad después del intercambio
                        if (load_a - d_a + d_b > instance.capacity or
                            load_b - d_b + d_a > instance.capacity):
                            continue

                        new_a = list(route_a.nodes)
                        new_b = list(route_b.nodes)
                        new_a[pa], new_b[pb] = c_b, c_a

                        new_cost_a = _route_cost_from_nodes(new_a, matrix, id_to_idx)
                        new_cost_b = _route_cost_from_nodes(new_b, matrix, id_to_idx)
                        delta = (new_cost_a + new_cost_b) - (cost_a + cost_b)
                        if delta < -1e-9:
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


# =============================================================================
#  Lista de operadores en el orden de aplicación del VNS
# =============================================================================

NEIGHBORHOOD_OPERATORS = [
    ("2-opt", two_opt_within_route),
    ("swap_within", swap_within_route),
    ("relocate", relocate_between_routes),
    ("swap_between", swap_between_routes),
]
