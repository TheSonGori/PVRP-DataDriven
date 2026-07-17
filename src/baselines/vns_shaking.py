"""
Operadores de perturbación (shaking) del VNS. Siguiendo el esquema de
Hemmelmayr et al. (2009), los niveles de perturbación k se organizan en dos
familias de operadores, aplicadas en orden:

    k = 1..3  cambio de combinación de días de visita de 1 a k clientes
    k = 4..6  intercambio de 1 a (k-3) clientes entre rutas de un mismo día

La primera familia es la que modifica la dimensión periódica del problema
(qué días se visita a cada cliente); la segunda solo reordena clientes dentro
de un día ya fijado.

Ambas preservan siempre los patrones de visita permitidos y el número de
vehículos disponibles por día. La capacidad, en cambio, NO se impone como
barrera dura: siguiendo el esquema de penalizaciones de Hemmelmayr et al.
(2009), la perturbación puede producir soluciones que exceden la capacidad, y
ese exceso queda penalizado en el objetivo que guía la búsqueda
(src/baselines/vns.py). Permitir el paso por la región infactible es lo que
deja al VNS moverse entre óptimos locales factibles en instancias con
saturación alta.

Entrada: una Solution, la Instance, un nivel de perturbación k, un generador
aleatorio y, opcionalmente, la matriz de distancias y el mapa id->índice.
Salida: una nueva Solution perturbada, posiblemente infactible por
capacidad (copia de la original si no se logra aplicar ningún movimiento).
"""

from __future__ import annotations

import copy
import random
from typing import List, Optional, Tuple

import numpy as np

from src.data.instance import Customer, Instance
from src.utils.distance import build_distance_matrix, build_id_to_index_map
from src.utils.solution import Route, Solution


# Número de niveles de perturbación: 3 de cambio de días + 3 de intercambio.
K_MAX_DEFAULT = 6


# Costo de una ruta dada como lista de IDs (incluye los depósitos 0).
def _route_cost_from_nodes(nodes: List[int], matrix: np.ndarray, id_to_idx: dict) -> float:
    cost = 0.0
    for i in range(len(nodes) - 1):
        cost += matrix[id_to_idx[nodes[i]], id_to_idx[nodes[i + 1]]]
    return cost


# Demanda total de los clientes en una ruta.
def _route_load(nodes: List[int], demands: dict) -> float:
    return sum(demands[c] for c in nodes if c != 0)


# Saca a un cliente de todas las rutas de un día dado; elimina rutas que quedan vacías.
def _remove_customer_from_day(solution: Solution, customer_id: int, day: int) -> bool:
    removed = False
    for route in solution.routes:
        if route.day == day and customer_id in route.nodes:
            route.nodes = [n for n in route.nodes if n != customer_id]
            removed = True
    if removed:
        solution.routes = [r for r in solution.routes if len(r.nodes) > 2]
    return removed


# Inserta a un cliente en el día indicado, en la posición más barata en
# distancia. La capacidad NO se impone como barrera dura: un día saturado
# admite igualmente la inserción y el exceso resultante queda penalizado en el
# objetivo del VNS. El número de vehículos por día sí se respeta: solo se abre
# una ruta nueva si el día tiene un vehículo libre.
# Devuelve True si logró insertar; False si el día no tiene ni rutas ni
# vehículos disponibles.
def _insert_customer_in_day(
    solution: Solution,
    instance: Instance,
    demands: dict,
    customer_id: int,
    day: int,
    matrix: np.ndarray,
    id_to_idx: dict,
) -> bool:
    best_delta = float("inf")
    best_route_idx: Optional[int] = None
    best_pos: Optional[int] = None

    routes_day_idx = [i for i, r in enumerate(solution.routes) if r.day == day]

    for r_idx in routes_day_idx:
        route = solution.routes[r_idx]
        base_cost = _route_cost_from_nodes(route.nodes, matrix, id_to_idx)
        for pos in range(1, len(route.nodes)):
            new_nodes = route.nodes[:pos] + [customer_id] + route.nodes[pos:]
            delta = _route_cost_from_nodes(new_nodes, matrix, id_to_idx) - base_cost
            if delta < best_delta:
                best_delta = delta
                best_route_idx = r_idx
                best_pos = pos

    # Alternativa: abrir una ruta nueva, si el día todavía tiene vehículos libres.
    can_open_route = len(routes_day_idx) < instance.num_vehicles
    delta_new_route = float("inf")
    if can_open_route:
        delta_new_route = 2.0 * matrix[id_to_idx[0], id_to_idx[customer_id]]

    if best_route_idx is None and not can_open_route:
        return False

    if delta_new_route < best_delta:
        used = {r.vehicle_id for r in solution.routes if r.day == day}
        libres = [v for v in range(1, instance.num_vehicles + 1) if v not in used]
        if not libres:
            if best_route_idx is None:
                return False
        else:
            solution.add_route(
                Route(day=day, vehicle_id=libres[0], nodes=[0, customer_id, 0])
            )
            return True

    route = solution.routes[best_route_idx]
    route.nodes = route.nodes[:best_pos] + [customer_id] + route.nodes[best_pos:]
    return True


# Cambia la combinación de días de un cliente: lo saca de los días que abandona
# y lo inserta en los días nuevos. Devuelve False si algún día nuevo no admite
# la inserción (en ese caso el llamador descarta el intento completo).
def _apply_pattern_change(
    solution: Solution,
    instance: Instance,
    demands: dict,
    customer_id: int,
    current_pattern: Tuple[int, ...],
    new_pattern: Tuple[int, ...],
    matrix: np.ndarray,
    id_to_idx: dict,
) -> bool:
    days_to_remove = set(current_pattern) - set(new_pattern)
    days_to_add = set(new_pattern) - set(current_pattern)

    for day in days_to_remove:
        _remove_customer_from_day(solution, customer_id, day)

    for day in sorted(days_to_add):
        if not _insert_customer_in_day(
            solution, instance, demands, customer_id, day, matrix, id_to_idx
        ):
            return False

    return True


# Familia 1 de shaking: cambia la combinación de días de visita de 1 a k clientes.
def _shake_change_combination(
    solution: Solution,
    instance: Instance,
    demands: dict,
    k: int,
    rng: random.Random,
    matrix: np.ndarray,
    id_to_idx: dict,
) -> Solution:
    current_sol = copy.deepcopy(solution)
    n_target = rng.randint(1, k)

    candidates: List[Customer] = [
        c for c in instance.customers if len(c.allowed_patterns) > 1
    ]
    rng.shuffle(candidates)

    changed = 0
    for customer in candidates:
        if changed >= n_target:
            break

        current_pattern = current_sol.customer_visit_days(customer.id)
        options = [p for p in customer.allowed_patterns if p != current_pattern]
        if not options:
            continue
        new_pattern = options[rng.randrange(len(options))]

        trial = copy.deepcopy(current_sol)
        ok = _apply_pattern_change(
            trial, instance, demands, customer.id,
            current_pattern, new_pattern, matrix, id_to_idx,
        )
        if ok:
            current_sol = trial
            changed += 1

    return current_sol


# Familia 2 de shaking: k intercambios aleatorios de clientes entre rutas del
# mismo día. No modifica la asignación de días. La capacidad no se impone como
# barrera dura: el exceso queda penalizado en el objetivo del VNS.
def _shake_swap_within_day(
    solution: Solution,
    instance: Instance,
    demands: dict,
    k: int,
    rng: random.Random,
) -> Solution:
    new_sol = copy.deepcopy(solution)
    days = sorted(set(r.day for r in new_sol.routes))

    swaps_done = 0
    for _ in range(k * 10):
        if swaps_done >= k:
            break

        day = rng.choice(days)
        routes_day = [r for r in new_sol.routes if r.day == day]
        if len(routes_day) < 2:
            continue

        r_a, r_b = rng.sample(routes_day, 2)
        if len(r_a.nodes) < 3 or len(r_b.nodes) < 3:
            continue

        pa = rng.randrange(1, len(r_a.nodes) - 1)
        pb = rng.randrange(1, len(r_b.nodes) - 1)

        c_a, c_b = r_a.nodes[pa], r_b.nodes[pb]
        r_a.nodes[pa], r_b.nodes[pb] = c_b, c_a
        swaps_done += 1

    return new_sol


# Perturbación del VNS en el nivel k: despacha a la familia de operadores
# correspondiente según el esquema declarado al inicio del módulo.
def shake(
    solution: Solution,
    instance: Instance,
    k: int = 1,
    rng: Optional[random.Random] = None,
    matrix: Optional[np.ndarray] = None,
    id_to_idx: Optional[dict] = None,
) -> Solution:
    if rng is None:
        rng = random.Random()
    if matrix is None:
        matrix = build_distance_matrix(instance)
    if id_to_idx is None:
        id_to_idx = build_id_to_index_map(instance)

    demands = {c.id: c.demand for c in instance.customers}
    demands[0] = 0.0

    if k <= 3:
        return _shake_change_combination(
            solution, instance, demands, k, rng, matrix, id_to_idx
        )
    return _shake_swap_within_day(solution, instance, demands, k - 3, rng)
