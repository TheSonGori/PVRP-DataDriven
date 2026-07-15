"""
Operador de perturbación (shaking) del VNS: rompe la solución actual de
forma controlada intercambiando clientes entre rutas del mismo día, para que
la búsqueda local pueda escapar de óptimos locales y converger a uno distinto.

Entrada: una Solution, la Instance, un nivel de perturbación k y un
generador aleatorio opcional.
Salida: una nueva Solution perturbada (copia de la original si no se logra
ningún intercambio válido).
"""

from __future__ import annotations

import copy
import random
from typing import Optional

from src.data.instance import Instance
from src.utils.solution import Route, Solution


# Realiza k intercambios aleatorios de clientes entre rutas del mismo día, respetando capacidad.
def shake(
    solution: Solution,
    instance: Instance,
    k: int = 1,
    rng: Optional[random.Random] = None,
) -> Solution:
    if rng is None:
        rng = random.Random()

    new_sol = copy.deepcopy(solution)
    days = sorted(set(r.day for r in new_sol.routes))

    swaps_done = 0
    max_attempts = k * 10

    for _ in range(max_attempts):
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
        d_a = instance.get_customer(c_a).demand
        d_b = instance.get_customer(c_b).demand

        load_a = sum(instance.get_customer(c).demand for c in r_a.nodes if c != 0)
        load_b = sum(instance.get_customer(c).demand for c in r_b.nodes if c != 0)

        if (load_a - d_a + d_b > instance.capacity or
            load_b - d_b + d_a > instance.capacity):
            continue

        r_a.nodes[pa], r_b.nodes[pb] = c_b, c_a
        swaps_done += 1

    return new_sol
