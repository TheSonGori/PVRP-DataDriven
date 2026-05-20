"""
Operador de perturbación (shaking) para el VNS.

El shaking introduce diversificación rompiendo la solución actual de forma
controlada, con el objetivo de escapar de óptimos locales. La idea clave:
después del shaking, la búsqueda local (operadores de vecindad) puede
converger a un óptimo local distinto y, con suerte, mejor.

Estrategia implementada
-----------------------

Para cada nivel `k` de perturbación (1, 2, 3, ...):

    - Seleccionar aleatoriamente `k` pares de rutas del mismo día.
    - Para cada par, intercambiar dos clientes elegidos al azar.
    - Si el intercambio viola capacidad, se intenta otro par.

Niveles más altos = más perturbación. El bucle externo del VNS aumenta `k`
gradualmente cuando no hay mejora, y lo reinicia a 1 cuando encuentra una.
"""

from __future__ import annotations

import copy
import random
from typing import Optional

from src.data.instance import Instance
from src.utils.solution import Route, Solution


def shake(
    solution: Solution,
    instance: Instance,
    k: int = 1,
    rng: Optional[random.Random] = None,
) -> Solution:
    """
    Perturba la solución actual realizando `k` intercambios aleatorios entre
    rutas del mismo día.

    Args:
        solution: Solución a perturbar.
        instance: Instancia del PVRP (para verificar capacidad).
        k: Nivel de perturbación (número de intercambios a intentar).
        rng: Generador aleatorio para reproducibilidad.

    Returns:
        Una nueva solución perturbada. Si no se logran intercambios válidos,
        retorna una copia de la original.
    """
    if rng is None:
        rng = random.Random()

    new_sol = copy.deepcopy(solution)
    days = sorted(set(r.day for r in new_sol.routes))

    swaps_done = 0
    max_attempts = k * 10  # margen para intentos fallidos

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

        # Ejecutar el intercambio in-place
        r_a.nodes[pa], r_b.nodes[pb] = c_b, c_a
        swaps_done += 1

    return new_sol
