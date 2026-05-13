"""
Representación de una solución del Periodic Vehicle Routing Problem (PVRP).

Una solución del PVRP es un conjunto de rutas que cumplen las restricciones del
problema (Sección 1.5.2 de la memoria):

    - Cada ruta comienza y termina en el depósito.
    - Cada cliente es visitado exactamente en los días estipulados por su
      patrón seleccionado.
    - La demanda total de cada ruta no excede la capacidad del vehículo.

Esta clase tiene tres propósitos:

    1. Encapsular soluciones generadas por cualquier método (RL, Greedy, VNS).
    2. Cargar soluciones de referencia (Best Known Solutions) del NEO Research
       Group desde archivos `.res`.
    3. Calcular el costo total y validar factibilidad de una solución dada
       una instancia.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from src.data.instance import Instance
from src.utils.distance import build_distance_matrix, build_id_to_index_map


@dataclass
class Route:
    """
    Una ruta corresponde al recorrido de un vehículo en un día determinado.

    Attributes:
        day: Día del horizonte de planificación (1-indexed) en que se ejecuta.
        vehicle_id: Identificador del vehículo que realiza la ruta.
        nodes: Secuencia de IDs de nodos visitados, comenzando y terminando
            en 0 (el depósito). Por ejemplo: [0, 8, 26, 31, 0].
    """
    day: int
    vehicle_id: int
    nodes: List[int]

    @property
    def customers(self) -> List[int]:
        """Retorna los IDs de los clientes visitados (excluye los depósitos)."""
        return [n for n in self.nodes if n != 0]

    def __repr__(self) -> str:
        return (
            f"Route(day={self.day}, vehicle={self.vehicle_id}, "
            f"customers={len(self.customers)})"
        )


@dataclass
class Solution:
    """
    Solución completa del PVRP, compuesta por un conjunto de rutas.

    Attributes:
        routes: Lista de rutas que componen la solución.
        reported_cost: Costo total declarado en el archivo de origen (si
            corresponde a una solución leída de disco). Es `None` para
            soluciones generadas por algoritmos.
    """
    routes: List[Route] = field(default_factory=list)
    reported_cost: float | None = None

    # =========================================================================
    #  Cálculo de costo
    # =========================================================================

    def total_cost(self, instance: Instance) -> float:
        """
        Calcula el costo total de la solución como la suma de las distancias
        euclidianas recorridas en todas las rutas.

        Corresponde a la función objetivo del modelo matemático (Ecuación 1,
        Sección 1.5.2 de la memoria):

            min  sum_{i,j,k,t} d_ij * x_ijkt

        Args:
            instance: Instancia del PVRP sobre la cual se evalúa la solución.

        Returns:
            Costo total acumulado (no negativo).
        """
        matrix = build_distance_matrix(instance)
        id_to_idx = build_id_to_index_map(instance)

        total = 0.0
        for route in self.routes:
            total += self._route_cost(route, matrix, id_to_idx)
        return total

    @staticmethod
    def _route_cost(route: Route, matrix: np.ndarray, id_to_idx: dict) -> float:
        """Calcula el costo de una sola ruta sumando distancias entre nodos consecutivos."""
        cost = 0.0
        for i in range(len(route.nodes) - 1):
            idx_from = id_to_idx[route.nodes[i]]
            idx_to = id_to_idx[route.nodes[i + 1]]
            cost += matrix[idx_from, idx_to]
        return cost

    # =========================================================================
    #  Validación de factibilidad
    # =========================================================================

    def is_feasible(self, instance: Instance) -> Tuple[bool, List[str]]:
        """
        Verifica si la solución cumple todas las restricciones del PVRP.

        Restricciones verificadas (referencias al modelo matemático en la
        Sección 1.5.2 de la memoria):

            (a) Cada ruta empieza y termina en el depósito.
            (b) Capacidad: la demanda total de cada ruta no excede Q (Ec. 6).
            (c) Frecuencia de visita: cada cliente es visitado exactamente
                `frequency` veces en el horizonte (Ec. 2-3).
            (d) Patrones válidos: el conjunto de días en que se visita un
                cliente corresponde a alguno de sus patrones permitidos
                (Ec. 2-3).

        Args:
            instance: Instancia contra la que se valida la solución.

        Returns:
            Tupla (es_factible, lista_de_violaciones). Si la solución es
            factible, la lista está vacía.
        """
        violations: List[str] = []

        # (a) Depósito al inicio y fin de cada ruta
        for r in self.routes:
            if not r.nodes or r.nodes[0] != 0 or r.nodes[-1] != 0:
                violations.append(
                    f"Ruta (día {r.day}, veh {r.vehicle_id}) no empieza/termina en depósito."
                )

        # (b) Capacidad
        for r in self.routes:
            load = sum(
                instance.get_customer(c).demand for c in r.customers
            )
            if load > instance.capacity:
                violations.append(
                    f"Ruta (día {r.day}, veh {r.vehicle_id}) excede capacidad: "
                    f"{load} > {instance.capacity}."
                )

        # (c) y (d): construir mapa cliente -> días visitados
        visits_per_customer: dict[int, List[int]] = {}
        for r in self.routes:
            for c_id in r.customers:
                visits_per_customer.setdefault(c_id, []).append(r.day)

        for customer in instance.customers:
            visited_days = sorted(visits_per_customer.get(customer.id, []))

            # (c) Frecuencia
            if len(visited_days) != customer.frequency:
                violations.append(
                    f"Cliente {customer.id}: visitado {len(visited_days)} veces, "
                    f"se requieren {customer.frequency}."
                )
                continue

            # (d) Patrón válido
            visited_tuple = tuple(visited_days)
            if visited_tuple not in customer.allowed_patterns:
                violations.append(
                    f"Cliente {customer.id}: patrón de visitas {visited_tuple} "
                    f"no está entre sus patrones permitidos."
                )

        return (len(violations) == 0, violations)

    # =========================================================================
    #  Utilidades
    # =========================================================================

    def routes_by_day(self, day: int) -> List[Route]:
        """Retorna todas las rutas asociadas al día indicado."""
        return [r for r in self.routes if r.day == day]

    def __repr__(self) -> str:
        return f"Solution(routes={len(self.routes)}, reported_cost={self.reported_cost})"
