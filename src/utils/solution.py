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
    3. Calcular el costo total, validar factibilidad y producir resúmenes
       estadísticos para análisis y reporte en el Capítulo 4 de la memoria.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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

    def load(self, instance: Instance) -> float:
        """
        Demanda total transportada en esta ruta.

        Raises:
            KeyError: si la ruta contiene IDs no presentes en la instancia.
        """
        total = 0.0
        for c_id in self.customers:
            try:
                total += instance.get_customer(c_id).demand
            except KeyError:
                raise KeyError(
                    f"Cliente {c_id} en la ruta (día {self.day}, "
                    f"veh {self.vehicle_id}) no existe en la instancia."
                ) from None
        return total

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
    reported_cost: Optional[float] = None

    # =========================================================================
    #  Construcción incremental
    # =========================================================================

    def add_route(self, route: Route) -> None:
        """
        Agrega una ruta a la solución.

        Útil para algoritmos constructivos como Greedy o el agente de RL, que
        construyen la solución paso a paso. No realiza validación inmediata;
        la factibilidad se verifica con `is_feasible()`.
        """
        self.routes.append(route)

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
            try:
                idx_from = id_to_idx[route.nodes[i]]
                idx_to = id_to_idx[route.nodes[i + 1]]
            except KeyError as e:
                raise KeyError(
                    f"Nodo {e.args[0]} referenciado en la ruta "
                    f"(día {route.day}, veh {route.vehicle_id}) no existe en "
                    f"la instancia. La solución es inválida."
                ) from None
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

        # Construir el conjunto de IDs válidos en la instancia (incluyendo depósito).
        valid_ids = {0} | {c.id for c in instance.customers}

        # (a) Depósito al inicio y fin de cada ruta + (a') IDs desconocidos
        for r in self.routes:
            if not r.nodes or r.nodes[0] != 0 or r.nodes[-1] != 0:
                violations.append(
                    f"Ruta (día {r.day}, veh {r.vehicle_id}) no empieza/termina en depósito."
                )
            unknown = [n for n in r.nodes if n not in valid_ids]
            if unknown:
                violations.append(
                    f"Ruta (día {r.day}, veh {r.vehicle_id}) contiene IDs "
                    f"no presentes en la instancia: {unknown}."
                )

        # (b) Capacidad — saltar rutas con IDs desconocidos para no propagar el KeyError
        for r in self.routes:
            if any(c not in valid_ids for c in r.customers):
                continue
            load = r.load(instance)
            if load > instance.capacity:
                violations.append(
                    f"Ruta (día {r.day}, veh {r.vehicle_id}) excede capacidad: "
                    f"{load} > {instance.capacity}."
                )

        # (c) y (d): construir mapa cliente -> días visitados (solo IDs válidos)
        visits_per_customer: Dict[int, List[int]] = {}
        for r in self.routes:
            for c_id in r.customers:
                if c_id in valid_ids:
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
    #  Consultas y análisis
    # =========================================================================

    def routes_by_day(self, day: int) -> List[Route]:
        """Retorna todas las rutas asociadas al día indicado."""
        return [r for r in self.routes if r.day == day]

    def customer_visit_days(self, customer_id: int) -> Tuple[int, ...]:
        """
        Retorna los días en que un cliente es visitado en esta solución.

        Útil para visualización (qué días aparecen las rutas que contienen al
        cliente) y para identificar el patrón asignado.

        Args:
            customer_id: ID del cliente consultado.

        Returns:
            Tupla ordenada de días de visita. Vacía si el cliente no se
            visita en la solución actual.
        """
        days = []
        for r in self.routes:
            if customer_id in r.customers:
                days.append(r.day)
        return tuple(sorted(days))

    def get_assigned_pattern(
        self, customer_id: int, instance: Instance
    ) -> Optional[Tuple[int, ...]]:
        """
        Retorna el patrón asignado al cliente en esta solución, si el conjunto
        de días visitados coincide con alguno de sus patrones permitidos.

        Args:
            customer_id: ID del cliente.
            instance: Instancia del PVRP que define los patrones permitidos.

        Returns:
            La tupla de días asignada (igual a uno de los patrones permitidos),
            o `None` si los días visitados no corresponden a ningún patrón
            válido (lo cual indica infactibilidad).
        """
        visited = self.customer_visit_days(customer_id)
        customer = instance.get_customer(customer_id)
        if visited in customer.allowed_patterns:
            return visited
        return None

    def summary(
        self,
        instance: Instance,
        bks_cost: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Genera un resumen estadístico de la solución.

        Las métricas reportadas son las que se utilizarán en el Capítulo 4 de
        la memoria para comparar distintos métodos de resolución.

        Args:
            instance: Instancia del PVRP correspondiente.
            bks_cost: Costo de la mejor solución conocida (Best Known Solution).
                Si se provee, se calcula el gap porcentual.

        Returns:
            Diccionario con las métricas:
                - total_cost: costo total de la solución.
                - num_routes: número de rutas (vehículos-día utilizados).
                - avg_routes_per_day: rutas promedio por día.
                - avg_load: carga promedio por ruta.
                - max_load: carga máxima registrada en una ruta.
                - capacity_utilization: max_load / capacidad (en [0, 1]).
                - is_feasible: 1.0 si la solución es factible, 0.0 si no.
                - num_violations: número de violaciones detectadas.
                - gap_to_bks (opcional): (cost - bks) / bks * 100, en %.
        """
        cost = self.total_cost(instance)
        is_feasible, violations = self.is_feasible(instance)

        loads = [r.load(instance) for r in self.routes]
        avg_load = float(np.mean(loads)) if loads else 0.0
        max_load = float(np.max(loads)) if loads else 0.0

        result: Dict[str, float] = {
            "total_cost": cost,
            "num_routes": float(len(self.routes)),
            "avg_routes_per_day": len(self.routes) / instance.horizon,
            "avg_load": avg_load,
            "max_load": max_load,
            "capacity_utilization": max_load / instance.capacity if instance.capacity > 0 else 0.0,
            "is_feasible": 1.0 if is_feasible else 0.0,
            "num_violations": float(len(violations)),
        }

        if bks_cost is not None and bks_cost > 0:
            result["gap_to_bks"] = (cost - bks_cost) / bks_cost * 100.0

        return result

    def __repr__(self) -> str:
        return f"Solution(routes={len(self.routes)}, reported_cost={self.reported_cost})"
