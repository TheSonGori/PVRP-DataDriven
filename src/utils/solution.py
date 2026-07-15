"""
Representa una solución del PVRP (conjunto de rutas) y ofrece las operaciones
para construirla, calcular su costo, validar su factibilidad y generar un
resumen estadístico.

Entrada: rutas (día, vehículo, secuencia de nodos) agregadas incrementalmente,
más una Instance (src/data/instance.py) contra la cual evaluar/validar.
Salida: costo total, tupla (es_factible, lista_de_violaciones) y un
diccionario de métricas (summary) usados para comparar métodos de resolución.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.data.instance import Instance
from src.utils.distance import build_distance_matrix, build_id_to_index_map


# Ruta de un vehículo en un día: secuencia de nodos que empieza y termina en el depósito.
@dataclass
class Route:
    day: int
    vehicle_id: int
    nodes: List[int]

    # IDs de clientes visitados (excluye el depósito).
    @property
    def customers(self) -> List[int]:
        return [n for n in self.nodes if n != 0]

    # Demanda total transportada en la ruta.
    def load(self, instance: Instance) -> float:
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


# Solución completa del PVRP: conjunto de rutas más el costo reportado (si viene de BKS).
@dataclass
class Solution:
    routes: List[Route] = field(default_factory=list)
    reported_cost: Optional[float] = None

    # Agrega una ruta a la solución (sin validar factibilidad).
    def add_route(self, route: Route) -> None:
        self.routes.append(route)

    # Costo total: suma de distancias euclidianas recorridas en todas las rutas.
    def total_cost(self, instance: Instance) -> float:
        matrix = build_distance_matrix(instance)
        id_to_idx = build_id_to_index_map(instance)

        total = 0.0
        for route in self.routes:
            total += self._route_cost(route, matrix, id_to_idx)
        return total

    # Costo de una sola ruta sumando distancias entre nodos consecutivos.
    @staticmethod
    def _route_cost(route: Route, matrix: np.ndarray, id_to_idx: dict) -> float:
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

    # Verifica depósito al inicio/fin, capacidad, frecuencia de visita y patrones válidos.
    def is_feasible(self, instance: Instance) -> Tuple[bool, List[str]]:
        violations: List[str] = []

        valid_ids = {0} | {c.id for c in instance.customers}

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

        for r in self.routes:
            if any(c not in valid_ids for c in r.customers):
                continue
            load = r.load(instance)
            if load > instance.capacity:
                violations.append(
                    f"Ruta (día {r.day}, veh {r.vehicle_id}) excede capacidad: "
                    f"{load} > {instance.capacity}."
                )

        visits_per_customer: Dict[int, List[int]] = {}
        for r in self.routes:
            for c_id in r.customers:
                if c_id in valid_ids:
                    visits_per_customer.setdefault(c_id, []).append(r.day)

        for customer in instance.customers:
            visited_days = sorted(visits_per_customer.get(customer.id, []))

            if len(visited_days) != customer.frequency:
                violations.append(
                    f"Cliente {customer.id}: visitado {len(visited_days)} veces, "
                    f"se requieren {customer.frequency}."
                )
                continue

            visited_tuple = tuple(visited_days)
            if visited_tuple not in customer.allowed_patterns:
                violations.append(
                    f"Cliente {customer.id}: patrón de visitas {visited_tuple} "
                    f"no está entre sus patrones permitidos."
                )

        return (len(violations) == 0, violations)

    # Rutas asociadas a un día dado.
    def routes_by_day(self, day: int) -> List[Route]:
        return [r for r in self.routes if r.day == day]

    # Días en que un cliente es visitado en esta solución.
    def customer_visit_days(self, customer_id: int) -> Tuple[int, ...]:
        days = []
        for r in self.routes:
            if customer_id in r.customers:
                days.append(r.day)
        return tuple(sorted(days))

    # Patrón permitido que coincide con los días visitados, o None si no coincide con ninguno.
    def get_assigned_pattern(
        self, customer_id: int, instance: Instance
    ) -> Optional[Tuple[int, ...]]:
        visited = self.customer_visit_days(customer_id)
        customer = instance.get_customer(customer_id)
        if visited in customer.allowed_patterns:
            return visited
        return None

    # Resumen estadístico: costo, rutas, cargas, factibilidad y gap opcional respecto a BKS.
    def summary(
        self,
        instance: Instance,
        bks_cost: Optional[float] = None,
    ) -> Dict[str, float]:
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
