"""
Define las estructuras de datos de una instancia del Periodic Vehicle Routing
Problem (PVRP): clientes, depósito e instancia completa.

Entradas: valores individuales (id, x, y, demanda, frecuencia, patrones de
visita, horizonte, número de vehículos, capacidad, etc.) usados para construir
Customer, Depot e Instance.
Salidas: objetos Customer, Depot e Instance ya construidos, listos para ser
usados por el resto del proyecto (entorno, baselines, agente).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# Cliente del PVRP: posición, demanda, frecuencia y patrones de visita permitidos.
@dataclass(frozen=True)
class Customer:
    id: int
    x: float
    y: float
    service_duration: float
    demand: float
    frequency: int
    allowed_patterns: Tuple[Tuple[int, ...], ...]

    # Devuelve (x, y) del cliente.
    @property
    def coords(self) -> Tuple[float, float]:
        return (self.x, self.y)


# Depósito central desde el cual parten todos los vehículos.
@dataclass(frozen=True)
class Depot:
    x: float
    y: float

    # Devuelve (x, y) del depósito.
    @property
    def coords(self) -> Tuple[float, float]:
        return (self.x, self.y)


# Instancia completa del PVRP: depósito, clientes, horizonte y flota.
@dataclass
class Instance:
    name: str
    depot: Depot
    customers: List[Customer]
    horizon: int
    num_vehicles: int
    capacity: float
    max_duration: float = 0.0

    # Índice id -> Customer, construido de forma perezosa la primera vez que se
    # consulta. No forma parte del constructor ni de la comparación: es solo
    # una caché para evitar el barrido lineal en get_customer.
    _customer_index: Dict[int, Customer] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    # Número de clientes (sin contar el depósito).
    @property
    def num_customers(self) -> int:
        return len(self.customers)

    # Número total de nodos (clientes + depósito).
    @property
    def num_nodes(self) -> int:
        return self.num_customers + 1

    # Busca y devuelve el cliente con el ID indicado; lanza KeyError si no existe.
    # Usa un índice cacheado: se reconstruye solo si cambió la cantidad de clientes.
    def get_customer(self, customer_id: int) -> Customer:
        if len(self._customer_index) != len(self.customers):
            self._customer_index = {c.id: c for c in self.customers}
        try:
            return self._customer_index[customer_id]
        except KeyError:
            raise KeyError(f"No existe un cliente con ID {customer_id}") from None

    def __repr__(self) -> str:
        return (
            f"Instance(name={self.name!r}, "
            f"customers={self.num_customers}, "
            f"horizon={self.horizon}, "
            f"vehicles={self.num_vehicles}, "
            f"capacity={self.capacity})"
        )
