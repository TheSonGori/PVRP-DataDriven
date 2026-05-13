"""
Representación de una instancia del Periodic Vehicle Routing Problem (PVRP).

Una instancia encapsula todos los datos necesarios para definir un problema:
el depósito, los clientes con sus demandas y patrones de visita, el horizonte
temporal y los parámetros de la flota.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class Customer:
    """
    Representa un cliente del PVRP.

    Attributes:
        id: Identificador único del cliente (entero positivo).
        x: Coordenada X en el plano euclidiano.
        y: Coordenada Y en el plano euclidiano.
        service_duration: Tiempo de servicio en el cliente (puede ser 0).
        demand: Demanda por visita (constante en todas las visitas del cliente).
        frequency: Número de visitas requeridas en el horizonte de planificación.
        allowed_patterns: Lista de patrones válidos de días de visita.
            Cada patrón es una tupla de días (1-indexed) en los que el cliente
            puede ser visitado. Por ejemplo, para un horizonte de 4 días y
            frecuencia 2, un patrón válido podría ser (1, 3) o (2, 4).
    """
    id: int
    x: float
    y: float
    service_duration: float
    demand: float
    frequency: int
    allowed_patterns: Tuple[Tuple[int, ...], ...]

    @property
    def coords(self) -> Tuple[float, float]:
        """Retorna las coordenadas del cliente como tupla (x, y)."""
        return (self.x, self.y)


@dataclass(frozen=True)
class Depot:
    """
    Representa el depósito central desde el cual parten todos los vehículos.

    Attributes:
        x: Coordenada X.
        y: Coordenada Y.
    """
    x: float
    y: float

    @property
    def coords(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class Instance:
    """
    Encapsula una instancia completa del Periodic Vehicle Routing Problem.

    Esta clase corresponde directamente al modelo matemático presentado en la
    Sección 1.5.2 de la memoria. Cada atributo se relaciona con un elemento
    formal del problema:

        - depot         <-->  nodo 0 del conjunto V
        - customers     <-->  conjunto V' = V \\ {0}
        - horizon       <-->  conjunto T = {1, ..., t}
        - num_vehicles  <-->  conjunto K
        - capacity      <-->  capacidad Q de cada vehículo

    Attributes:
        name: Identificador legible de la instancia (ej. "p01").
        depot: Depósito central.
        customers: Lista de clientes ordenada por ID.
        horizon: Número de días del horizonte de planificación (T).
        num_vehicles: Número de vehículos disponibles por día (|K|).
        capacity: Capacidad máxima de carga por vehículo (Q).
        max_duration: Duración máxima de una ruta por día (0 = sin límite).
    """
    name: str
    depot: Depot
    customers: List[Customer]
    horizon: int
    num_vehicles: int
    capacity: float
    max_duration: float = 0.0

    @property
    def num_customers(self) -> int:
        """Número total de clientes (excluyendo el depósito)."""
        return len(self.customers)

    @property
    def num_nodes(self) -> int:
        """Número total de nodos (clientes + depósito)."""
        return self.num_customers + 1

    def get_customer(self, customer_id: int) -> Customer:
        """
        Retorna el cliente con el ID indicado.

        Args:
            customer_id: ID del cliente buscado.

        Raises:
            KeyError: si no existe un cliente con ese ID.
        """
        for c in self.customers:
            if c.id == customer_id:
                return c
        raise KeyError(f"No existe un cliente con ID {customer_id}")

    def __repr__(self) -> str:
        return (
            f"Instance(name={self.name!r}, "
            f"customers={self.num_customers}, "
            f"horizon={self.horizon}, "
            f"vehicles={self.num_vehicles}, "
            f"capacity={self.capacity})"
        )
