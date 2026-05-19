"""
Representación del estado del Periodic Vehicle Routing Problem (PVRP) modelado
como Proceso de Decisión de Markov (MDP).

Este módulo implementa la traducción formal entre el modelo matemático del PVRP
(Sección 1.5.2 de la memoria) y la representación de estado utilizada por el
agente de Aprendizaje por Refuerzo (Sección 2.3 de la memoria).

Diseño del estado
-----------------

El estado se compone de dos partes:

    1. Variables globales del sistema (longitud fija):
        - current_position    : índice de nodo donde se encuentra el vehículo.
        - remaining_capacity  : capacidad restante normalizada por Q.
        - current_day         : día actual del horizonte, normalizado por T.
        - current_vehicle     : índice del vehículo actual en el día.

    2. Vector de información por cliente (longitud N × 4):
        Para cada cliente i ∈ {1, ..., N}:
            - demand_norm     : demanda del cliente / capacidad Q.
            - remaining_visits: visitas pendientes / frecuencia requerida.
            - distance_norm   : distancia desde la posición actual,
                                normalizada por la distancia máxima de la matriz.
            - pattern_ok_today: 1 si visitar al cliente hoy es consistente
                                con al menos uno de sus patrones aún viables;
                                0 en caso contrario.

Todas las variables son flotantes en [0, 1] (excepto current_position que es
un entero); esto facilita el entrenamiento de redes neuronales sin escalado
adicional.

Decisión metodológica clave
----------------------------

La asignación de patrones (Ecuación 2 del modelo matemático) no se representa
como una variable de decisión explícita en el estado. En cambio, emerge
implícitamente de la secuencia de visitas que ejecuta el agente: cada vez que
el agente visita a un cliente, eliminamos del conjunto de patrones viables
aquellos incompatibles con la decisión tomada. Esta es la materialización
concreta del enfoque de "vía media" descrito en la propuesta de solución
(Capítulo 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import numpy as np

from src.data.instance import Instance
from src.utils.distance import build_distance_matrix, build_id_to_index_map


# Número de variables globales en la cabecera del vector de estado.
NUM_GLOBAL_FEATURES = 4

# Número de variables por cliente.
NUM_CUSTOMER_FEATURES = 4


@dataclass
class PVRPState:
    """
    Estado dinámico del PVRP durante un episodio de RL.

    Esta clase mantiene la información mutable que cambia paso a paso durante
    la construcción de una solución. Los datos estáticos de la instancia
    (coordenadas, demandas, patrones permitidos) se acceden vía referencia a
    la instancia original — esta clase NO los duplica.

    Attributes:
        current_position: Índice de matriz del nodo actual (0 = depósito).
        remaining_capacity: Capacidad disponible en el vehículo activo.
        current_day: Día del horizonte en curso (1-indexed).
        current_vehicle: Identificador del vehículo activo en el día actual.
        visits_completed: Por cada ID de cliente, lista de días en que ya
            fue visitado durante el episodio.
        viable_patterns: Por cada ID de cliente, conjunto de patrones que
            siguen siendo consistentes con las visitas ya realizadas.
            Se reduce monotónicamente conforme el agente toma decisiones.
        visited_today: Conjunto de IDs de clientes ya visitados en el día
            actual (no se pueden volver a visitar el mismo día).
    """
    current_position: int = 0
    remaining_capacity: float = 0.0
    current_day: int = 1
    current_vehicle: int = 1
    visits_completed: Dict[int, List[int]] = field(default_factory=dict)
    viable_patterns: Dict[int, Set[Tuple[int, ...]]] = field(default_factory=dict)
    visited_today: Set[int] = field(default_factory=set)


class StateEncoder:
    """
    Encodifica el estado dinámico (`PVRPState`) en un vector NumPy plano,
    listo para ser consumido por la red neuronal del agente.

    El encoder precomputa estructuras dependientes solo de la instancia (matriz
    de distancias, máximo de distancia para normalización, mapeo ID↔índice)
    para evitar trabajo redundante en cada llamada a `encode()`.
    """

    def __init__(self, instance: Instance):
        """
        Args:
            instance: Instancia del PVRP. Su estructura define la dimensión y
                la semántica del vector de estado durante toda la vida del
                encoder (NO mezclar encoders entre instancias distintas).
        """
        self.instance = instance
        self.num_customers = instance.num_customers
        self.horizon = instance.horizon
        self.capacity = instance.capacity

        # Matriz de distancias e índices precomputados.
        self.distance_matrix = build_distance_matrix(instance)
        self.id_to_idx = build_id_to_index_map(instance)
        self.idx_to_id = {idx: cid for cid, idx in self.id_to_idx.items()}

        # Distancia máxima utilizada para normalizar las features de distancia.
        # Se evita dividir por cero en instancias degeneradas.
        self.max_distance = float(self.distance_matrix.max())
        if self.max_distance == 0.0:
            self.max_distance = 1.0

        # Lista ordenada de IDs de cliente (para iterar en orden estable).
        self.customer_ids: List[int] = [c.id for c in instance.customers]

        # Dimensión total del vector de estado.
        self.state_dim: int = (
            NUM_GLOBAL_FEATURES
            + self.num_customers * NUM_CUSTOMER_FEATURES
        )

    # =========================================================================
    #  Construcción del estado inicial
    # =========================================================================

    def initial_state(self) -> PVRPState:
        """
        Construye el estado inicial al comienzo de un episodio.

        El vehículo 1 del día 1 parte desde el depósito con capacidad llena.
        Ningún cliente ha sido visitado todavía y todos sus patrones permitidos
        siguen siendo viables.
        """
        state = PVRPState(
            current_position=0,
            remaining_capacity=self.capacity,
            current_day=1,
            current_vehicle=1,
            visits_completed={cid: [] for cid in self.customer_ids},
            viable_patterns={
                c.id: set(c.allowed_patterns) for c in self.instance.customers
            },
            visited_today=set(),
        )
        return state

    # =========================================================================
    #  Encodificación a vector NumPy
    # =========================================================================

    def encode(self, state: PVRPState) -> np.ndarray:
        """
        Codifica un estado dinámico en su representación vectorial.

        Args:
            state: Estado actual del MDP.

        Returns:
            Vector NumPy de tamaño `state_dim` con valores en [0, 1]
            (excepto current_position que es un entero pero también está
            normalizado por num_nodes).
        """
        vec = np.zeros(self.state_dim, dtype=np.float32)

        # --- Variables globales ---
        n_nodes = self.instance.num_nodes
        vec[0] = state.current_position / max(n_nodes - 1, 1)
        vec[1] = state.remaining_capacity / self.capacity
        vec[2] = state.current_day / self.horizon
        vec[3] = state.current_vehicle / max(self.instance.num_vehicles, 1)

        # --- Variables por cliente ---
        offset = NUM_GLOBAL_FEATURES
        cust_by_id = {c.id: c for c in self.instance.customers}

        for slot, cid in enumerate(self.customer_ids):
            customer = cust_by_id[cid]
            base = offset + slot * NUM_CUSTOMER_FEATURES

            # Feature 0: demanda normalizada por Q.
            vec[base + 0] = customer.demand / self.capacity

            # Feature 1: visitas restantes / frecuencia requerida.
            visits_done = len(state.visits_completed.get(cid, []))
            remaining = customer.frequency - visits_done
            vec[base + 1] = max(remaining, 0) / customer.frequency

            # Feature 2: distancia desde la posición actual, normalizada.
            cust_idx = self.id_to_idx[cid]
            dist = self.distance_matrix[state.current_position, cust_idx]
            vec[base + 2] = dist / self.max_distance

            # Feature 3: ¿es viable visitar a este cliente HOY dado los
            # patrones aún compatibles y el estado del día?
            vec[base + 3] = float(
                self._is_viable_today(state, cid)
            )

        return vec

    # =========================================================================
    #  Consultas auxiliares (usadas por el entorno)
    # =========================================================================

    def _is_viable_today(self, state: PVRPState, customer_id: int) -> bool:
        """
        Determina si el cliente puede ser visitado en el día actual.

        Un cliente es viable hoy si y solo si existe al menos un patrón
        permitido P ∈ viable_patterns[customer_id] tal que:

            - todos los días ya visitados están en P, y
            - el día actual está en P.

        Args:
            state: Estado actual.
            customer_id: ID del cliente consultado.

        Returns:
            True si visitar al cliente hoy es compatible con al menos un
            patrón aún viable.
        """
        if customer_id in state.visited_today:
            return False

        visits = set(state.visits_completed.get(customer_id, []))
        today = state.current_day

        for pattern in state.viable_patterns.get(customer_id, set()):
            days_in_pattern = set(pattern)
            if visits.issubset(days_in_pattern) and today in days_in_pattern:
                return True
        return False

    def update_viable_patterns(self, state: PVRPState, customer_id: int) -> None:
        """
        Actualiza el conjunto de patrones viables de un cliente tras una visita.

        Esta operación implementa la "emergencia implícita" de la asignación
        de patrones: en lugar de elegir un patrón antes de tiempo, restringimos
        progresivamente el conjunto de patrones compatibles con las decisiones
        ya tomadas.

        Args:
            state: Estado actual (se modifica en sitio).
            customer_id: ID del cliente recién visitado.
        """
        visits = set(state.visits_completed[customer_id])
        state.viable_patterns[customer_id] = {
            p for p in state.viable_patterns[customer_id]
            if visits.issubset(p)
        }
