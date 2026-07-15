"""
Representa el estado dinámico del PVRP como MDP y lo codifica en un vector
NumPy para la red del agente de RL: posición actual, capacidad restante, día,
vehículo, y por cada cliente su demanda, visitas pendientes, distancia y si
puede visitarse hoy según sus patrones aún viables.

Entrada: una Instance (src/data/instance.py) para construir el StateEncoder;
luego, un PVRPState para initial_state()/encode()/update_viable_patterns().
Salida: un PVRPState inicial, un vector numpy normalizado en [0, 1]
(state_dim = 4 + num_customers * 4), y actualizaciones en sitio del estado.
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


# Estado mutable de un episodio: posición, capacidad, día/vehículo activo y visitas/patrones por cliente.
@dataclass
class PVRPState:
    current_position: int = 0
    remaining_capacity: float = 0.0
    current_day: int = 1
    current_vehicle: int = 1
    visits_completed: Dict[int, List[int]] = field(default_factory=dict)
    viable_patterns: Dict[int, Set[Tuple[int, ...]]] = field(default_factory=dict)
    visited_today: Set[int] = field(default_factory=set)


# Codifica un PVRPState en el vector de observación que consume la red del agente.
class StateEncoder:

    def __init__(self, instance: Instance):
        self.instance = instance
        self.num_customers = instance.num_customers
        self.horizon = instance.horizon
        self.capacity = instance.capacity

        self.distance_matrix = build_distance_matrix(instance)
        self.id_to_idx = build_id_to_index_map(instance)
        self.idx_to_id = {idx: cid for cid, idx in self.id_to_idx.items()}

        self.max_distance = float(self.distance_matrix.max())
        if self.max_distance == 0.0:
            self.max_distance = 1.0

        self.customer_ids: List[int] = [c.id for c in instance.customers]

        self.state_dim: int = (
            NUM_GLOBAL_FEATURES
            + self.num_customers * NUM_CUSTOMER_FEATURES
        )

    # Construye el estado inicial: vehículo 1, día 1, capacidad llena, sin visitas.
    def initial_state(self) -> PVRPState:
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

    # Convierte el estado dinámico en el vector NumPy normalizado que ve la red.
    def encode(self, state: PVRPState) -> np.ndarray:
        vec = np.zeros(self.state_dim, dtype=np.float32)

        n_nodes = self.instance.num_nodes
        vec[0] = state.current_position / max(n_nodes - 1, 1)
        vec[1] = state.remaining_capacity / self.capacity
        vec[2] = state.current_day / self.horizon
        vec[3] = state.current_vehicle / max(self.instance.num_vehicles, 1)

        offset = NUM_GLOBAL_FEATURES
        cust_by_id = {c.id: c for c in self.instance.customers}

        for slot, cid in enumerate(self.customer_ids):
            customer = cust_by_id[cid]
            base = offset + slot * NUM_CUSTOMER_FEATURES

            vec[base + 0] = customer.demand / self.capacity

            visits_done = len(state.visits_completed.get(cid, []))
            remaining = customer.frequency - visits_done
            vec[base + 1] = max(remaining, 0) / customer.frequency

            cust_idx = self.id_to_idx[cid]
            dist = self.distance_matrix[state.current_position, cust_idx]
            vec[base + 2] = dist / self.max_distance

            vec[base + 3] = float(
                self._is_viable_today(state, cid)
            )

        return vec

    # True si algún patrón aún viable del cliente admite visitarlo en el día actual.
    def _is_viable_today(self, state: PVRPState, customer_id: int) -> bool:
        if customer_id in state.visited_today:
            return False

        visits = set(state.visits_completed.get(customer_id, []))
        today = state.current_day

        for pattern in state.viable_patterns.get(customer_id, set()):
            days_in_pattern = set(pattern)
            if visits.issubset(days_in_pattern) and today in days_in_pattern:
                return True
        return False

    # Reduce en sitio los patrones viables del cliente tras una nueva visita.
    def update_viable_patterns(self, state: PVRPState, customer_id: int) -> None:
        visits = set(state.visits_completed[customer_id])
        state.viable_patterns[customer_id] = {
            p for p in state.viable_patterns[customer_id]
            if visits.issubset(p)
        }
