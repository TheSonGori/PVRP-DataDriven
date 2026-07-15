"""
Calcula la máscara de acciones válidas del entorno PVRP (Action Masking),
compatible con MaskablePPO (sb3-contrib): la acción 0 es "cerrar ruta" y las
acciones 1..N son "visitar cliente con índice de matriz i", válidas solo si
el cliente no fue visitado hoy, hay capacidad suficiente, le quedan visitas
pendientes y algún patrón aún viable permite visitarlo hoy.

Entrada: un PVRPState, una Instance y un StateEncoder (src/environment/state.py).
Salida: un array booleano de tamaño (num_customers + 1,) indicando qué
acciones son válidas en el estado actual.
"""

from __future__ import annotations

import numpy as np

from src.data.instance import Instance
from src.environment.state import PVRPState, StateEncoder


# Construye la máscara booleana de acciones válidas para el estado actual.
def compute_action_mask(
    state: PVRPState,
    instance: Instance,
    state_encoder: StateEncoder,
) -> np.ndarray:
    n_actions = instance.num_customers + 1
    mask = np.zeros(n_actions, dtype=bool)

    # Cerrar ruta: permitido salvo en el instante inicial absoluto del episodio.
    is_episode_start = (
        state.current_position == 0
        and state.current_day == 1
        and state.current_vehicle == 1
        and not state.visited_today
    )
    mask[0] = not is_episode_start

    for cust_idx in range(1, instance.num_customers + 1):
        customer_id = state_encoder.idx_to_id[cust_idx]
        customer = instance.get_customer(customer_id)

        if customer_id in state.visited_today:
            continue

        if customer.demand > state.remaining_capacity:
            continue

        if len(state.visits_completed.get(customer_id, [])) >= customer.frequency:
            continue

        if not state_encoder._is_viable_today(state, customer_id):
            continue

        mask[cust_idx] = True

    # Salvaguarda: si ninguna acción es válida, permitir cerrar ruta para no bloquear el episodio.
    if not mask.any():
        mask[0] = True

    return mask
