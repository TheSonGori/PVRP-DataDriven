"""
Máscara de acciones válidas para el entorno PVRP.

El enmascaramiento de acciones (Action Masking) traduce las restricciones del
modelo matemático del PVRP (Sección 1.5.2 de la memoria) en una señal directa
para el agente: en lugar de aprender por ensayo y error que una acción viola
una restricción dura, el agente recibe explícitamente la lista de acciones
permitidas en cada estado.

Esto:

    - Acelera el aprendizaje (el agente no malgasta capacidad explorando
      acciones imposibles).
    - Garantiza factibilidad por construcción durante el entrenamiento.
    - Es compatible directamente con MaskablePPO (sb3-contrib).

Reglas de validez para cada acción
-----------------------------------

Acción 0 (cerrar ruta):
    Siempre válida si no estamos al inicio del episodio. La política puede
    optar por "cerrar inmediatamente" un día sin visitar clientes si lo
    considera óptimo.

Acción i ∈ {1, ..., N} (visitar al cliente con índice de matriz i):
    Es válida si y solo si TODAS estas condiciones se cumplen:

    (a) El cliente NO ha sido visitado en el día actual (restricción
        implícita de ruteo: cada cliente se visita a lo sumo una vez por día).
    (b) La demanda del cliente NO excede la capacidad restante del vehículo
        activo (Ecuación 6 del modelo matemático).
    (c) El cliente aún tiene visitas pendientes (frecuencia no completada,
        Ecuación 2).
    (d) Existe al menos un patrón aún viable que incluya el día actual
        (Ecuación 3 — consistencia entre días visitados y patrón asignado).
"""

from __future__ import annotations

import numpy as np

from src.data.instance import Instance
from src.environment.state import PVRPState, StateEncoder


def compute_action_mask(
    state: PVRPState,
    instance: Instance,
    state_encoder: StateEncoder,
) -> np.ndarray:
    """
    Calcula la máscara de acciones válidas para el estado actual.

    Args:
        state: Estado dinámico actual del MDP.
        instance: Instancia del PVRP.
        state_encoder: Encoder que provee el mapeo índice-de-matriz ↔ cliente.

    Returns:
        Array booleano de tamaño (N + 1,), donde el elemento i indica si la
        acción i es válida en el estado actual.

            mask[0] = True/False  -> "cerrar ruta"
            mask[i] = True/False  -> "visitar cliente con índice de matriz i"
    """
    n_actions = instance.num_customers + 1
    mask = np.zeros(n_actions, dtype=bool)

    # ---------------------------------------------------------------------
    #  Acción 0: cerrar ruta. Permitida si no estamos al inicio del episodio.
    # ---------------------------------------------------------------------
    # El "inicio absoluto" es el único momento en que NO permitimos cerrar
    # (forzar al agente a tomar al menos una acción de visita).
    # Aceptamos cerrar incluso si el vehículo está en el depósito tras
    # una visita previa: el cierre será trivial (distancia 0).
    is_episode_start = (
        state.current_position == 0
        and state.current_day == 1
        and state.current_vehicle == 1
        and not state.visited_today
    )
    mask[0] = not is_episode_start

    # ---------------------------------------------------------------------
    #  Acciones 1..N: visitar al cliente con índice de matriz i.
    # ---------------------------------------------------------------------
    for cust_idx in range(1, instance.num_customers + 1):
        customer_id = state_encoder.idx_to_id[cust_idx]
        customer = instance.get_customer(customer_id)

        # (a) Ya visitado hoy
        if customer_id in state.visited_today:
            continue

        # (b) Capacidad insuficiente
        if customer.demand > state.remaining_capacity:
            continue

        # (c) Frecuencia ya completada
        if len(state.visits_completed.get(customer_id, [])) >= customer.frequency:
            continue

        # (d) Patrón inviable hoy
        if not state_encoder._is_viable_today(state, customer_id):
            continue

        mask[cust_idx] = True

    # ---------------------------------------------------------------------
    #  Salvaguarda: si por algún motivo ninguna acción es válida,
    #  permitir al menos cerrar ruta para que el episodio pueda avanzar.
    # ---------------------------------------------------------------------
    if not mask.any():
        mask[0] = True

    return mask
