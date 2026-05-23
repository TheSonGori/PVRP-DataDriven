"""
Entorno Gymnasium para el Periodic Vehicle Routing Problem (PVRP).

Implementa la formulación del PVRP como Proceso de Decisión de Markov según el
diseño descrito en el Capítulo 3 de la memoria, utilizando el enfoque de
"vía media" para la granularidad de las decisiones del agente:

    - Espacio de acciones discreto de tamaño N+1.
    - Acción 0 = "cerrar ruta actual y avanzar".
    - Acciones 1..N = "visitar al cliente con índice de matriz i".

La asignación de patrones de visita (Ecuación 2 del modelo matemático) NO
se modela como una decisión explícita: emerge implícitamente del orden en
que el agente elige visitar a cada cliente.

El entorno expone el método `action_masks()` para compatibilidad directa
con MaskablePPO de sb3-contrib (ver Día 10).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.data.instance import Instance
from src.environment.action_mask import compute_action_mask
from src.environment.reward import (
    DEFAULT_REWARD_CONFIG,
    RewardConfig,
    distance_reward,
    terminal_reward,
)
from src.environment.state import PVRPState, StateEncoder
from src.utils.solution import Route, Solution


class PVRPEnv(gym.Env):
    """Entorno Gymnasium para el PVRP con Action Masking."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        instance: Instance,
        reward_config: RewardConfig = DEFAULT_REWARD_CONFIG,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.instance = instance
        self.reward_config = reward_config
        self.state_encoder = StateEncoder(instance)

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.state_encoder.state_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.instance.num_customers + 1)

        # Estado interno
        self._state: Optional[PVRPState] = None
        self._current_route_nodes: list[int] = []
        self._solution: Optional[Solution] = None
        self._steps_taken: int = 0

        if seed is not None:
            self.reset(seed=seed)

    # =========================================================================
    #  API estándar de Gymnasium
    # =========================================================================

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self._state = self.state_encoder.initial_state()
        self._current_route_nodes = [0]
        self._solution = Solution()
        self._steps_taken = 0

        obs = self.state_encoder.encode(self._state)
        info = self._build_info()
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self._state is not None, "Debes llamar reset() antes de step()."
        self._steps_taken += 1

        # Bajo enmascaramiento, todas las acciones recibidas deben ser válidas.
        # Aun así verificamos por robustez (algoritmos no enmascarados podrían
        # llamar step() con acciones arbitrarias).
        mask = self.action_masks()
        if not mask[action]:
            obs = self.state_encoder.encode(self._state)
            return obs, -1.0, False, False, {"invalid_action": int(action)}

        # Ejecutar la acción
        if action == 0:
            reward, step_info = self._close_route()
        else:
            reward, step_info = self._visit_customer(action)

        terminated = self._is_episode_done()
        if terminated:
            self._finalize_open_route()
            feasible, _ = self._solution.is_feasible(self.instance)
            reward += terminal_reward(feasible, self.reward_config)
            step_info["is_feasible"] = feasible
            # Exponer métricas del episodio en info para callbacks de logging.
            # Se hace AQUÍ (antes del reset automático del VecEnv) para evitar
            # problemas de timing al consultar el entorno tras la terminación.
            step_info["episode_cost"] = self._solution.total_cost(self.instance)
            step_info["episode_num_routes"] = len(self._solution.routes)
            step_info["episode_feasible"] = 1.0 if feasible else 0.0

        obs = self.state_encoder.encode(self._state)
        info = {**self._build_info(), **step_info}
        return obs, float(reward), terminated, False, info

    # =========================================================================
    #  Acciones internas
    # =========================================================================

    def _visit_customer(self, action: int) -> Tuple[float, Dict[str, Any]]:
        """Ejecuta una visita (acción ya validada por la máscara)."""
        customer_id = self.state_encoder.idx_to_id[action]
        customer = self.instance.get_customer(customer_id)

        from_idx = self._state.current_position
        to_idx = action
        distance = self.state_encoder.distance_matrix[from_idx, to_idx]

        self._state.current_position = to_idx
        self._state.remaining_capacity -= customer.demand
        self._state.visits_completed[customer_id].append(self._state.current_day)
        self._state.visited_today.add(customer_id)
        self.state_encoder.update_viable_patterns(self._state, customer_id)
        self._current_route_nodes.append(customer_id)

        return distance_reward(distance), {
            "visited": customer_id,
            "distance": float(distance),
        }

    def _close_route(self) -> Tuple[float, Dict[str, Any]]:
        """Cierra la ruta actual y avanza al siguiente vehículo o día."""
        from_idx = self._state.current_position
        distance = self.state_encoder.distance_matrix[from_idx, 0]

        self._current_route_nodes.append(0)
        if len(self._current_route_nodes) > 2:
            self._solution.add_route(Route(
                day=self._state.current_day,
                vehicle_id=self._state.current_vehicle,
                nodes=list(self._current_route_nodes),
            ))

        if self._state.current_vehicle < self.instance.num_vehicles:
            self._state.current_vehicle += 1
        else:
            self._state.current_day += 1
            self._state.current_vehicle = 1
            self._state.visited_today.clear()

        self._state.current_position = 0
        self._state.remaining_capacity = self.instance.capacity
        self._current_route_nodes = [0]

        return distance_reward(distance), {
            "closed_route": True,
            "distance": float(distance),
        }

    def _finalize_open_route(self) -> None:
        """Si queda una ruta abierta al terminar el episodio, la cierra."""
        if len(self._current_route_nodes) > 1 and self._current_route_nodes[-1] != 0:
            self._current_route_nodes.append(0)
            self._solution.add_route(Route(
                day=self._state.current_day,
                vehicle_id=self._state.current_vehicle,
                nodes=list(self._current_route_nodes),
            ))
            self._current_route_nodes = [0]

    # =========================================================================
    #  Action Masking (API esperada por sb3-contrib MaskablePPO)
    # =========================================================================

    def action_masks(self) -> np.ndarray:
        """Devuelve el vector booleano de acciones válidas en el estado actual."""
        assert self._state is not None
        return compute_action_mask(self._state, self.instance, self.state_encoder)

    # =========================================================================
    #  Terminación y utilidades
    # =========================================================================

    def _is_episode_done(self) -> bool:
        if self._state.current_day > self.instance.horizon:
            return True
        for c in self.instance.customers:
            if len(self._state.visits_completed[c.id]) < c.frequency:
                return False
        return True

    def _build_info(self) -> Dict[str, Any]:
        return {
            "current_route": list(self._current_route_nodes),
            "day": self._state.current_day,
            "vehicle": self._state.current_vehicle,
        }

    def get_solution(self) -> Solution:
        """Retorna la solución parcial o final construida."""
        assert self._solution is not None
        return self._solution
