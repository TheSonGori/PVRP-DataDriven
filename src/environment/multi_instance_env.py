"""
Envuelve varias instancias del PVRP con la misma dimensión de estado/acción y
elige una en cada reset() (cíclica o aleatoriamente), forzando al agente a
aprender una política que generalice en vez de memorizar una sola instancia.
Delega toda la lógica del PVRP en PVRPEnv.

Entrada: lista de Instance compatibles (mismo número de clientes), modo de
selección ("cyclic"/"random") y semilla.
Salida: un entorno Gymnasium (observación, recompensa, terminado, info) que
reexpone la interfaz de PVRPEnv sobre la instancia activa en cada episodio.
"""

from __future__ import annotations

import random
from typing import List, Optional

import gymnasium as gym
import numpy as np

from src.data.instance import Instance
from src.environment.pvrp_env import PVRPEnv
from src.utils.solution import Solution


# Entorno que rota entre varias instancias del PVRP en cada episodio.
class MultiInstancePVRPEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        instances: List[Instance],
        selection: str = "cyclic",
        seed: int = 0,
    ):
        super().__init__()
        if not instances:
            raise ValueError("Se requiere al menos una instancia.")

        self.instances = instances
        self.selection = selection
        self._rng = random.Random(seed)
        self._cyclic_idx = 0

        n0 = instances[0].num_customers
        for inst in instances:
            if inst.num_customers != n0:
                raise ValueError(
                    f"Instancias incompatibles: {instances[0].name} tiene {n0} "
                    f"clientes pero {inst.name} tiene {inst.num_customers}. "
                    "Todas deben tener el mismo número de clientes."
                )

        self._envs = {
            inst.name: PVRPEnv(inst, seed=seed) for inst in instances
        }

        ref_env = self._envs[instances[0].name]
        self.observation_space = ref_env.observation_space
        self.action_space = ref_env.action_space

        self._current_env: Optional[PVRPEnv] = None
        self._current_name: Optional[str] = None

    # Elige el nombre de la instancia para el próximo episodio.
    def _select_instance(self) -> str:
        if self.selection == "random":
            return self._rng.choice(list(self._envs.keys()))
        else:
            names = [inst.name for inst in self.instances]
            name = names[self._cyclic_idx % len(names)]
            self._cyclic_idx += 1
            return name

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng.seed(seed)
        self._current_name = self._select_instance()
        self._current_env = self._envs[self._current_name]
        return self._current_env.reset(seed=seed, options=options)

    def step(self, action):
        return self._current_env.step(action)

    # Delegado al entorno activo (requerido por MaskablePPO).
    def action_masks(self) -> np.ndarray:
        return self._current_env.action_masks()

    # Solución del episodio actual.
    def get_solution(self) -> Solution:
        return self._current_env.get_solution()

    # Instancia activa (para que los callbacks puedan leer sus métricas).
    @property
    def instance(self) -> Instance:
        return self._current_env.instance
