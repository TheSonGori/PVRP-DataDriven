"""
Entorno multi-instancia para entrenar un agente que generalice.

`MultiInstancePVRPEnv` envuelve varias instancias del PVRP que comparten la
misma dimensión de estado y de acción, y en cada `reset()` selecciona una
(de forma cíclica o aleatoria). Esto fuerza al agente a aprender una política
que funcione en TODAS las instancias, en lugar de memorizar una sola.

Requisito: todas las instancias deben producir el mismo `state_dim` y el mismo
número de acciones (mismo número de clientes). En este proyecto, p01, p02 y
p03 cumplen esto (50 clientes → state_dim=204, action_dim=51), aunque difieren
en el horizonte temporal, lo que las hace un buen banco de prueba de
generalización.

El diseño delega toda la lógica del PVRP en `PVRPEnv`: este wrapper solo decide
QUÉ instancia usar en cada episodio y reexpone la interfaz Gymnasium.
"""

from __future__ import annotations

import random
from typing import List, Optional

import gymnasium as gym
import numpy as np

from src.data.instance import Instance
from src.environment.pvrp_env import PVRPEnv
from src.utils.solution import Solution


class MultiInstancePVRPEnv(gym.Env):
    """
    Entorno que rota entre varias instancias del PVRP en cada episodio.

    Args:
        instances: Lista de instancias compatibles (mismo state_dim y
            action_dim).
        selection: "cyclic" (rota en orden) o "random" (elige al azar).
        seed: Semilla para reproducibilidad.
    """

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

        # Verificar compatibilidad: todas deben tener el mismo número de clientes.
        n0 = instances[0].num_customers
        for inst in instances:
            if inst.num_customers != n0:
                raise ValueError(
                    f"Instancias incompatibles: {instances[0].name} tiene {n0} "
                    f"clientes pero {inst.name} tiene {inst.num_customers}. "
                    "Todas deben tener el mismo número de clientes."
                )

        # Construir un sub-entorno por instancia (se reutilizan).
        self._envs = {
            inst.name: PVRPEnv(inst, seed=seed) for inst in instances
        }

        # Los espacios son los mismos para todas: tomamos los de la primera.
        ref_env = self._envs[instances[0].name]
        self.observation_space = ref_env.observation_space
        self.action_space = ref_env.action_space

        self._current_env: Optional[PVRPEnv] = None
        self._current_name: Optional[str] = None

    def _select_instance(self) -> str:
        """Elige el nombre de la instancia para el próximo episodio."""
        if self.selection == "random":
            return self._rng.choice(list(self._envs.keys()))
        else:  # cyclic
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

    def action_masks(self) -> np.ndarray:
        """Delegado al entorno activo (requerido por MaskablePPO)."""
        return self._current_env.action_masks()

    def get_solution(self) -> Solution:
        """Solución del episodio actual."""
        return self._current_env.get_solution()

    @property
    def instance(self) -> Instance:
        """Instancia activa (para que los callbacks puedan leer sus métricas)."""
        return self._current_env.instance
