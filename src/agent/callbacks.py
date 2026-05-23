"""
Callback de logging para el entrenamiento del agente PVRP.

Este callback se ejecuta al final de cada episodio durante el entrenamiento y
registra métricas específicas del PVRP que TensorBoard grafica en tiempo real:

    - pvrp/ep_cost_mean       : costo medio (ventana móvil) de las soluciones.
    - pvrp/ep_cost_min        : mejor costo observado en la ventana.
    - pvrp/ep_feasibility_rate: fracción de episodios con solución factible.
    - pvrp/ep_num_routes_mean : número medio de rutas usadas.

IMPORTANTE: las métricas se leen del diccionario `info` que el entorno emite
en el paso terminal (claves `episode_cost`, `episode_feasible`,
`episode_num_routes`). Esto evita un problema sutil de *timing*: cuando un
episodio termina dentro de un `DummyVecEnv`, el entorno se reinicia
automáticamente, por lo que consultar `env.get_solution()` después de la
terminación devolvería la solución del episodio NUEVO (vacío), no del que
acaba de terminar. Leer del `info` capturado en el step terminal resuelve esto.
"""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class PVRPMetricsCallback(BaseCallback):
    """Registra métricas del PVRP al cierre de cada episodio."""

    def __init__(self, verbose: int = 0, window_size: int = 100):
        super().__init__(verbose)
        self._costs: list[float] = []
        self._feasibility: list[float] = []
        self._num_routes: list[float] = []
        self._window_size = window_size

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals.get("dones", [])):
            if not done:
                continue

            info = self.locals["infos"][i]

            # Las métricas vienen en el info del step terminal.
            # (En un VecEnv, SB3 conserva el info del episodio terminado.)
            cost = info.get("episode_cost")
            feasible = info.get("episode_feasible")
            num_routes = info.get("episode_num_routes")

            if cost is None:
                # Algunos VecEnv mueven el info terminal a "final_info".
                final = info.get("final_info") or {}
                cost = final.get("episode_cost")
                feasible = final.get("episode_feasible")
                num_routes = final.get("episode_num_routes")

            if cost is None:
                continue  # no había métricas en este info; saltar

            self._costs.append(float(cost))
            self._feasibility.append(float(feasible))
            self._num_routes.append(float(num_routes))

            # Ventana móvil
            if len(self._costs) > self._window_size:
                self._costs.pop(0)
                self._feasibility.pop(0)
                self._num_routes.pop(0)

            # Publicar a TensorBoard
            self.logger.record("pvrp/ep_cost_mean", float(np.mean(self._costs)))
            self.logger.record("pvrp/ep_cost_min", float(np.min(self._costs)))
            self.logger.record(
                "pvrp/ep_feasibility_rate", float(np.mean(self._feasibility))
            )
            self.logger.record(
                "pvrp/ep_num_routes_mean", float(np.mean(self._num_routes))
            )

        return True
