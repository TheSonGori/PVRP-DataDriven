"""
Callback de Stable-Baselines3 que registra en TensorBoard, al cierre de cada
episodio, métricas propias del PVRP (costo medio/mínimo, tasa de
factibilidad, número medio de rutas) leídas del `info` que PVRPEnv emite en
el paso terminal.

Entrada: los `locals` del entrenamiento SB3 (dones, infos) en cada _on_step().
Salida: ninguna directa; publica escalares en el logger de TensorBoard bajo
el prefijo "pvrp/".
"""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


# Acumula costo/factibilidad/rutas por episodio (ventana móvil) y los publica en TensorBoard.
class PVRPMetricsCallback(BaseCallback):

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

            cost = info.get("episode_cost")
            feasible = info.get("episode_feasible")
            num_routes = info.get("episode_num_routes")

            if cost is None:
                final = info.get("final_info") or {}
                cost = final.get("episode_cost")
                feasible = final.get("episode_feasible")
                num_routes = final.get("episode_num_routes")

            if cost is None:
                continue

            self._costs.append(float(cost))
            self._feasibility.append(float(feasible))
            self._num_routes.append(float(num_routes))

            if len(self._costs) > self._window_size:
                self._costs.pop(0)
                self._feasibility.pop(0)
                self._num_routes.pop(0)

            self.logger.record("pvrp/ep_cost_mean", float(np.mean(self._costs)))
            self.logger.record("pvrp/ep_cost_min", float(np.min(self._costs)))
            self.logger.record(
                "pvrp/ep_feasibility_rate", float(np.mean(self._feasibility))
            )
            self.logger.record(
                "pvrp/ep_num_routes_mean", float(np.mean(self._num_routes))
            )

        return True
