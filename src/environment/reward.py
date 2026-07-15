"""
Define la función de recompensa del entorno PVRP: una penalización por la
distancia recorrida en cada paso, más una recompensa terminal (bonus si la
solución final es factible, penalización creciente con el número de visitas
faltantes si no lo es).

Entrada: distancia recorrida en un paso, o el resultado de factibilidad y
número de visitas faltantes al terminar el episodio, junto con un
RewardConfig opcional.
Salida: valores float de recompensa usados por PVRPEnv.step().
"""

from __future__ import annotations

from dataclasses import dataclass


# Hiperparámetros de la recompensa: bonus por factibilidad y penalizaciones por incompletitud.
@dataclass(frozen=True)
class RewardConfig:
    terminal_bonus: float = 100.0
    infeasibility_penalty: float = -500.0
    per_missing_penalty: float = -50.0


DEFAULT_REWARD_CONFIG = RewardConfig()


# Penalización proporcional a la distancia recorrida en el paso.
def distance_reward(distance: float) -> float:
    return -float(distance)


# Recompensa aplicada al finalizar el episodio: bonus si es factible, penalización si no.
def terminal_reward(
    is_feasible: bool,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
    num_missing_visits: int = 0,
) -> float:
    if is_feasible:
        return config.terminal_bonus
    return config.infeasibility_penalty + config.per_missing_penalty * num_missing_visits
