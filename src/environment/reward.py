"""
Función de recompensa del entorno PVRP.

La recompensa se descompone en dos términos, siguiendo la formulación
descrita en la Sección 3.3 de la memoria (diseño del MDP):

    r_t = r_distancia(t) + r_terminal(t)

donde:

    r_distancia(t) = -d(from, to)
        Penalización por la distancia recorrida en la transición. Su signo
        negativo convierte la maximización de la recompensa acumulada en
        equivalente a la minimización del costo total (Ecuación 1 del modelo
        matemático, Sección 1.5.2).

    r_terminal(t) = R_solved si el episodio termina con todos los clientes
                    visitados según su frecuencia y patrones válidos.
                  = R_infeasible si el episodio termina con clientes pendientes
                    o patrones inválidos.
                  = 0 en pasos intermedios.

La recompensa terminal proporciona una señal global que guía al agente hacia
soluciones factibles, complementando la señal local de la distancia. Los
valores numéricos (R_solved, R_infeasible) son hiperparámetros que se ajustan
empíricamente en el Capítulo 4 de la memoria.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    """
    Configuración de la función de recompensa.

    Attributes:
        terminal_bonus: Bonus aplicado al cerrar un episodio con solución
            factible. Debe ser positivo y grande respecto al costo típico
            para incentivar la completitud.
        infeasibility_penalty: Penalización aplicada si el episodio termina
            con visitas incompletas. Debe ser negativa y de magnitud
            considerable.
    """
    terminal_bonus: float = 100.0
    infeasibility_penalty: float = -500.0


DEFAULT_REWARD_CONFIG = RewardConfig()


def distance_reward(distance: float) -> float:
    """Componente de recompensa basado en la distancia recorrida."""
    return -float(distance)


def terminal_reward(is_feasible: bool, config: RewardConfig = DEFAULT_REWARD_CONFIG) -> float:
    """Componente de recompensa aplicado al final del episodio."""
    return config.terminal_bonus if is_feasible else config.infeasibility_penalty
