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
        infeasibility_penalty: Penalización BASE aplicada si el episodio
            termina con visitas incompletas. Debe ser negativa.
        per_missing_penalty: Penalización ADICIONAL por cada visita de cliente
            que quedó sin realizar. Hace que abandonar clientes sea cada vez
            más caro, evitando que el agente explote el atajo de "visitar
            pocos clientes y cerrar" (reward hacking). Debe ser negativa.
    """
    terminal_bonus: float = 100.0
    infeasibility_penalty: float = -500.0
    per_missing_penalty: float = -50.0


DEFAULT_REWARD_CONFIG = RewardConfig()


def distance_reward(distance: float) -> float:
    """Componente de recompensa basado en la distancia recorrida."""
    return -float(distance)


def terminal_reward(
    is_feasible: bool,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
    num_missing_visits: int = 0,
) -> float:
    """
    Componente de recompensa aplicado al final del episodio.

    Args:
        is_feasible: Si la solución final es factible.
        config: Configuración de recompensa.
        num_missing_visits: Número de visitas de clientes que quedaron sin
            realizar. Cada una añade `per_missing_penalty` a la penalización,
            de modo que abandonar muchos clientes es mucho peor que abandonar
            pocos. Esto elimina el incentivo perverso a terminar episodios
            triviales (visitar 1 cliente y cerrar).

    Returns:
        El bonus si es factible; en caso contrario, la penalización base más
        la penalización proporcional a las visitas faltantes.
    """
    if is_feasible:
        return config.terminal_bonus
    return config.infeasibility_penalty + config.per_missing_penalty * num_missing_visits
