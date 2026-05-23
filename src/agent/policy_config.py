"""
Hiperparámetros del agente de Aprendizaje por Refuerzo.

Esta configuración se separa del código de entrenamiento (`train.py`) para
facilitar la experimentación y la documentación en el Capítulo 4 de la memoria.
Los valores por defecto siguen las recomendaciones de la documentación oficial
de stable-baselines3 para problemas combinatorios pequeños.

Para ajustes específicos, instanciar `PPOConfig` con los valores deseados:

    cfg = PPOConfig(total_timesteps=200_000, learning_rate=1e-4)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class PPOConfig:
    """
    Configuración del agente MaskablePPO.

    Attributes:
        total_timesteps: Número total de interacciones agente-entorno durante
            el entrenamiento. Más pasos = mejor convergencia pero más tiempo.
        learning_rate: Tasa de aprendizaje del optimizador Adam.
        n_steps: Número de pasos recolectados antes de cada actualización
            de política. Valores grandes estabilizan; valores pequeños son
            más eficientes en memoria.
        batch_size: Tamaño del minibatch para SGD. Debe dividir a n_steps.
        n_epochs: Número de pases sobre los datos recolectados por actualización.
        gamma: Factor de descuento. Valores cercanos a 1 dan más peso a
            recompensas futuras (importante para problemas con horizonte largo
            como el PVRP).
        gae_lambda: Coeficiente GAE para estimar ventajas.
        clip_range: Recorte de PPO. Limita cuánto puede cambiar la política
            en cada actualización.
        ent_coef: Coeficiente de entropía. Mayores valores fomentan exploración.
        policy_kwargs: Parámetros adicionales de la política (arquitectura
            de la red neuronal).
        verbose: 0 = silencioso, 1 = info, 2 = debug.
        seed: Semilla para reproducibilidad.
    """
    total_timesteps: int = 100_000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    policy_kwargs: dict = field(default_factory=lambda: {
        "net_arch": [128, 128],
    })
    verbose: int = 0
    seed: int = 42


# Configuraciones nombradas para escenarios típicos
SMOKE_TEST_CONFIG = PPOConfig(
    total_timesteps=2_048,
    n_steps=512,
    batch_size=64,
    verbose=0,
)

# Configuración recomendada tras el análisis del Día 11: más exploración
# (ent_coef alto) y red más grande mejoran notablemente la convergencia en
# problemas combinatorios como el PVRP.
QUICK_TRAIN_CONFIG = PPOConfig(
    total_timesteps=150_000,
    ent_coef=0.05,
    policy_kwargs={"net_arch": [256, 256]},
    verbose=1,
)

FULL_TRAIN_CONFIG = PPOConfig(
    total_timesteps=500_000,
    ent_coef=0.05,
    policy_kwargs={"net_arch": [256, 256]},
    verbose=1,
)
