"""
Hiperparámetros del agente MaskablePPO, separados del código de entrenamiento
para facilitar la experimentación.

Entrada: valores opcionales para instanciar PPOConfig (total_timesteps,
learning_rate, arquitectura de red, etc.).
Salida: un PPOConfig (o una de las configuraciones predefinidas
SMOKE_TEST_CONFIG, QUICK_TRAIN_CONFIG, FULL_TRAIN_CONFIG) usado por train.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# Configuración del agente MaskablePPO: hiperparámetros de entrenamiento y arquitectura de red.
@dataclass(frozen=True)
class PPOConfig:
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


SMOKE_TEST_CONFIG = PPOConfig(
    total_timesteps=2_048,
    n_steps=512,
    batch_size=64,
    verbose=0,
)

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
