"""
Entrenamiento del agente MaskablePPO sobre el entorno PVRP.

Versión del Día 11: incorpora TensorBoard para monitoreo en vivo y callbacks
personalizados para registrar métricas específicas del PVRP.

MaskablePPO es la versión de PPO con soporte nativo para enmascaramiento de
acciones (sb3-contrib). En cada paso, la política consulta el método
`action_masks()` del entorno y solo considera las acciones permitidas
durante el muestreo y el cálculo de gradientes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from src.agent.callbacks import PVRPMetricsCallback
from src.agent.policy_config import PPOConfig
from src.data.instance import Instance
from src.environment.pvrp_env import PVRPEnv


def _mask_fn(env: PVRPEnv):
    """Función que extrae la máscara del entorno (requerida por ActionMasker)."""
    return env.action_masks()


def build_env(instance: Instance, seed: int = 0) -> ActionMasker:
    """
    Construye un entorno PVRPEnv envuelto en ActionMasker.

    Args:
        instance: Instancia del PVRP sobre la cual entrenar.
        seed: Semilla para reproducibilidad.

    Returns:
        Un entorno listo para `MaskablePPO.learn(...)`.
    """
    env = PVRPEnv(instance, seed=seed)
    env = ActionMasker(env, _mask_fn)
    return env


def train_agent(
    instance: Instance,
    config: PPOConfig,
    save_path: Optional[Path] = None,
    tensorboard_log: Optional[Path] = None,
    extra_callbacks: Optional[list[BaseCallback]] = None,
) -> MaskablePPO:
    """
    Entrena un agente MaskablePPO sobre la instancia indicada.

    Args:
        instance: Instancia del PVRP de entrenamiento.
        config: Configuración de hiperparámetros (`PPOConfig`).
        save_path: Si se provee, el modelo entrenado se guarda en esa ruta.
        tensorboard_log: Si se provee, los logs de TensorBoard se guardan ahí.
            Para visualizarlos: `tensorboard --logdir <ruta>`.
        extra_callbacks: Callbacks adicionales para registrar métricas o
            controlar el entrenamiento.

    Returns:
        Modelo `MaskablePPO` entrenado.
    """
    env = build_env(instance, seed=config.seed)

    model = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        policy_kwargs=config.policy_kwargs,
        verbose=config.verbose,
        seed=config.seed,
        tensorboard_log=str(tensorboard_log) if tensorboard_log else None,
    )

    # Componer lista de callbacks
    callbacks: list[BaseCallback] = [PVRPMetricsCallback(verbose=0)]
    if extra_callbacks:
        callbacks.extend(extra_callbacks)
    callback = CallbackList(callbacks) if callbacks else None

    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callback,
        progress_bar=False,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(save_path))

    return model
