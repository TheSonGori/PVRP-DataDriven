"""
Entrena un agente MaskablePPO (sb3-contrib) sobre el entorno PVRP, con
Action Masking nativo (la política solo considera acciones permitidas por
`action_masks()`) y un callback que registra métricas del PVRP en
TensorBoard.

Entrada: una Instance (o lista de Instance para el modo multi-instancia) y
un PPOConfig (src/agent/policy_config.py); opcionalmente rutas de guardado
y de logs de TensorBoard.
Salida: un modelo MaskablePPO entrenado (y, si se indica save_path, guardado
en disco).
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


# Extrae la máscara de acciones del entorno (requerida por ActionMasker).
def _mask_fn(env: PVRPEnv):
    return env.action_masks()


# Construye un PVRPEnv envuelto en ActionMasker, listo para MaskablePPO.
def build_env(instance: Instance, seed: int = 0) -> ActionMasker:
    env = PVRPEnv(instance, seed=seed)
    env = ActionMasker(env, _mask_fn)
    return env


# Construye un entorno multi-instancia envuelto en ActionMasker.
def build_multi_env(instances, selection: str = "cyclic", seed: int = 0):
    from src.environment.multi_instance_env import MultiInstancePVRPEnv

    env = MultiInstancePVRPEnv(instances, selection=selection, seed=seed)
    env = ActionMasker(env, _mask_fn)
    return env


# Entrena un agente MaskablePPO rotando entre varias instancias compatibles.
def train_agent_multi(
    instances,
    config: PPOConfig,
    selection: str = "cyclic",
    save_path: Optional[Path] = None,
    tensorboard_log: Optional[Path] = None,
) -> MaskablePPO:
    env = build_multi_env(instances, selection=selection, seed=config.seed)

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

    callbacks = [PVRPMetricsCallback(verbose=0)]
    callback = CallbackList(callbacks)

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


# Entrena un agente MaskablePPO sobre una única instancia.
def train_agent(
    instance: Instance,
    config: PPOConfig,
    save_path: Optional[Path] = None,
    tensorboard_log: Optional[Path] = None,
    extra_callbacks: Optional[list[BaseCallback]] = None,
) -> MaskablePPO:
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
