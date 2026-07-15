"""
Test de humo del entrenamiento del agente PPO: no verifica calidad, solo que
el pipeline completo (entorno + ActionMasker + MaskablePPO + guardado/carga +
predicción) funcione de punta a punta sin errores.

Entrada: la instancia p01 (data/raw/p01.txt) y SMOKE_TEST_CONFIG.
Salida: aserciones pytest; no retorna valores.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agent.policy_config import SMOKE_TEST_CONFIG
from src.agent.train import build_env, train_agent
from src.data.instance_loader import load_instance


DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


class TestPipelineConnectivity:

    # build_env() expone espacios de observación/acción coherentes con la instancia.
    def test_build_env_returns_wrapped_env(self):
        instance = load_instance(DATA_DIR / "p01.txt")
        env = build_env(instance, seed=0)
        assert env.observation_space.shape[0] > 0
        assert env.action_space.n == instance.num_customers + 1

    # El entorno envuelto expone action_masks() con el tamaño esperado.
    def test_env_has_action_masks_method(self):
        instance = load_instance(DATA_DIR / "p01.txt")
        env = build_env(instance, seed=0)
        env.reset()
        mask = env.action_masks()
        assert mask.shape == (env.action_space.n,)


class TestSmokeTraining:

    # Un entrenamiento breve con SMOKE_TEST_CONFIG completa sin errores.
    def test_train_completes_without_errors(self):
        instance = load_instance(DATA_DIR / "p01.txt")
        model = train_agent(instance, config=SMOKE_TEST_CONFIG)
        assert model is not None

    # Tras entrenar, el agente predice una acción válida según la máscara.
    def test_trained_agent_can_predict(self):
        instance = load_instance(DATA_DIR / "p01.txt")
        model = train_agent(instance, config=SMOKE_TEST_CONFIG)
        env = build_env(instance, seed=0)

        obs, _ = env.reset()
        mask = env.action_masks()
        action, _ = model.predict(
            obs, action_masks=mask, deterministic=True
        )
        assert mask[int(action)] == True

    # El agente entrenado completa un episodio del PVRP sin crashear.
    def test_trained_agent_completes_episode(self):
        instance = load_instance(DATA_DIR / "p01.txt")
        model = train_agent(instance, config=SMOKE_TEST_CONFIG)
        env = build_env(instance, seed=0)

        obs, _ = env.reset()
        terminated = False
        steps = 0
        while not terminated and steps < 5000:
            mask = env.action_masks()
            action, _ = model.predict(
                obs, action_masks=mask, deterministic=True
            )
            obs, reward, terminated, truncated, info = env.step(int(action))
            steps += 1

        assert terminated, "Episodio no terminó en 5000 pasos"

    # El modelo entrenado se guarda a disco y, al recargarlo, predice la misma acción.
    def test_save_and_load_model(self, tmp_path):
        instance = load_instance(DATA_DIR / "p01.txt")
        save_path = tmp_path / "test_model"

        model_a = train_agent(instance, config=SMOKE_TEST_CONFIG, save_path=save_path)
        assert (tmp_path / "test_model.zip").exists()

        from sb3_contrib import MaskablePPO
        model_b = MaskablePPO.load(str(save_path))

        env = build_env(instance, seed=0)
        obs, _ = env.reset()
        mask = env.action_masks()
        action_a, _ = model_a.predict(obs, action_masks=mask, deterministic=True)
        action_b, _ = model_b.predict(obs, action_masks=mask, deterministic=True)
        assert int(action_a) == int(action_b)
