"""
Test de humo del entrenamiento del agente PPO (Día 10).

NO verifica calidad del agente, solo que el pipeline completo funcione:

    - Crear entorno envuelto en ActionMasker.
    - Instanciar MaskablePPO.
    - Ejecutar un breve entrenamiento sin errores.
    - Cargar/guardar el modelo.
    - Predecir una acción válida y ejecutarla en el entorno.

Si este test pasa, todas las piezas (entorno, máscara, modelo, librería)
están conectadas correctamente.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agent.policy_config import SMOKE_TEST_CONFIG
from src.agent.train import build_env, train_agent
from src.data.instance_loader import load_instance


DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


class TestPipelineConnectivity:
    """Verifica que cada componente del pipeline se enlaza correctamente."""

    def test_build_env_returns_wrapped_env(self):
        instance = load_instance(DATA_DIR / "p01.txt")
        env = build_env(instance, seed=0)
        # ActionMasker expone los mismos espacios que el entorno base
        assert env.observation_space.shape[0] > 0
        assert env.action_space.n == instance.num_customers + 1

    def test_env_has_action_masks_method(self):
        instance = load_instance(DATA_DIR / "p01.txt")
        env = build_env(instance, seed=0)
        env.reset()
        mask = env.action_masks()
        assert mask.shape == (env.action_space.n,)


class TestSmokeTraining:
    """
    Test de humo del entrenamiento. Es lento (~5-10 segundos) por lo que
    usamos la configuración mínima `SMOKE_TEST_CONFIG`.
    """

    def test_train_completes_without_errors(self):
        instance = load_instance(DATA_DIR / "p01.txt")
        model = train_agent(instance, config=SMOKE_TEST_CONFIG)
        assert model is not None

    def test_trained_agent_can_predict(self):
        """Tras un breve entrenamiento, el agente puede predecir acciones
        válidas usando la máscara."""
        instance = load_instance(DATA_DIR / "p01.txt")
        model = train_agent(instance, config=SMOKE_TEST_CONFIG)
        env = build_env(instance, seed=0)

        obs, _ = env.reset()
        mask = env.action_masks()
        action, _ = model.predict(
            obs, action_masks=mask, deterministic=True
        )
        # La acción predicha debe ser una de las válidas según la máscara
        assert mask[int(action)] == True

    def test_trained_agent_completes_episode(self):
        """El agente debe poder completar un episodio del PVRP sin crashear."""
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

    def test_save_and_load_model(self, tmp_path):
        """El modelo entrenado se guarda y se puede recargar."""
        instance = load_instance(DATA_DIR / "p01.txt")
        save_path = tmp_path / "test_model"

        model_a = train_agent(instance, config=SMOKE_TEST_CONFIG, save_path=save_path)
        # MaskablePPO añade ".zip" automáticamente al guardar
        assert (tmp_path / "test_model.zip").exists()

        # Recargar y verificar que predice
        from sb3_contrib import MaskablePPO
        model_b = MaskablePPO.load(str(save_path))

        env = build_env(instance, seed=0)
        obs, _ = env.reset()
        mask = env.action_masks()
        action_a, _ = model_a.predict(obs, action_masks=mask, deterministic=True)
        action_b, _ = model_b.predict(obs, action_masks=mask, deterministic=True)
        assert int(action_a) == int(action_b)
