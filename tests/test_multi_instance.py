"""
Tests del entorno multi-instancia (Día 13).

Verifican que:
    - El entorno rota correctamente entre instancias (cíclico y aleatorio).
    - Rechaza instancias incompatibles (distinto número de clientes).
    - Expone correctamente action_masks, get_solution e instance.
    - El entrenamiento multi-instancia corre sin errores (humo).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agent.policy_config import SMOKE_TEST_CONFIG
from src.agent.train import build_multi_env, train_agent_multi
from src.data.instance_loader import load_instance
from src.environment.multi_instance_env import MultiInstancePVRPEnv


DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

# Instancias compatibles (50 clientes) que deberían existir en el dataset.
_COMPAT = ["p01", "p02", "p03"]
COMPAT_INSTANCES = [n for n in _COMPAT if (DATA_DIR / f"{n}.txt").exists()]

# Se necesitan al menos 2 instancias compatibles para los tests de rotación.
pytestmark = pytest.mark.skipif(
    len(COMPAT_INSTANCES) < 2,
    reason="Se requieren >=2 instancias de 50 clientes (p01, p02, p03).",
)


@pytest.fixture
def instances():
    return [load_instance(DATA_DIR / f"{n}.txt") for n in COMPAT_INSTANCES]


class TestMultiInstanceEnv:
    def test_spaces_match_single_env(self, instances):
        env = MultiInstancePVRPEnv(instances, seed=0)
        # state_dim = 50*4 + 4 = 204; action_dim = 50 + 1 = 51
        assert env.observation_space.shape[0] == 204
        assert env.action_space.n == 51

    def test_cyclic_rotation(self, instances):
        env = MultiInstancePVRPEnv(instances, selection="cyclic", seed=0)
        seen = []
        for _ in range(len(instances) * 2):
            env.reset()
            seen.append(env.instance.name)
        # En modo cíclico, las primeras N deben ser todas distintas y cubrir todo
        first_cycle = seen[: len(instances)]
        assert set(first_cycle) == set(COMPAT_INSTANCES)

    def test_random_selection_stays_in_set(self, instances):
        env = MultiInstancePVRPEnv(instances, selection="random", seed=1)
        for _ in range(10):
            env.reset()
            assert env.instance.name in COMPAT_INSTANCES

    def test_rejects_incompatible_instances(self):
        """Instancias de distinto tamaño deben ser rechazadas."""
        p01 = load_instance(DATA_DIR / "p01.txt")
        # p04 tiene 75 clientes (si existe en el dataset de prueba)
        if (DATA_DIR / "p04.txt").exists():
            p04 = load_instance(DATA_DIR / "p04.txt")
            with pytest.raises(ValueError):
                MultiInstancePVRPEnv([p01, p04])

    def test_action_masks_delegated(self, instances):
        env = MultiInstancePVRPEnv(instances, seed=0)
        env.reset()
        mask = env.action_masks()
        assert mask.shape == (env.action_space.n,)

    def test_episode_runs_to_completion(self, instances):
        env = MultiInstancePVRPEnv(instances, seed=0)
        obs, _ = env.reset()
        terminated = False
        steps = 0
        while not terminated and steps < 5000:
            mask = env.action_masks()
            valid = [i for i, m in enumerate(mask) if m]
            obs, _, terminated, _, _ = env.step(valid[0])
            steps += 1
        assert terminated


class TestMultiInstanceTraining:
    def test_smoke_training_runs(self, instances):
        """El entrenamiento multi-instancia corre sin errores."""
        model = train_agent_multi(
            instances, config=SMOKE_TEST_CONFIG, selection="cyclic"
        )
        assert model is not None

    def test_trained_model_predicts_on_all(self, instances):
        """El agente entrenado puede predecir en todas las instancias."""
        model = train_agent_multi(
            instances, config=SMOKE_TEST_CONFIG, selection="cyclic"
        )
        for inst in instances:
            env = build_multi_env([inst], seed=0)
            obs, _ = env.reset()
            mask = env.action_masks()
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            assert mask[int(action)]
