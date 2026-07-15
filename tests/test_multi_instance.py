"""
Tests del entorno multi-instancia: rotación entre instancias (cíclica y
aleatoria), rechazo de instancias incompatibles, delegación correcta de
action_masks/get_solution/instance, y entrenamiento multi-instancia de humo.

Entrada: instancias compatibles del dataset (data/raw/p01.txt, p02.txt, p03.txt).
Salida: aserciones pytest; no retorna valores.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agent.policy_config import SMOKE_TEST_CONFIG
from src.agent.train import build_multi_env, train_agent_multi
from src.data.instance_loader import load_instance
from src.environment.multi_instance_env import MultiInstancePVRPEnv


DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

_COMPAT = ["p01", "p02", "p03"]
COMPAT_INSTANCES = [n for n in _COMPAT if (DATA_DIR / f"{n}.txt").exists()]

pytestmark = pytest.mark.skipif(
    len(COMPAT_INSTANCES) < 2,
    reason="Se requieren >=2 instancias de 50 clientes (p01, p02, p03).",
)


@pytest.fixture
def instances():
    return [load_instance(DATA_DIR / f"{n}.txt") for n in COMPAT_INSTANCES]


class TestMultiInstanceEnv:

    # Los espacios de observación/acción coinciden con los de un PVRPEnv individual.
    def test_spaces_match_single_env(self, instances):
        env = MultiInstancePVRPEnv(instances, seed=0)
        assert env.observation_space.shape[0] == 204
        assert env.action_space.n == 51

    # En modo cíclico, el primer ciclo de resets cubre todas las instancias sin repetir.
    def test_cyclic_rotation(self, instances):
        env = MultiInstancePVRPEnv(instances, selection="cyclic", seed=0)
        seen = []
        for _ in range(len(instances) * 2):
            env.reset()
            seen.append(env.instance.name)
        first_cycle = seen[: len(instances)]
        assert set(first_cycle) == set(COMPAT_INSTANCES)

    # En modo aleatorio, la instancia elegida siempre pertenece al conjunto compatible.
    def test_random_selection_stays_in_set(self, instances):
        env = MultiInstancePVRPEnv(instances, selection="random", seed=1)
        for _ in range(10):
            env.reset()
            assert env.instance.name in COMPAT_INSTANCES

    # Instancias de distinto número de clientes son rechazadas al construir el entorno.
    def test_rejects_incompatible_instances(self):
        p01 = load_instance(DATA_DIR / "p01.txt")
        if (DATA_DIR / "p04.txt").exists():
            p04 = load_instance(DATA_DIR / "p04.txt")
            with pytest.raises(ValueError):
                MultiInstancePVRPEnv([p01, p04])

    # action_masks() se delega correctamente al sub-entorno activo.
    def test_action_masks_delegated(self, instances):
        env = MultiInstancePVRPEnv(instances, seed=0)
        env.reset()
        mask = env.action_masks()
        assert mask.shape == (env.action_space.n,)

    # Un episodio con acciones válidas termina en tiempo finito.
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

    # El entrenamiento multi-instancia corre sin errores.
    def test_smoke_training_runs(self, instances):
        model = train_agent_multi(
            instances, config=SMOKE_TEST_CONFIG, selection="cyclic"
        )
        assert model is not None

    # El agente entrenado predice una acción válida en cada instancia del conjunto.
    def test_trained_model_predicts_on_all(self, instances):
        model = train_agent_multi(
            instances, config=SMOKE_TEST_CONFIG, selection="cyclic"
        )
        for inst in instances:
            env = build_multi_env([inst], seed=0)
            obs, _ = env.reset()
            mask = env.action_masks()
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            assert mask[int(action)]
