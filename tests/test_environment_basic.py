"""
Tests básicos de PVRPEnv: dimensiones de observación/acción, reset() con
estado inicial coherente, step() con acción válida (visita) e inválida, y
terminación del episodio.

Entrada: la instancia p01 (data/raw/p01.txt).
Salida: aserciones pytest; no retorna valores.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.instance_loader import load_instance
from src.environment.pvrp_env import PVRPEnv
from src.environment.state import NUM_GLOBAL_FEATURES, NUM_CUSTOMER_FEATURES


DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


@pytest.fixture(scope="module")
def env():
    instance = load_instance(DATA_DIR / "p01.txt")
    return PVRPEnv(instance, seed=42)


class TestSpaces:

    # La dimensión de observación es 4 globales + 50 clientes * 4 features = 204.
    def test_observation_dim(self, env):
        expected = NUM_GLOBAL_FEATURES + 50 * NUM_CUSTOMER_FEATURES
        assert env.observation_space.shape == (expected,)

    # El espacio de acciones tiene 50 clientes + 1 acción de cierre = 51.
    def test_action_space(self, env):
        assert env.action_space.n == 51

    # La observación se codifica en float32.
    def test_observation_dtype(self, env):
        obs, _ = env.reset()
        assert obs.dtype == np.float32

    # Todos los valores de la observación están en [0, 1].
    def test_observation_in_bounds(self, env):
        obs, _ = env.reset()
        assert (obs >= 0.0).all()
        assert (obs <= 1.0).all()


class TestReset:

    # reset() devuelve la tupla (obs, info).
    def test_returns_obs_and_info(self, env):
        result = env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2

    # Al iniciar, la ruta actual está en el depósito.
    def test_initial_position_is_depot(self, env):
        obs, info = env.reset()
        assert info["current_route"] == [0]

    # Al iniciar, la capacidad restante es la capacidad total del vehículo.
    def test_initial_remaining_capacity_full(self, env):
        env.reset()
        assert env._state.remaining_capacity == env.instance.capacity

    # Al iniciar, el día y el vehículo activos son ambos 1.
    def test_initial_day_and_vehicle(self, env):
        env.reset()
        assert env._state.current_day == 1
        assert env._state.current_vehicle == 1


class TestStepVisit:

    # Visitar un cliente válido da recompensa negativa (distancia) y actualiza la ruta.
    def test_visit_valid_customer(self, env):
        env.reset()
        obs, reward, terminated, truncated, info = env.step(1)
        assert reward < 0
        assert not terminated
        assert info["visited"] == 1
        assert info["current_route"] == [0, 1]

    # La capacidad restante disminuye tras visitar un cliente.
    def test_capacity_decreases_after_visit(self, env):
        env.reset()
        initial = env._state.remaining_capacity
        env.step(1)
        assert env._state.remaining_capacity < initial

    # Visitar a un cliente registra una visita completada para ese cliente.
    def test_visit_decreases_remaining_visits(self, env):
        env.reset()
        c = env.instance.customers[0]
        env.step(1)
        visits = env._state.visits_completed[c.id]
        assert len(visits) == 1


class TestStepInvalid:

    # Tras visitar un cliente, la máscara prohíbe volver a visitarlo el mismo día.
    def test_double_visit_same_day_blocked_by_mask(self, env):
        env.reset()
        env.step(1)
        mask = env.action_masks()
        assert mask[1] == False


class TestCloseRoute:

    # Cerrar la ruta actual devuelve la posición al depósito.
    def test_close_route_returns_to_depot(self, env):
        env.reset()
        env.step(1)
        obs, reward, *_ = env.step(0)
        assert env._state.current_position == 0

    # Cerrar la ruta avanza al siguiente vehículo del día.
    def test_close_route_advances_vehicle(self, env):
        env.reset()
        env.step(1)
        env.step(0)
        assert env._state.current_vehicle == 2

    # Cerrar la ruta la registra en la solución del entorno.
    def test_close_route_registers_route_in_solution(self, env):
        env.reset()
        env.step(1)
        env.step(0)
        sol = env.get_solution()
        assert len(sol.routes) == 1
        assert sol.routes[0].nodes == [0, 1, 0]


class TestRandomEpisode:

    # Un episodio con acciones aleatorias válidas termina en tiempo finito.
    def test_random_episode_terminates(self):
        instance = load_instance(DATA_DIR / "p01.txt")
        env = PVRPEnv(instance, seed=0)
        env.reset()

        max_steps = 5000
        rng = np.random.default_rng(0)
        for _ in range(max_steps):
            mask = env.action_masks()
            valid_actions = np.flatnonzero(mask)
            action = int(rng.choice(valid_actions))
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        else:
            pytest.fail(f"Episodio no terminó en {max_steps} pasos.")

        assert terminated
