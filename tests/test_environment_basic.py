"""
Tests básicos del entorno PVRPEnv (Día 4).

Estos tests verifican que:
    - El entorno se crea con dimensiones correctas (observación y acción).
    - `reset()` devuelve un estado inicial coherente.
    - `step()` ejecuta acciones válidas (visitar cliente) y suma costos.
    - `step()` con acción inválida devuelve penalización sin crashear.
    - El episodio termina cuando se completan las visitas o se agota el horizonte.

Los tests de máscara y recompensa final se agregan en el Día 5.
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


# =============================================================================
#  Espacios de observación y acción
# =============================================================================

class TestSpaces:
    def test_observation_dim(self, env):
        # 4 globales + 50 clientes * 4 features = 204
        expected = NUM_GLOBAL_FEATURES + 50 * NUM_CUSTOMER_FEATURES
        assert env.observation_space.shape == (expected,)

    def test_action_space(self, env):
        # 50 clientes + 1 acción de cierre = 51 acciones posibles
        assert env.action_space.n == 51

    def test_observation_dtype(self, env):
        obs, _ = env.reset()
        assert obs.dtype == np.float32

    def test_observation_in_bounds(self, env):
        obs, _ = env.reset()
        assert (obs >= 0.0).all()
        assert (obs <= 1.0).all()


# =============================================================================
#  Reset
# =============================================================================

class TestReset:
    def test_returns_obs_and_info(self, env):
        result = env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_initial_position_is_depot(self, env):
        obs, info = env.reset()
        assert info["current_route"] == [0]

    def test_initial_remaining_capacity_full(self, env):
        env.reset()
        assert env._state.remaining_capacity == env.instance.capacity

    def test_initial_day_and_vehicle(self, env):
        env.reset()
        assert env._state.current_day == 1
        assert env._state.current_vehicle == 1


# =============================================================================
#  Step válido (visitar cliente)
# =============================================================================

class TestStepVisit:
    def test_visit_valid_customer(self, env):
        env.reset()
        # El cliente con índice de matriz 1 es el primer cliente de la instancia
        obs, reward, terminated, truncated, info = env.step(1)
        assert reward < 0  # costo negativo (distancia)
        assert not terminated
        assert info["visited"] == 1
        assert info["current_route"] == [0, 1]

    def test_capacity_decreases_after_visit(self, env):
        env.reset()
        initial = env._state.remaining_capacity
        env.step(1)
        assert env._state.remaining_capacity < initial

    def test_visit_decreases_remaining_visits(self, env):
        env.reset()
        c = env.instance.customers[0]
        env.step(1)
        visits = env._state.visits_completed[c.id]
        assert len(visits) == 1


# =============================================================================
#  Step inválido
# =============================================================================

class TestStepInvalid:
    def test_double_visit_same_day_blocked_by_mask(self, env):
        """Tras visitar un cliente, la máscara debe prohibir visitarlo de nuevo el mismo día."""
        env.reset()
        env.step(1)
        mask = env.action_masks()
        # La acción 1 (volver a visitar el mismo cliente) está enmascarada
        assert mask[1] == False


# =============================================================================
#  Cierre de ruta
# =============================================================================

class TestCloseRoute:
    def test_close_route_returns_to_depot(self, env):
        env.reset()
        env.step(1)  # visitar cliente
        obs, reward, *_ = env.step(0)  # cerrar ruta
        # Después de cerrar, la nueva ruta comienza en el depósito
        assert env._state.current_position == 0

    def test_close_route_advances_vehicle(self, env):
        env.reset()
        env.step(1)
        env.step(0)
        assert env._state.current_vehicle == 2

    def test_close_route_registers_route_in_solution(self, env):
        env.reset()
        env.step(1)
        env.step(0)
        sol = env.get_solution()
        assert len(sol.routes) == 1
        assert sol.routes[0].nodes == [0, 1, 0]


# =============================================================================
#  Episodio aleatorio completo
# =============================================================================

class TestRandomEpisode:
    def test_random_episode_terminates(self):
        """Un episodio con acciones aleatorias válidas debe terminar en tiempo finito."""
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
