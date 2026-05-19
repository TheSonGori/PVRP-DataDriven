"""
Tests profundos del entorno PVRP (Día 6).

Estos tests cierran la Fase B verificando propiedades de alto nivel que
garantizan la corrección del entorno:

    - Invariantes durante el episodio (capacidad, día, frecuencia).
    - Determinismo bajo misma semilla.
    - Funcionamiento sobre múltiples instancias del dataset.
    - Coherencia entre la solución del entorno y el costo reportado.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.instance_loader import load_instance
from src.environment.pvrp_env import PVRPEnv


DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

# Instancias representativas de cada escala (pequeña, mediana, grande).
# Solo se prueban las que estén físicamente disponibles en data/raw/.
_CANDIDATE_INSTANCES = ["p01", "p02", "p04", "p07", "p23"]
SAMPLE_INSTANCES = [
    name for name in _CANDIDATE_INSTANCES
    if (DATA_DIR / f"{name}.txt").exists()
]


def _run_random_episode(env: PVRPEnv, seed: int = 0, max_steps: int = 5000):
    """Ejecuta un episodio con política aleatoria respetando la máscara."""
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(max_steps):
        mask = env.action_masks()
        valid = np.flatnonzero(mask)
        action = int(rng.choice(valid))
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            return True
    return False


# =============================================================================
#  Invariantes durante el episodio
# =============================================================================

class TestInvariants:
    def test_capacity_never_negative(self):
        """La capacidad restante nunca puede ser negativa."""
        instance = load_instance(DATA_DIR / "p01.txt")
        env = PVRPEnv(instance, seed=0)
        env.reset()

        rng = np.random.default_rng(0)
        terminated = False
        for _ in range(5000):
            mask = env.action_masks()
            valid = np.flatnonzero(mask)
            action = int(rng.choice(valid))
            env.step(action)
            assert env._state.remaining_capacity >= 0, (
                f"Capacidad negativa: {env._state.remaining_capacity}"
            )
            if env._is_episode_done():
                break

    def test_visits_never_exceed_frequency(self):
        """Ningún cliente recibe más visitas que su frecuencia requerida."""
        instance = load_instance(DATA_DIR / "p01.txt")
        env = PVRPEnv(instance, seed=0)
        env.reset()

        rng = np.random.default_rng(0)
        for _ in range(5000):
            mask = env.action_masks()
            valid = np.flatnonzero(mask)
            action = int(rng.choice(valid))
            env.step(action)
            for c in instance.customers:
                visits = env._state.visits_completed[c.id]
                assert len(visits) <= c.frequency, (
                    f"Cliente {c.id}: {len(visits)} visitas > frecuencia {c.frequency}"
                )
            if env._is_episode_done():
                break

    def test_no_duplicate_day_visits(self):
        """Un cliente no puede ser visitado dos veces el mismo día."""
        instance = load_instance(DATA_DIR / "p01.txt")
        env = PVRPEnv(instance, seed=0)
        env.reset()

        rng = np.random.default_rng(0)
        for _ in range(5000):
            mask = env.action_masks()
            valid = np.flatnonzero(mask)
            action = int(rng.choice(valid))
            env.step(action)
            for c_id, days in env._state.visits_completed.items():
                assert len(days) == len(set(days)), (
                    f"Cliente {c_id} visitado dos veces el mismo día: {days}"
                )
            if env._is_episode_done():
                break


# =============================================================================
#  Determinismo
# =============================================================================

class TestDeterminism:
    def test_same_seed_same_trajectory(self):
        """Dos episodios con misma semilla y misma política producen la misma trayectoria."""
        instance = load_instance(DATA_DIR / "p01.txt")

        def run(seed):
            env = PVRPEnv(instance, seed=seed)
            env.reset(seed=seed)
            rng = np.random.default_rng(seed)
            actions = []
            for _ in range(5000):
                mask = env.action_masks()
                valid = np.flatnonzero(mask)
                action = int(rng.choice(valid))
                actions.append(action)
                _, _, terminated, _, _ = env.step(action)
                if terminated:
                    break
            return actions, env.get_solution().total_cost(instance)

        actions_1, cost_1 = run(seed=123)
        actions_2, cost_2 = run(seed=123)
        assert actions_1 == actions_2
        assert cost_1 == cost_2


# =============================================================================
#  Múltiples instancias
# =============================================================================

class TestMultipleInstances:
    @pytest.mark.parametrize("name", SAMPLE_INSTANCES)
    def test_episode_terminates_on_instance(self, name):
        instance = load_instance(DATA_DIR / f"{name}.txt")
        env = PVRPEnv(instance, seed=0)
        # Más pasos para instancias grandes
        max_steps = 50 * instance.num_customers
        assert _run_random_episode(env, seed=0, max_steps=max_steps), (
            f"Episodio en {name} no terminó."
        )

    @pytest.mark.parametrize("name", SAMPLE_INSTANCES)
    def test_solution_local_constraints_on_instance(self, name):
        instance = load_instance(DATA_DIR / f"{name}.txt")
        env = PVRPEnv(instance, seed=0)
        max_steps = 50 * instance.num_customers
        _run_random_episode(env, seed=0, max_steps=max_steps)

        sol = env.get_solution()
        _, violations = sol.is_feasible(instance)
        # Como en Día 5: pueden quedar visitas pendientes pero NO violaciones
        # de capacidad ni patrones inválidos.
        for v in violations:
            assert "capacidad" not in v.lower()
            assert "patrón de visitas" not in v.lower()


# =============================================================================
#  Coherencia costo / acumulación de distancias
# =============================================================================

class TestCostCoherence:
    @pytest.mark.parametrize("name", SAMPLE_INSTANCES)
    def test_solution_cost_matches_route_cost_sum(self, name):
        """
        El costo de la solución (calculado por `Solution.total_cost`) debe ser
        idéntico a la suma de costos de cada ruta calculada independientemente.
        Esto verifica que el cálculo de costo es consistente.
        """
        instance = load_instance(DATA_DIR / f"{name}.txt")
        env = PVRPEnv(instance, seed=0)
        max_steps = 50 * instance.num_customers
        _run_random_episode(env, seed=0, max_steps=max_steps)

        sol = env.get_solution()
        total = sol.total_cost(instance)

        # Suma manual ruta por ruta
        from src.utils.distance import build_distance_matrix, build_id_to_index_map
        matrix = build_distance_matrix(instance)
        id_to_idx = build_id_to_index_map(instance)
        manual = sum(
            sol._route_cost(r, matrix, id_to_idx) for r in sol.routes
        )
        assert abs(total - manual) < 1e-6
