"""
Tests de propiedades de alto nivel de PVRPEnv: invariantes durante el
episodio (capacidad, frecuencia, visitas por día), determinismo bajo la
misma semilla, funcionamiento sobre múltiples instancias, y coherencia entre
el costo de la solución y la suma de costos por ruta.

Entrada: instancias del dataset (data/raw/*.txt) presentes en disco.
Salida: aserciones pytest; no retorna valores.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.instance_loader import load_instance
from src.environment.pvrp_env import PVRPEnv


DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

_CANDIDATE_INSTANCES = ["p01", "p02", "p04", "p07", "p23"]
SAMPLE_INSTANCES = [
    name for name in _CANDIDATE_INSTANCES
    if (DATA_DIR / f"{name}.txt").exists()
]


# Ejecuta un episodio con política aleatoria respetando la máscara; devuelve si terminó.
def _run_random_episode(env: PVRPEnv, seed: int = 0, max_steps: int = 5000):
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


class TestInvariants:

    # La capacidad restante nunca es negativa a lo largo de un episodio.
    def test_capacity_never_negative(self):
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

    # Ningún cliente recibe más visitas que su frecuencia requerida.
    def test_visits_never_exceed_frequency(self):
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

    # Un cliente no puede ser visitado dos veces el mismo día.
    def test_no_duplicate_day_visits(self):
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


class TestDeterminism:

    # Dos episodios con la misma semilla y política producen la misma trayectoria y costo.
    def test_same_seed_same_trajectory(self):
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


class TestMultipleInstances:

    # El episodio termina en un número finito de pasos en cada instancia de muestra.
    @pytest.mark.parametrize("name", SAMPLE_INSTANCES)
    def test_episode_terminates_on_instance(self, name):
        instance = load_instance(DATA_DIR / f"{name}.txt")
        env = PVRPEnv(instance, seed=0)
        max_steps = 50 * instance.num_customers
        assert _run_random_episode(env, seed=0, max_steps=max_steps), (
            f"Episodio en {name} no terminó."
        )

    # La solución resultante no viola restricciones locales de capacidad ni de patrón.
    @pytest.mark.parametrize("name", SAMPLE_INSTANCES)
    def test_solution_local_constraints_on_instance(self, name):
        instance = load_instance(DATA_DIR / f"{name}.txt")
        env = PVRPEnv(instance, seed=0)
        max_steps = 50 * instance.num_customers
        _run_random_episode(env, seed=0, max_steps=max_steps)

        sol = env.get_solution()
        _, violations = sol.is_feasible(instance)
        for v in violations:
            assert "capacidad" not in v.lower()
            assert "patrón de visitas" not in v.lower()


class TestCostCoherence:

    # El costo total de Solution coincide con la suma manual de costos por ruta.
    @pytest.mark.parametrize("name", SAMPLE_INSTANCES)
    def test_solution_cost_matches_route_cost_sum(self, name):
        instance = load_instance(DATA_DIR / f"{name}.txt")
        env = PVRPEnv(instance, seed=0)
        max_steps = 50 * instance.num_customers
        _run_random_episode(env, seed=0, max_steps=max_steps)

        sol = env.get_solution()
        total = sol.total_cost(instance)

        from src.utils.distance import build_distance_matrix, build_id_to_index_map
        matrix = build_distance_matrix(instance)
        id_to_idx = build_id_to_index_map(instance)
        manual = sum(
            sol._route_cost(r, matrix, id_to_idx) for r in sol.routes
        )
        assert abs(total - manual) < 1e-6
