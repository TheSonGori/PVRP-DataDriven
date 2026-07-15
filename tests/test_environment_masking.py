"""
Tests de la máscara de acciones y de la recompensa terminal: forma y
propiedades básicas de la máscara, su comportamiento ante capacidad/día
visitado, factibilidad local de episodios generados respetando la máscara,
y aplicación correcta del bonus/penalización terminal.

Entrada: la instancia p01 (data/raw/p01.txt).
Salida: aserciones pytest; no retorna valores.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.instance_loader import load_instance
from src.environment.pvrp_env import PVRPEnv
from src.environment.reward import RewardConfig


DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


@pytest.fixture
def env():
    instance = load_instance(DATA_DIR / "p01.txt")
    return PVRPEnv(instance, seed=42)


class TestMaskShape:

    # La máscara tiene el tamaño del espacio de acciones y es booleana.
    def test_mask_size(self, env):
        env.reset()
        mask = env.action_masks()
        assert mask.shape == (env.action_space.n,)
        assert mask.dtype == bool

    # Al iniciar el episodio no se permite cerrar, pero sí hay clientes visitables.
    def test_mask_has_valid_actions_initially(self, env):
        env.reset()
        mask = env.action_masks()
        assert mask[0] == False
        assert mask[1:].any()

    # En ningún estado intermedio la máscara queda completamente vacía.
    def test_mask_always_has_at_least_one_valid_action(self, env):
        env.reset()
        rng = np.random.default_rng(7)
        for _ in range(500):
            mask = env.action_masks()
            assert mask.any(), "Máscara vacía: el agente no puede continuar."
            valid = np.flatnonzero(mask)
            action = int(rng.choice(valid))
            _, _, terminated, _, _ = env.step(action)
            if terminated:
                break


class TestMaskBehavior:

    # Un cliente ya visitado queda enmascarado.
    def test_visited_customer_is_masked(self, env):
        env.reset()
        env.step(1)
        mask = env.action_masks()
        assert mask[1] == False

    # Tras la primera visita, cerrar ruta se vuelve una acción válida.
    def test_close_route_becomes_valid_after_first_visit(self, env):
        env.reset()
        env.step(1)
        mask = env.action_masks()
        assert mask[0] == True

    # Cuando la capacidad restante no alcanza, la acción de ese cliente queda enmascarada.
    def test_high_demand_customer_masked_when_capacity_low(self):
        instance = load_instance(DATA_DIR / "p01.txt")
        env = PVRPEnv(instance, seed=0)
        env.reset()

        rng = np.random.default_rng(0)
        for _ in range(8):
            mask = env.action_masks()
            valid_visits = [i for i in range(1, env.action_space.n) if mask[i]]
            if not valid_visits:
                break
            env.step(int(rng.choice(valid_visits)))

        mask = env.action_masks()
        for cust_idx in range(1, env.action_space.n):
            if mask[cust_idx]:
                continue
            cust_id = env.state_encoder.idx_to_id[cust_idx]
            customer = instance.get_customer(cust_id)
            already_visited = cust_id in env._state.visited_today
            no_capacity = customer.demand > env._state.remaining_capacity
            assert already_visited or no_capacity, (
                f"Acción {cust_idx} enmascarada sin razón clara"
            )


class TestEpisodeFeasibility:

    # Un episodio respetando la máscara no genera violaciones de capacidad ni de patrón.
    def test_masked_episode_respects_local_constraints(self):
        instance = load_instance(DATA_DIR / "p01.txt")
        env = PVRPEnv(instance, seed=0)
        env.reset()

        rng = np.random.default_rng(0)
        terminated = False
        for _ in range(5000):
            mask = env.action_masks()
            valid = np.flatnonzero(mask)
            action = int(rng.choice(valid))
            _, _, terminated, _, _ = env.step(action)
            if terminated:
                break

        assert terminated
        sol = env.get_solution()
        feasible, violations = sol.is_feasible(instance)

        for v in violations:
            v_lower = v.lower()
            assert "capacidad" not in v_lower, f"Violación de capacidad inesperada: {v}"
            assert "patrón de visitas" not in v_lower, f"Patrón inválido inesperado: {v}"

    # El costo de la solución coincide con la distancia acumulada reportada en info.
    def test_solution_cost_matches_accumulated_distance(self):
        instance = load_instance(DATA_DIR / "p01.txt")
        env = PVRPEnv(instance, seed=0)
        env.reset()

        accumulated_distance = 0.0
        rng = np.random.default_rng(0)
        terminated = False
        for _ in range(5000):
            mask = env.action_masks()
            valid = np.flatnonzero(mask)
            action = int(rng.choice(valid))
            _, _, terminated, _, info = env.step(action)
            if "distance" in info:
                accumulated_distance += info["distance"]
            if terminated:
                break

        sol = env.get_solution()
        cost = sol.total_cost(instance)
        assert abs(cost - accumulated_distance) < 0.1


class TestTerminalReward:

    # terminal_reward aplica el bonus si es factible y la penalización si no.
    def test_terminal_bonus_signature(self):
        from src.environment.reward import RewardConfig, terminal_reward
        cfg = RewardConfig(terminal_bonus=100.0, infeasibility_penalty=-500.0)
        assert terminal_reward(True, cfg) == 100.0
        assert terminal_reward(False, cfg) == -500.0

    # La penalización crece proporcionalmente con el número de visitas faltantes.
    def test_terminal_penalty_scales_with_missing_visits(self):
        from src.environment.reward import RewardConfig, terminal_reward
        cfg = RewardConfig(
            terminal_bonus=100.0,
            infeasibility_penalty=-500.0,
            per_missing_penalty=-50.0,
        )
        assert terminal_reward(False, cfg, num_missing_visits=0) == -500.0
        assert terminal_reward(False, cfg, num_missing_visits=10) == -1000.0
        assert terminal_reward(False, cfg, num_missing_visits=49) == -2950.0
        assert terminal_reward(True, cfg, num_missing_visits=0) == 100.0

    # El último step del episodio refleja el bonus o la penalización terminal.
    def test_terminal_reward_reflected_in_last_step(self):
        instance = load_instance(DATA_DIR / "p01.txt")
        cfg = RewardConfig(terminal_bonus=1000.0, infeasibility_penalty=-1000.0)
        env = PVRPEnv(instance, reward_config=cfg, seed=0)
        env.reset()

        rng = np.random.default_rng(0)
        last_reward = None
        last_info = None
        terminated = False
        for _ in range(5000):
            mask = env.action_masks()
            valid = np.flatnonzero(mask)
            action = int(rng.choice(valid))
            _, reward, terminated, _, info = env.step(action)
            last_reward = reward
            last_info = info
            if terminated:
                break

        assert terminated
        feasible = last_info.get("is_feasible", False)
        if feasible:
            assert last_reward > 800, f"Bonus esperado, last_reward={last_reward}"
        else:
            assert last_reward < -800, f"Penalización esperada, last_reward={last_reward}"
