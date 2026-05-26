"""
Tests del Día 5: máscara de acciones y recompensa terminal.

Verifican que:
    - La máscara devuelve un vector de tamaño correcto.
    - La máscara prohíbe acciones que violan capacidad, frecuencia, día visitado.
    - La máscara siempre tiene al menos una acción válida (no quedan estados sin salida).
    - Episodios completos generados por máscara producen soluciones factibles.
    - La recompensa terminal se aplica al final del episodio.
    - El costo acumulado (sumando todas las distancias) coincide con el costo
      de la solución calculado externamente vía `Solution.total_cost()`.
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


# =============================================================================
#  Máscara: forma y propiedades básicas
# =============================================================================

class TestMaskShape:
    def test_mask_size(self, env):
        env.reset()
        mask = env.action_masks()
        assert mask.shape == (env.action_space.n,)
        assert mask.dtype == bool

    def test_mask_has_valid_actions_initially(self, env):
        env.reset()
        mask = env.action_masks()
        # Al inicio del episodio NO se permite cerrar inmediatamente,
        # pero debe haber clientes visitables.
        assert mask[0] == False  # no cerrar al arrancar
        assert mask[1:].any()

    def test_mask_always_has_at_least_one_valid_action(self, env):
        """En ningún estado intermedio la máscara debe quedar completamente vacía."""
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


# =============================================================================
#  Máscara: comportamiento específico
# =============================================================================

class TestMaskBehavior:
    def test_visited_customer_is_masked(self, env):
        env.reset()
        env.step(1)  # visitar cliente con índice 1
        mask = env.action_masks()
        assert mask[1] == False

    def test_close_route_becomes_valid_after_first_visit(self, env):
        env.reset()
        env.step(1)
        mask = env.action_masks()
        assert mask[0] == True

    def test_high_demand_customer_masked_when_capacity_low(self):
        """
        Si el vehículo no tiene capacidad para un cliente, esa acción se
        enmascara. Construimos un escenario donde tras varias visitas
        la capacidad ya no alcanza.
        """
        instance = load_instance(DATA_DIR / "p01.txt")
        env = PVRPEnv(instance, seed=0)
        env.reset()

        # Visitar varios clientes hasta que la capacidad se agote significativamente
        rng = np.random.default_rng(0)
        for _ in range(8):
            mask = env.action_masks()
            valid_visits = [i for i in range(1, env.action_space.n) if mask[i]]
            if not valid_visits:
                break
            env.step(int(rng.choice(valid_visits)))

        # Ahora verificamos: para cada acción enmascarada, su demanda excede la capacidad
        # restante (o ya fue visitada).
        mask = env.action_masks()
        for cust_idx in range(1, env.action_space.n):
            if mask[cust_idx]:
                continue
            cust_id = env.state_encoder.idx_to_id[cust_idx]
            customer = instance.get_customer(cust_id)
            # Razones posibles para enmascarar: capacidad o ya visitado o frecuencia.
            already_visited = cust_id in env._state.visited_today
            no_capacity = customer.demand > env._state.remaining_capacity
            assert already_visited or no_capacity, (
                f"Acción {cust_idx} enmascarada sin razón clara"
            )


# =============================================================================
#  Episodios completos: factibilidad por construcción
# =============================================================================

class TestEpisodeFeasibility:
    def test_masked_episode_respects_local_constraints(self):
        """
        Un episodio con política aleatoria pero respetando la máscara puede
        terminar siendo INFACTIBLE globalmente (clientes sin visitar) — la
        factibilidad global la debe aprender el agente. Pero las restricciones
        LOCALES (capacidad y patrón) deben cumplirse en cada ruta producida.

        Este test verifica que NO aparecen violaciones de capacidad ni
        de patrones inválidos en la solución resultante.
        """
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

        # Las únicas violaciones aceptables provienen de frecuencias incompletas;
        # NO debe haber violaciones de capacidad ni de patrones inválidos.
        for v in violations:
            v_lower = v.lower()
            assert "capacidad" not in v_lower, f"Violación de capacidad inesperada: {v}"
            assert "patrón de visitas" not in v_lower, f"Patrón inválido inesperado: {v}"

    def test_solution_cost_matches_accumulated_distance(self):
        """
        El costo total de la solución (suma de distancias por ruta) debe
        coincidir con la suma acumulada de los componentes de distancia
        de cada recompensa (es decir, ignorando el bonus terminal).
        """
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
        # Tolerancia generosa: los errores de redondeo se acumulan a lo largo del episodio
        assert abs(cost - accumulated_distance) < 0.1


# =============================================================================
#  Recompensa terminal
# =============================================================================

class TestTerminalReward:
    def test_terminal_bonus_signature(self):
        """La función `terminal_reward` aplica bonus si factible, penalización si no."""
        from src.environment.reward import RewardConfig, terminal_reward
        cfg = RewardConfig(terminal_bonus=100.0, infeasibility_penalty=-500.0)
        assert terminal_reward(True, cfg) == 100.0
        assert terminal_reward(False, cfg) == -500.0

    def test_terminal_penalty_scales_with_missing_visits(self):
        """La penalización crece con el número de visitas sin realizar,
        evitando el atajo de terminar episodios triviales (reward hacking)."""
        from src.environment.reward import RewardConfig, terminal_reward
        cfg = RewardConfig(
            terminal_bonus=100.0,
            infeasibility_penalty=-500.0,
            per_missing_penalty=-50.0,
        )
        # Sin faltantes: solo la penalización base.
        assert terminal_reward(False, cfg, num_missing_visits=0) == -500.0
        # Con faltantes: base + proporcional.
        assert terminal_reward(False, cfg, num_missing_visits=10) == -1000.0
        assert terminal_reward(False, cfg, num_missing_visits=49) == -2950.0
        # Si es factible, el número de faltantes (0) no afecta el bonus.
        assert terminal_reward(True, cfg, num_missing_visits=0) == 100.0

    def test_terminal_reward_reflected_in_last_step(self):
        """
        El último step de un episodio debe incluir el componente terminal:
        si la solución resultante es factible, debe haber bonus; si no,
        penalización. Verificamos que la última recompensa contiene el
        magnitud esperada respecto a la configuración.
        """
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
            # bonus 1000 - distancia del paso final (típicamente < 100)
            assert last_reward > 800, f"Bonus esperado, last_reward={last_reward}"
        else:
            # penalización -1000 - distancia del paso final
            assert last_reward < -800, f"Penalización esperada, last_reward={last_reward}"
