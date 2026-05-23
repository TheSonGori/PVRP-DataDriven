"""
Tests del módulo de evaluación (Día 12).

Verifican la mecánica de las funciones de evaluación usando un agente
mínimamente entrenado (SMOKE_TEST_CONFIG). No verifican CALIDAD del agente
(eso requiere entrenamiento largo), solo que las funciones:

    - Devuelven la estructura de datos correcta.
    - Calculan gaps coherentes cuando hay BKS.
    - Producen estadísticas sensatas en la evaluación estocástica.
    - La comparación incluye los tres métodos.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agent.evaluate import (
    EvalResult,
    StochasticEvalResult,
    compare_methods,
    evaluate_deterministic,
    evaluate_greedy,
    evaluate_stochastic,
    evaluate_vns,
)
from src.agent.policy_config import SMOKE_TEST_CONFIG
from src.agent.train import train_agent
from src.data.instance_loader import load_instance


DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


@pytest.fixture(scope="module")
def trained_model():
    """Agente mínimamente entrenado, compartido por todos los tests."""
    instance = load_instance(DATA_DIR / "p01.txt")
    return train_agent(instance, config=SMOKE_TEST_CONFIG)


@pytest.fixture(scope="module")
def instance():
    return load_instance(DATA_DIR / "p01.txt")


class TestDeterministicEval:
    def test_returns_eval_result(self, trained_model, instance):
        result = evaluate_deterministic(trained_model, instance, bks_cost=524.61)
        assert isinstance(result, EvalResult)
        assert result.cost > 0
        assert result.num_routes > 0

    def test_gap_computed_when_bks_given(self, trained_model, instance):
        result = evaluate_deterministic(trained_model, instance, bks_cost=524.61)
        assert result.gap_pct is not None
        # El gap debe ser coherente con el costo y la BKS
        expected = (result.cost - 524.61) / 524.61 * 100
        assert abs(result.gap_pct - expected) < 1e-6

    def test_gap_none_without_bks(self, trained_model, instance):
        result = evaluate_deterministic(trained_model, instance, bks_cost=None)
        assert result.gap_pct is None

    def test_deterministic_is_reproducible(self, trained_model, instance):
        r1 = evaluate_deterministic(trained_model, instance, seed=7)
        r2 = evaluate_deterministic(trained_model, instance, seed=7)
        assert r1.cost == r2.cost


class TestStochasticEval:
    def test_returns_stochastic_result(self, trained_model, instance):
        result = evaluate_stochastic(trained_model, instance, n_runs=5)
        assert isinstance(result, StochasticEvalResult)
        assert result.n_runs == 5

    def test_stats_are_consistent(self, trained_model, instance):
        """La media debe estar entre el peor costo y algún valor razonable;
        la desviación no puede ser negativa.

        Nota: `best_cost` es el mejor costo FACTIBLE, mientras que
        `mean_cost`/`worst_cost` agregan TODAS las corridas (factibles e
        infactibles). Por eso `best_cost` no necesariamente es <= mean_cost:
        una solución infactible puede tener menor costo que la mejor factible.
        """
        result = evaluate_stochastic(trained_model, instance, n_runs=5)
        assert result.std_cost >= 0.0
        assert result.mean_cost <= result.worst_cost + 1e-6
        assert result.best_cost > 0

    def test_feasibility_rate_in_range(self, trained_model, instance):
        result = evaluate_stochastic(trained_model, instance, n_runs=5)
        assert 0.0 <= result.feasibility_rate <= 1.0


class TestBaselineEval:
    def test_greedy_eval(self, instance):
        result = evaluate_greedy(instance, bks_cost=524.61)
        assert result.method == "Greedy"
        assert result.feasible
        assert result.gap_pct is not None

    def test_vns_eval(self, instance):
        result = evaluate_vns(instance, max_iterations=10, bks_cost=524.61)
        assert result.method == "VNS"
        assert result.feasible


class TestCompareMethods:
    def test_comparison_has_all_methods(self, trained_model, instance):
        results = compare_methods(
            trained_model, instance, DATA_DIR,
            n_stochastic_runs=3, vns_iterations=10,
        )
        assert "greedy" in results
        assert "vns" in results
        assert "rl_deterministic" in results
        assert "rl_stochastic" in results
        assert results["bks"] is not None  # p01.res existe

    def test_comparison_bks_loaded(self, trained_model, instance):
        results = compare_methods(
            trained_model, instance, DATA_DIR,
            n_stochastic_runs=3, vns_iterations=10,
        )
        # La BKS de p01 es 524.61
        assert abs(results["bks"] - 524.61) < 1.0
