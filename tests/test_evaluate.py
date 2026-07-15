"""
Tests del módulo de evaluación: verifican la mecánica de evaluate_deterministic,
evaluate_stochastic, evaluate_greedy, evaluate_vns y compare_methods usando un
agente mínimamente entrenado (SMOKE_TEST_CONFIG); no evalúan calidad del agente.

Entrada: la instancia p01 (data/raw/p01.txt) y un modelo entrenado con
SMOKE_TEST_CONFIG.
Salida: aserciones pytest; no retorna valores.
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
    instance = load_instance(DATA_DIR / "p01.txt")
    return train_agent(instance, config=SMOKE_TEST_CONFIG)


@pytest.fixture(scope="module")
def instance():
    return load_instance(DATA_DIR / "p01.txt")


class TestDeterministicEval:

    # evaluate_deterministic devuelve un EvalResult con costo y rutas positivos.
    def test_returns_eval_result(self, trained_model, instance):
        result = evaluate_deterministic(trained_model, instance, bks_cost=524.61)
        assert isinstance(result, EvalResult)
        assert result.cost > 0
        assert result.num_routes > 0

    # El gap se calcula correctamente respecto a la BKS provista.
    def test_gap_computed_when_bks_given(self, trained_model, instance):
        result = evaluate_deterministic(trained_model, instance, bks_cost=524.61)
        assert result.gap_pct is not None
        expected = (result.cost - 524.61) / 524.61 * 100
        assert abs(result.gap_pct - expected) < 1e-6

    # Sin BKS, el gap queda en None.
    def test_gap_none_without_bks(self, trained_model, instance):
        result = evaluate_deterministic(trained_model, instance, bks_cost=None)
        assert result.gap_pct is None

    # Misma semilla produce el mismo costo en la evaluación determinística.
    def test_deterministic_is_reproducible(self, trained_model, instance):
        r1 = evaluate_deterministic(trained_model, instance, seed=7)
        r2 = evaluate_deterministic(trained_model, instance, seed=7)
        assert r1.cost == r2.cost


class TestStochasticEval:

    # evaluate_stochastic devuelve un StochasticEvalResult con el número de corridas pedido.
    def test_returns_stochastic_result(self, trained_model, instance):
        result = evaluate_stochastic(trained_model, instance, n_runs=5)
        assert isinstance(result, StochasticEvalResult)
        assert result.n_runs == 5

    # Las estadísticas agregadas (std, mean vs worst, best_cost) son coherentes.
    def test_stats_are_consistent(self, trained_model, instance):
        result = evaluate_stochastic(trained_model, instance, n_runs=5)
        assert result.std_cost >= 0.0
        assert result.mean_cost <= result.worst_cost + 1e-6
        assert result.best_cost > 0

    # La tasa de factibilidad está en [0, 1].
    def test_feasibility_rate_in_range(self, trained_model, instance):
        result = evaluate_stochastic(trained_model, instance, n_runs=5)
        assert 0.0 <= result.feasibility_rate <= 1.0


class TestBaselineEval:

    # evaluate_greedy produce una solución factible con gap respecto a la BKS.
    def test_greedy_eval(self, instance):
        result = evaluate_greedy(instance, bks_cost=524.61)
        assert result.method == "Greedy"
        assert result.feasible
        assert result.gap_pct is not None

    # evaluate_vns produce una solución factible.
    def test_vns_eval(self, instance):
        result = evaluate_vns(instance, max_iterations=10, bks_cost=524.61)
        assert result.method == "VNS"
        assert result.feasible


class TestCompareMethods:

    # compare_methods incluye los cuatro resultados y la BKS cuando el .res existe.
    def test_comparison_has_all_methods(self, trained_model, instance):
        results = compare_methods(
            trained_model, instance, DATA_DIR,
            n_stochastic_runs=3, vns_iterations=10,
        )
        assert "greedy" in results
        assert "vns" in results
        assert "rl_deterministic" in results
        assert "rl_stochastic" in results
        assert results["bks"] is not None

    # La BKS cargada para p01 coincide con el valor conocido (524.61).
    def test_comparison_bks_loaded(self, trained_model, instance):
        results = compare_methods(
            trained_model, instance, DATA_DIR,
            n_stochastic_runs=3, vns_iterations=10,
        )
        assert abs(results["bks"] - 524.61) < 1.0
