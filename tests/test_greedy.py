"""
Tests de la heurística Greedy: produce soluciones factibles, mejora
claramente sobre la política aleatoria, no supera a la BKS, y es
determinística.

Entrada: instancias del dataset (data/raw/*.txt) presentes en disco.
Salida: aserciones pytest; no retorna valores.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.baselines.greedy import greedy_solve
from src.data.instance_loader import load_instance
from src.data.solution_loader import load_solution


DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

_CANDIDATE_INSTANCES = ["p01", "p02", "p04", "p07", "p23"]
SAMPLE_INSTANCES = [
    name for name in _CANDIDATE_INSTANCES
    if (DATA_DIR / f"{name}.txt").exists()
]


class TestGreedyP01:

    @pytest.fixture(scope="class")
    def setup(self):
        instance = load_instance(DATA_DIR / "p01.txt")
        solution = greedy_solve(instance)
        return instance, solution

    # La solución de Greedy sobre p01 es factible.
    def test_solution_is_feasible(self, setup):
        instance, solution = setup
        feasible, violations = solution.is_feasible(instance)
        assert feasible, f"Greedy infactible en p01: {violations[:3]}"

    # El costo de Greedy es sensiblemente menor que el de la política aleatoria (~1590).
    def test_better_than_random_baseline(self, setup):
        instance, solution = setup
        cost = solution.total_cost(instance)
        assert cost < 1500, (
            f"Greedy debería ser mejor que aleatorio (~1590), pero obtuvo {cost:.2f}"
        )

    # Greedy no supera a la BKS (524.61); si lo hiciera habría un bug en el cálculo de costo.
    def test_not_better_than_bks(self, setup):
        instance, solution = setup
        bks = load_solution(DATA_DIR / "p01.res")
        assert solution.total_cost(instance) >= bks.reported_cost - 0.5


class TestGreedyMultipleInstances:

    # Greedy produce una solución factible en cada instancia de muestra.
    @pytest.mark.parametrize("name", SAMPLE_INSTANCES)
    def test_produces_feasible_solution(self, name):
        instance = load_instance(DATA_DIR / f"{name}.txt")
        solution = greedy_solve(instance)
        feasible, violations = solution.is_feasible(instance)
        assert feasible, (
            f"Greedy infactible en {name}: {violations[:2]}"
        )


class TestGreedyDeterminism:

    # Dos llamadas a greedy_solve sobre la misma instancia dan el mismo resultado.
    def test_two_runs_same_solution(self):
        instance = load_instance(DATA_DIR / "p01.txt")
        s1 = greedy_solve(instance)
        s2 = greedy_solve(instance)
        assert s1.total_cost(instance) == s2.total_cost(instance)
        assert len(s1.routes) == len(s2.routes)
