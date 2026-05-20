"""
Tests de la heurística Greedy (Día 7).

Verifican que Greedy:
    - Produce soluciones FACTIBLES en instancias razonables.
    - Tiene costo significativamente mejor que la política aleatoria.
    - Tiene costo significativamente peor (o como mucho igual) que la BKS.
    - Es determinístico (misma entrada → misma salida).
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
    """Tests específicos sobre p01."""

    @pytest.fixture(scope="class")
    def setup(self):
        instance = load_instance(DATA_DIR / "p01.txt")
        solution = greedy_solve(instance)
        return instance, solution

    def test_solution_is_feasible(self, setup):
        instance, solution = setup
        feasible, violations = solution.is_feasible(instance)
        assert feasible, f"Greedy infactible en p01: {violations[:3]}"

    def test_better_than_random_baseline(self, setup):
        """El costo de Greedy debe ser sensiblemente menor que el de la
        política aleatoria (~1590 según el notebook 02)."""
        instance, solution = setup
        cost = solution.total_cost(instance)
        assert cost < 1500, (
            f"Greedy debería ser mejor que aleatorio (~1590), pero obtuvo {cost:.2f}"
        )

    def test_not_better_than_bks(self, setup):
        """Greedy no debería superar la BKS (524.61); si lo hiciera, hay un bug
        en el cálculo de costo."""
        instance, solution = setup
        bks = load_solution(DATA_DIR / "p01.res")
        assert solution.total_cost(instance) >= bks.reported_cost - 0.5


class TestGreedyMultipleInstances:
    """Greedy debe funcionar en todas las instancias del dataset."""

    @pytest.mark.parametrize("name", SAMPLE_INSTANCES)
    def test_produces_feasible_solution(self, name):
        instance = load_instance(DATA_DIR / f"{name}.txt")
        solution = greedy_solve(instance)
        feasible, violations = solution.is_feasible(instance)
        assert feasible, (
            f"Greedy infactible en {name}: {violations[:2]}"
        )


class TestGreedyDeterminism:
    def test_two_runs_same_solution(self):
        """Greedy no usa aleatoriedad: dos llamadas dan el mismo resultado."""
        instance = load_instance(DATA_DIR / "p01.txt")
        s1 = greedy_solve(instance)
        s2 = greedy_solve(instance)
        assert s1.total_cost(instance) == s2.total_cost(instance)
        assert len(s1.routes) == len(s2.routes)
