"""
Tests del VNS: mejora (o iguala) la solución inicial de Greedy, produce
soluciones factibles, es determinístico y respeta el tope de tiempo.

Entrada: instancias del dataset (data/raw/p01.txt, p23.txt) y su BKS
(p01.res).
Salida: aserciones pytest; no retorna valores.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.baselines.greedy import greedy_solve
from src.baselines.vns import vns_solve
from src.data.instance_loader import load_instance
from src.data.solution_loader import load_solution


DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

_CANDIDATE_INSTANCES = ["p01", "p23"]
SAMPLE_INSTANCES = [
    name for name in _CANDIDATE_INSTANCES
    if (DATA_DIR / f"{name}.txt").exists()
]


class TestVNSP01:

    @pytest.fixture(scope="class")
    def setup(self):
        instance = load_instance(DATA_DIR / "p01.txt")
        result = vns_solve(instance, max_iterations=50, seed=0, verbose=False)
        return instance, result

    # La solución del VNS sobre p01 es factible.
    def test_solution_is_feasible(self, setup):
        instance, result = setup
        feasible, violations = result.solution.is_feasible(instance)
        assert feasible, f"VNS infactible en p01: {violations[:3]}"

    # El VNS mejora sustancialmente sobre Greedy puro (sin búsqueda local) en p01.
    def test_improves_over_pure_greedy(self, setup):
        from src.baselines.greedy import greedy_solve
        instance, result = setup
        pure_greedy_cost = greedy_solve(instance).total_cost(instance)
        assert result.final_cost < pure_greedy_cost * 0.95, (
            f"VNS={result.final_cost:.2f} debería ser <95% de Greedy={pure_greedy_cost:.2f}"
        )

    # El VNS no puede superar (ser mejor que) la BKS.
    def test_better_than_bks_lower_bound(self, setup):
        instance, result = setup
        bks = load_solution(DATA_DIR / "p01.res")
        assert result.final_cost >= bks.reported_cost - 0.5


class TestVNSMultipleInstances:

    # El VNS produce una solución factible en cada instancia de muestra.
    @pytest.mark.parametrize("name", SAMPLE_INSTANCES)
    def test_produces_feasible_on(self, name):
        instance = load_instance(DATA_DIR / f"{name}.txt")
        result = vns_solve(instance, max_iterations=20, seed=0, verbose=False)
        feasible, violations = result.solution.is_feasible(instance)
        assert feasible, f"VNS infactible en {name}: {violations[:2]}"


class TestVNSDeterminism:

    # El VNS con la misma semilla produce siempre el mismo costo final.
    def test_same_seed_same_result(self):
        instance = load_instance(DATA_DIR / "p01.txt")
        r1 = vns_solve(instance, max_iterations=30, seed=42, verbose=False)
        r2 = vns_solve(instance, max_iterations=30, seed=42, verbose=False)
        assert r1.final_cost == r2.final_cost


class TestVNSTimeLimit:

    # Con un time_limit, el VNS termina en un tiempo acorde (con margen de tolerancia).
    def test_respects_time_limit(self):
        instance = load_instance(DATA_DIR / "p01.txt")
        result = vns_solve(
            instance, max_iterations=10000, time_limit=2.0, seed=0, verbose=False
        )
        assert result.elapsed_time < 10.0
