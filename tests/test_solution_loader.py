"""
Tests del parser de soluciones .res del NEO Research Group y de la clase
Solution. El test crítico valida que el costo recalculado desde p01.txt +
p01.res coincide con el costo publicado de la BKS (524.61), lo que confirma
end-to-end el parser de instancias, el cálculo de distancias, el parser de
soluciones y la validación de factibilidad.

Entrada: la instancia p01 (data/raw/p01.txt) y su solución de referencia
(data/raw/p01.res).
Salida: aserciones pytest; no retorna valores.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.instance_loader import load_instance
from src.data.solution_loader import load_solution
from src.utils.solution import Route, Solution


DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


class TestSolutionLoader:

    @pytest.fixture(scope="class")
    def solution(self) -> Solution:
        return load_solution(DATA_DIR / "p01.res")

    # El costo total reportado en el .res es 524.61.
    def test_reported_cost(self, solution):
        assert solution.reported_cost == pytest.approx(524.61)

    # p01 BKS tiene 2 rutas en día 1 y 3 rutas en día 2 = 5 rutas.
    def test_number_of_routes(self, solution):
        assert len(solution.routes) == 5

    # El número de rutas por día coincide con lo esperado.
    def test_routes_per_day(self, solution):
        day1 = solution.routes_by_day(1)
        day2 = solution.routes_by_day(2)
        assert len(day1) == 2
        assert len(day2) == 3

    # Toda ruta empieza y termina en el depósito (nodo 0).
    def test_routes_start_and_end_at_depot(self, solution):
        for r in solution.routes:
            assert r.nodes[0] == 0
            assert r.nodes[-1] == 0

    # El contenido de la primera ruta (día 1, vehículo 1) coincide con el .res.
    def test_first_route_content(self, solution):
        r1 = solution.routes_by_day(1)[0]
        assert r1.day == 1
        assert r1.vehicle_id == 1
        assert r1.nodes == [0, 8, 26, 31, 28, 3, 36, 35, 20, 22, 1, 32, 0]


class TestBKSValidation:

    @pytest.fixture(scope="class")
    def instance(self):
        return load_instance(DATA_DIR / "p01.txt")

    @pytest.fixture(scope="class")
    def solution(self):
        return load_solution(DATA_DIR / "p01.res")

    # El costo recalculado desde las coordenadas coincide con el costo publicado.
    def test_total_cost_matches_reported(self, instance, solution):
        recalculated = solution.total_cost(instance)
        assert recalculated == pytest.approx(
            solution.reported_cost,
            abs=0.5
        )

    # La BKS es una solución factible.
    def test_solution_is_feasible(self, instance, solution):
        is_feasible, violations = solution.is_feasible(instance)
        assert is_feasible, f"BKS no es factible: {violations}"

    # Cada cliente de p01 (todos con frecuencia 1) aparece exactamente una vez.
    def test_all_customers_visited(self, instance, solution):
        all_visited = []
        for r in solution.routes:
            all_visited.extend(r.customers)

        assert len(all_visited) == instance.num_customers
        assert len(set(all_visited)) == instance.num_customers


class TestSolutionMethods:

    # Route.customers excluye los nodos de depósito.
    def test_route_customers_excludes_depot(self):
        r = Route(day=1, vehicle_id=1, nodes=[0, 5, 7, 0])
        assert r.customers == [5, 7]

    # Una Solution vacía no tiene rutas ni costo reportado.
    def test_empty_solution(self):
        s = Solution()
        assert len(s.routes) == 0
        assert s.reported_cost is None

    # routes_by_day filtra correctamente las rutas de cada día.
    def test_routes_by_day_filter(self):
        s = Solution(routes=[
            Route(day=1, vehicle_id=1, nodes=[0, 1, 0]),
            Route(day=2, vehicle_id=1, nodes=[0, 2, 0]),
            Route(day=1, vehicle_id=2, nodes=[0, 3, 0]),
        ])
        assert len(s.routes_by_day(1)) == 2
        assert len(s.routes_by_day(2)) == 1
        assert len(s.routes_by_day(3)) == 0


class TestInfeasibilityDetection:

    @pytest.fixture(scope="class")
    def instance(self):
        return load_instance(DATA_DIR / "p01.txt")

    # Una solución que deja clientes sin visitar es detectada como infactible.
    def test_detects_missing_customer(self, instance):
        partial = Solution(routes=[
            Route(day=1, vehicle_id=1, nodes=[0, 1, 0]),
        ])
        is_feasible, violations = partial.is_feasible(instance)
        assert not is_feasible
        assert len(violations) > 0

    # Una ruta que excede la capacidad es marcada como infactible.
    def test_detects_capacity_violation(self, instance):
        all_ids = [c.id for c in instance.customers]
        nodes = [0] + all_ids + [0]
        s = Solution(routes=[Route(day=1, vehicle_id=1, nodes=nodes)])
        is_feasible, violations = s.is_feasible(instance)
        assert not is_feasible
        assert any("capacidad" in v.lower() for v in violations)

    # Una ruta que no empieza en el depósito es detectada.
    def test_detects_route_not_starting_at_depot(self, instance):
        bad = Solution(routes=[
            Route(day=1, vehicle_id=1, nodes=[1, 2, 0]),
        ])
        is_feasible, violations = bad.is_feasible(instance)
        assert not is_feasible
        assert any("depósito" in v.lower() for v in violations)
