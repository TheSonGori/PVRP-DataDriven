"""
Tests para el parser de soluciones del NEO Research Group (`.res`) y la clase
`Solution`.

El test crítico es la validación contra la Best Known Solution (BKS) de p01,
cuyo costo reportado es 524.61. Si nuestro cálculo de costo total coincide
con ese valor (dentro de tolerancia numérica), entonces:

    - El parser de instancias funciona (Día 2).
    - El cálculo de distancias funciona.
    - El parser de soluciones funciona.
    - La validación de factibilidad funciona.

Es la primera prueba end-to-end del proyecto.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.instance_loader import load_instance
from src.data.solution_loader import load_solution
from src.utils.solution import Route, Solution


DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


# =============================================================================
#  Tests del parser .res
# =============================================================================

class TestSolutionLoader:
    """Verifica el parsing del archivo .res de p01."""

    @pytest.fixture(scope="class")
    def solution(self) -> Solution:
        return load_solution(DATA_DIR / "p01.res")

    def test_reported_cost(self, solution):
        assert solution.reported_cost == pytest.approx(524.61)

    def test_number_of_routes(self, solution):
        # p01 BKS: 2 rutas en día 1, 3 rutas en día 2 = 5 rutas
        assert len(solution.routes) == 5

    def test_routes_per_day(self, solution):
        day1 = solution.routes_by_day(1)
        day2 = solution.routes_by_day(2)
        assert len(day1) == 2
        assert len(day2) == 3

    def test_routes_start_and_end_at_depot(self, solution):
        for r in solution.routes:
            assert r.nodes[0] == 0
            assert r.nodes[-1] == 0

    def test_first_route_content(self, solution):
        # Día 1, Vehículo 1: 0 8 26 31 28 3 36 35 20 22 1 32 0
        r1 = solution.routes_by_day(1)[0]
        assert r1.day == 1
        assert r1.vehicle_id == 1
        assert r1.nodes == [0, 8, 26, 31, 28, 3, 36, 35, 20, 22, 1, 32, 0]


# =============================================================================
#  Test crítico: validación end-to-end contra la BKS
# =============================================================================

class TestBKSValidation:
    """
    Test integral: el costo recalculado a partir de las coordenadas de p01.txt
    y las rutas de p01.res debe coincidir con el costo reportado (524.61).
    """

    @pytest.fixture(scope="class")
    def instance(self):
        return load_instance(DATA_DIR / "p01.txt")

    @pytest.fixture(scope="class")
    def solution(self):
        return load_solution(DATA_DIR / "p01.res")

    def test_total_cost_matches_reported(self, instance, solution):
        recalculated = solution.total_cost(instance)
        # Tolerancia: las distancias euclidianas pueden tener pequeñas
        # diferencias de redondeo respecto al valor publicado.
        assert recalculated == pytest.approx(
            solution.reported_cost,
            abs=0.5  # tolerancia de medio kilómetro acumulado
        )

    def test_solution_is_feasible(self, instance, solution):
        is_feasible, violations = solution.is_feasible(instance)
        assert is_feasible, f"BKS no es factible: {violations}"

    def test_all_customers_visited(self, instance, solution):
        """Cada cliente de p01 (todos con frecuencia 1) aparece exactamente una vez."""
        all_visited = []
        for r in solution.routes:
            all_visited.extend(r.customers)

        # En p01 todos los clientes tienen frecuencia 1
        assert len(all_visited) == instance.num_customers
        assert len(set(all_visited)) == instance.num_customers


# =============================================================================
#  Tests de la clase Solution con datos sintéticos
# =============================================================================

class TestSolutionMethods:
    """Verifica métodos de Solution con casos construidos manualmente."""

    def test_route_customers_excludes_depot(self):
        r = Route(day=1, vehicle_id=1, nodes=[0, 5, 7, 0])
        assert r.customers == [5, 7]

    def test_empty_solution(self):
        s = Solution()
        assert len(s.routes) == 0
        assert s.reported_cost is None

    def test_routes_by_day_filter(self):
        s = Solution(routes=[
            Route(day=1, vehicle_id=1, nodes=[0, 1, 0]),
            Route(day=2, vehicle_id=1, nodes=[0, 2, 0]),
            Route(day=1, vehicle_id=2, nodes=[0, 3, 0]),
        ])
        assert len(s.routes_by_day(1)) == 2
        assert len(s.routes_by_day(2)) == 1
        assert len(s.routes_by_day(3)) == 0


# =============================================================================
#  Tests de infactibilidad (regresión)
# =============================================================================

class TestInfeasibilityDetection:
    """Verifica que la validación detecta correctamente soluciones inválidas."""

    @pytest.fixture(scope="class")
    def instance(self):
        return load_instance(DATA_DIR / "p01.txt")

    def test_detects_missing_customer(self, instance):
        """Una solución sin todos los clientes debe ser infactible."""
        # Solo visitamos un cliente, dejando todos los demás sin atender
        partial = Solution(routes=[
            Route(day=1, vehicle_id=1, nodes=[0, 1, 0]),
        ])
        is_feasible, violations = partial.is_feasible(instance)
        assert not is_feasible
        assert len(violations) > 0

    def test_detects_capacity_violation(self, instance):
        """Una ruta que excede capacidad debe marcarse como infactible."""
        # Construimos una ruta artificial con todos los clientes (demanda total
        # superará ampliamente la capacidad de 160).
        all_ids = [c.id for c in instance.customers]
        nodes = [0] + all_ids + [0]
        s = Solution(routes=[Route(day=1, vehicle_id=1, nodes=nodes)])
        is_feasible, violations = s.is_feasible(instance)
        assert not is_feasible
        assert any("capacidad" in v.lower() for v in violations)

    def test_detects_route_not_starting_at_depot(self, instance):
        """Una ruta que no empieza en el depósito debe ser detectada."""
        bad = Solution(routes=[
            Route(day=1, vehicle_id=1, nodes=[1, 2, 0]),  # no empieza en 0
        ])
        is_feasible, violations = bad.is_feasible(instance)
        assert not is_feasible
        assert any("depósito" in v.lower() for v in violations)
