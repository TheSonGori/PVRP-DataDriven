"""
Tests para los métodos de la clase `Solution` agregados en el Día 3:
construcción incremental, consultas de patrón asignado y resumen estadístico.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.instance_loader import load_instance
from src.data.solution_loader import load_solution
from src.utils.solution import Route, Solution


DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


@pytest.fixture(scope="module")
def instance():
    return load_instance(DATA_DIR / "p01.txt")


@pytest.fixture(scope="module")
def bks_solution():
    return load_solution(DATA_DIR / "p01.res")


# =============================================================================
#  Construcción incremental
# =============================================================================

class TestIncrementalConstruction:
    """Verifica que add_route() permite construir soluciones paso a paso."""

    def test_add_single_route(self):
        s = Solution()
        s.add_route(Route(day=1, vehicle_id=1, nodes=[0, 5, 0]))
        assert len(s.routes) == 1

    def test_add_multiple_routes(self):
        s = Solution()
        s.add_route(Route(day=1, vehicle_id=1, nodes=[0, 1, 0]))
        s.add_route(Route(day=1, vehicle_id=2, nodes=[0, 2, 0]))
        s.add_route(Route(day=2, vehicle_id=1, nodes=[0, 3, 0]))
        assert len(s.routes) == 3
        assert len(s.routes_by_day(1)) == 2
        assert len(s.routes_by_day(2)) == 1


# =============================================================================
#  customer_visit_days
# =============================================================================

class TestCustomerVisitDays:
    """Verifica la consulta de días en que se visita un cliente."""

    def test_customer_not_visited(self):
        s = Solution()
        assert s.customer_visit_days(42) == ()

    def test_customer_visited_once(self):
        s = Solution(routes=[Route(day=2, vehicle_id=1, nodes=[0, 7, 0])])
        assert s.customer_visit_days(7) == (2,)

    def test_customer_visited_twice_sorted(self):
        s = Solution(routes=[
            Route(day=3, vehicle_id=1, nodes=[0, 7, 0]),
            Route(day=1, vehicle_id=1, nodes=[0, 7, 0]),
        ])
        # El resultado debe estar ordenado
        assert s.customer_visit_days(7) == (1, 3)

    def test_on_bks_p01(self, bks_solution, instance):
        """En p01 cada cliente tiene frecuencia 1, así que la BKS visita a
        cada uno exactamente un día."""
        for c in instance.customers:
            days = bks_solution.customer_visit_days(c.id)
            assert len(days) == 1, (
                f"Cliente {c.id}: esperaba 1 visita, hay {len(days)}: {days}"
            )


# =============================================================================
#  get_assigned_pattern
# =============================================================================

class TestAssignedPattern:
    """Verifica la identificación del patrón asignado a cada cliente."""

    def test_valid_pattern_on_bks(self, bks_solution, instance):
        """Toda visita en la BKS debe corresponder a un patrón válido del cliente."""
        for c in instance.customers:
            pattern = bks_solution.get_assigned_pattern(c.id, instance)
            assert pattern is not None, (
                f"Cliente {c.id} no tiene un patrón válido en la BKS"
            )
            assert pattern in c.allowed_patterns

    def test_invalid_pattern_returns_none(self, instance):
        """Si un cliente se visita en días que no forman un patrón válido,
        el método debe devolver None."""
        # Buscamos un cliente cuya frecuencia sea 1 (típico en p01).
        c = next(c for c in instance.customers if c.frequency == 1)

        # Construimos una solución que visita ese cliente 2 veces (incorrecto)
        s = Solution(routes=[
            Route(day=1, vehicle_id=1, nodes=[0, c.id, 0]),
            Route(day=2, vehicle_id=1, nodes=[0, c.id, 0]),
        ])
        # El conjunto de días visitados (1, 2) no debería estar en patrones
        # válidos para un cliente de frecuencia 1.
        result = s.get_assigned_pattern(c.id, instance)
        assert result is None


# =============================================================================
#  summary
# =============================================================================

class TestSummary:
    """Verifica el resumen estadístico de la solución."""

    def test_summary_on_bks(self, bks_solution, instance):
        summary = bks_solution.summary(instance, bks_cost=524.61)

        # Campos obligatorios
        assert "total_cost" in summary
        assert "num_routes" in summary
        assert "avg_routes_per_day" in summary
        assert "avg_load" in summary
        assert "max_load" in summary
        assert "capacity_utilization" in summary
        assert "is_feasible" in summary
        assert "num_violations" in summary
        assert "gap_to_bks" in summary

        # Valores esperados de p01
        assert summary["total_cost"] == pytest.approx(524.61, abs=0.5)
        assert summary["num_routes"] == 5.0
        assert summary["avg_routes_per_day"] == 2.5
        assert summary["is_feasible"] == 1.0
        assert summary["num_violations"] == 0.0

        # El gap respecto a sí misma es prácticamente cero
        assert abs(summary["gap_to_bks"]) < 0.5

    def test_summary_without_bks(self, bks_solution, instance):
        """Sin proveer bks_cost, no debe existir la clave gap_to_bks."""
        summary = bks_solution.summary(instance, bks_cost=None)
        assert "gap_to_bks" not in summary

    def test_capacity_utilization_within_unit(self, bks_solution, instance):
        """capacity_utilization debe estar en [0, 1] para una solución factible."""
        summary = bks_solution.summary(instance)
        assert 0.0 <= summary["capacity_utilization"] <= 1.0

    def test_gap_is_zero_for_optimal(self, instance):
        """Cargar la BKS y compararla contra sí misma da gap ≈ 0."""
        sol = load_solution(DATA_DIR / "p01.res")
        cost = sol.total_cost(instance)
        summary = sol.summary(instance, bks_cost=cost)
        assert abs(summary["gap_to_bks"]) < 1e-6


# =============================================================================
#  Route.load
# =============================================================================

class TestRouteLoad:
    """Verifica el cálculo de carga total de una ruta."""

    def test_load_matches_reported_for_bks(self, bks_solution, instance):
        """
        Las cargas calculadas deben coincidir con las reportadas en p01.res:
            día 1 veh 1 -> 149
            día 1 veh 2 -> 157
            día 2 veh 1 -> 152
            día 2 veh 2 -> 159
            día 2 veh 3 -> 160
        """
        expected_loads = {
            (1, 1): 149,
            (1, 2): 157,
            (2, 1): 152,
            (2, 2): 159,
            (2, 3): 160,
        }
        for r in bks_solution.routes:
            expected = expected_loads[(r.day, r.vehicle_id)]
            assert r.load(instance) == pytest.approx(expected)
