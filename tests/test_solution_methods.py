"""
Tests de los métodos de Solution: construcción incremental, consulta de
días de visita y patrón asignado, resumen estadístico, y cálculo de carga
por ruta (Route.load).

Entrada: la instancia p01 y su BKS (data/raw/p01.txt, p01.res), además de
Solution/Route construidas manualmente.
Salida: aserciones pytest; no retorna valores.
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


class TestIncrementalConstruction:

    # add_route agrega una única ruta a la solución.
    def test_add_single_route(self):
        s = Solution()
        s.add_route(Route(day=1, vehicle_id=1, nodes=[0, 5, 0]))
        assert len(s.routes) == 1

    # add_route permite construir la solución agregando varias rutas en distintos días.
    def test_add_multiple_routes(self):
        s = Solution()
        s.add_route(Route(day=1, vehicle_id=1, nodes=[0, 1, 0]))
        s.add_route(Route(day=1, vehicle_id=2, nodes=[0, 2, 0]))
        s.add_route(Route(day=2, vehicle_id=1, nodes=[0, 3, 0]))
        assert len(s.routes) == 3
        assert len(s.routes_by_day(1)) == 2
        assert len(s.routes_by_day(2)) == 1


class TestCustomerVisitDays:

    # Un cliente no visitado devuelve una tupla vacía de días.
    def test_customer_not_visited(self):
        s = Solution()
        assert s.customer_visit_days(42) == ()

    # Un cliente visitado una vez devuelve una tupla con ese único día.
    def test_customer_visited_once(self):
        s = Solution(routes=[Route(day=2, vehicle_id=1, nodes=[0, 7, 0])])
        assert s.customer_visit_days(7) == (2,)

    # Los días de visita se devuelven ordenados.
    def test_customer_visited_twice_sorted(self):
        s = Solution(routes=[
            Route(day=3, vehicle_id=1, nodes=[0, 7, 0]),
            Route(day=1, vehicle_id=1, nodes=[0, 7, 0]),
        ])
        assert s.customer_visit_days(7) == (1, 3)

    # En p01 (todos frecuencia 1), la BKS visita a cada cliente exactamente un día.
    def test_on_bks_p01(self, bks_solution, instance):
        for c in instance.customers:
            days = bks_solution.customer_visit_days(c.id)
            assert len(days) == 1, (
                f"Cliente {c.id}: esperaba 1 visita, hay {len(days)}: {days}"
            )


class TestAssignedPattern:

    # Toda visita en la BKS corresponde a un patrón permitido del cliente.
    def test_valid_pattern_on_bks(self, bks_solution, instance):
        for c in instance.customers:
            pattern = bks_solution.get_assigned_pattern(c.id, instance)
            assert pattern is not None, (
                f"Cliente {c.id} no tiene un patrón válido en la BKS"
            )
            assert pattern in c.allowed_patterns

    # Si los días visitados no forman un patrón válido, se devuelve None.
    def test_invalid_pattern_returns_none(self, instance):
        c = next(c for c in instance.customers if c.frequency == 1)

        s = Solution(routes=[
            Route(day=1, vehicle_id=1, nodes=[0, c.id, 0]),
            Route(day=2, vehicle_id=1, nodes=[0, c.id, 0]),
        ])
        result = s.get_assigned_pattern(c.id, instance)
        assert result is None


class TestSummary:

    # summary() incluye todos los campos esperados con los valores correctos para la BKS de p01.
    def test_summary_on_bks(self, bks_solution, instance):
        summary = bks_solution.summary(instance, bks_cost=524.61)

        assert "total_cost" in summary
        assert "num_routes" in summary
        assert "avg_routes_per_day" in summary
        assert "avg_load" in summary
        assert "max_load" in summary
        assert "capacity_utilization" in summary
        assert "is_feasible" in summary
        assert "num_violations" in summary
        assert "gap_to_bks" in summary

        assert summary["total_cost"] == pytest.approx(524.61, abs=0.5)
        assert summary["num_routes"] == 5.0
        assert summary["avg_routes_per_day"] == 2.5
        assert summary["is_feasible"] == 1.0
        assert summary["num_violations"] == 0.0

        assert abs(summary["gap_to_bks"]) < 0.5

    # Sin bks_cost, la clave gap_to_bks no aparece en el resumen.
    def test_summary_without_bks(self, bks_solution, instance):
        summary = bks_solution.summary(instance, bks_cost=None)
        assert "gap_to_bks" not in summary

    # capacity_utilization está en [0, 1] para una solución factible.
    def test_capacity_utilization_within_unit(self, bks_solution, instance):
        summary = bks_solution.summary(instance)
        assert 0.0 <= summary["capacity_utilization"] <= 1.0

    # Comparar la BKS contra sí misma da un gap prácticamente cero.
    def test_gap_is_zero_for_optimal(self, instance):
        sol = load_solution(DATA_DIR / "p01.res")
        cost = sol.total_cost(instance)
        summary = sol.summary(instance, bks_cost=cost)
        assert abs(summary["gap_to_bks"]) < 1e-6


class TestRouteLoad:

    # Las cargas calculadas por Route.load coinciden con las reportadas en p01.res.
    def test_load_matches_reported_for_bks(self, bks_solution, instance):
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
