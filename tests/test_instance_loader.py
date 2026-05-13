"""
Tests unitarios para el cargador de instancias PVRP y el módulo de distancias.

Estos tests validan que el parser interpreta correctamente el formato del NEO
Research Group sobre tres instancias de tamaño creciente (p01, p23, pr06).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.instance import Customer, Depot, Instance
from src.data.instance_loader import _decode_pattern, load_instance
from src.utils.distance import (
    build_distance_matrix,
    build_id_to_index_map,
    euclidean_distance,
)


DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


# =============================================================================
#  Tests de decodificación de patrones
# =============================================================================

class TestPatternDecoding:
    """Verifica la conversión de patrones enteros a días de visita."""

    def test_horizon_2_pattern_1(self):
        # 1 = 01 en 2 bits -> día 2
        assert _decode_pattern(1, 2) == (2,)

    def test_horizon_2_pattern_2(self):
        # 2 = 10 en 2 bits -> día 1
        assert _decode_pattern(2, 2) == (1,)

    def test_horizon_4_single_day_patterns(self):
        # En horizonte 4, los patrones de frecuencia 1 son 1, 2, 4, 8
        assert _decode_pattern(1, 4) == (4,)
        assert _decode_pattern(2, 4) == (3,)
        assert _decode_pattern(4, 4) == (2,)
        assert _decode_pattern(8, 4) == (1,)

    def test_horizon_4_double_day_patterns(self):
        # 5 = 0101 -> días 2 y 4
        assert _decode_pattern(5, 4) == (2, 4)
        # 10 = 1010 -> días 1 y 3
        assert _decode_pattern(10, 4) == (1, 3)


# =============================================================================
#  Tests de carga de instancias
# =============================================================================

class TestInstanceLoaderP01:
    """Tests sobre la instancia p01 (51 clientes, 2 días)."""

    @pytest.fixture(scope="class")
    def instance(self) -> Instance:
        return load_instance(DATA_DIR / "p01.txt")

    def test_metadata(self, instance):
        assert instance.name == "p01"
        assert instance.horizon == 2
        assert instance.num_vehicles == 3
        assert instance.capacity == 160

    def test_depot_coordinates(self, instance):
        assert instance.depot.x == 30
        assert instance.depot.y == 40

    def test_customer_count(self, instance):
        # p01 declara 51 clientes; el "cliente 51" es duplicado del depósito
        # y se descarta. Esperamos 50 clientes reales.
        assert instance.num_customers == 50

    def test_first_customer(self, instance):
        c1 = instance.get_customer(1)
        assert c1.x == 37
        assert c1.y == 52
        assert c1.demand == 7
        assert c1.frequency == 1
        # En p01: patrones [1, 2] con horizonte 2 -> días (2,) y (1,)
        assert set(c1.allowed_patterns) == {(1,), (2,)}


class TestInstanceLoaderP23:
    """Tests sobre la instancia p23 (168 clientes, 4 días, frecuencias variadas)."""

    @pytest.fixture(scope="class")
    def instance(self) -> Instance:
        return load_instance(DATA_DIR / "p23.txt")

    def test_metadata(self, instance):
        assert instance.horizon == 4
        assert instance.num_vehicles == 6
        assert instance.capacity == 40

    def test_customer_with_frequency_2(self, instance):
        # Cliente 25 tiene frecuencia 2 con patrones [5, 10]
        c = instance.get_customer(25)
        assert c.frequency == 2
        # 5 -> (2, 4); 10 -> (1, 3)
        assert set(c.allowed_patterns) == {(2, 4), (1, 3)}

    def test_customer_with_frequency_1_full_flexibility(self, instance):
        # Cliente 97 tiene frecuencia 1 con patrones [1, 2, 4, 8]
        c = instance.get_customer(97)
        assert c.frequency == 1
        assert set(c.allowed_patterns) == {(1,), (2,), (3,), (4,)}

    def test_all_patterns_match_frequency(self, instance):
        # Invariante: todos los patrones de un cliente deben tener exactamente
        # tantos días como su frecuencia.
        for c in instance.customers:
            for pattern in c.allowed_patterns:
                assert len(pattern) == c.frequency


class TestInstanceLoaderPR06:
    """Tests sobre la instancia pr06 (288 clientes, 4 días, instancia grande)."""

    @pytest.fixture(scope="class")
    def instance(self) -> Instance:
        return load_instance(DATA_DIR / "pr06.txt")

    def test_metadata(self, instance):
        assert instance.horizon == 4
        assert instance.num_vehicles == 12
        assert instance.capacity == 175
        assert instance.max_duration == 400

    def test_customer_count(self, instance):
        assert instance.num_customers == 288

    def test_no_negative_demands(self, instance):
        for c in instance.customers:
            assert c.demand >= 0

    def test_demands_within_capacity(self, instance):
        # Toda demanda individual debe ser menor o igual a la capacidad,
        # sino la instancia sería infactible.
        for c in instance.customers:
            assert c.demand <= instance.capacity


# =============================================================================
#  Tests de distancia
# =============================================================================

class TestDistance:
    """Verifica el cálculo de distancias euclidianas."""

    def test_euclidean_basic(self):
        assert euclidean_distance((0, 0), (3, 4)) == pytest.approx(5.0)

    def test_euclidean_zero(self):
        assert euclidean_distance((1, 2), (1, 2)) == 0.0

    def test_euclidean_symmetric(self):
        d1 = euclidean_distance((1, 2), (4, 6))
        d2 = euclidean_distance((4, 6), (1, 2))
        assert d1 == d2

    def test_distance_matrix_p01(self):
        instance = load_instance(DATA_DIR / "p01.txt")
        matrix = build_distance_matrix(instance)

        # Dimensión correcta
        assert matrix.shape == (instance.num_nodes, instance.num_nodes)

        # Diagonal cero
        assert np.allclose(np.diag(matrix), 0)

        # Simetría
        assert np.allclose(matrix, matrix.T)

        # No negativa
        assert (matrix >= 0).all()

    def test_id_to_index_map(self):
        instance = load_instance(DATA_DIR / "p01.txt")
        mapping = build_id_to_index_map(instance)

        # Depósito siempre en 0
        assert mapping[0] == 0
        # Cliente 1 debería estar en índice 1 (primer cliente cargado)
        assert mapping[1] == 1
        # Cada cliente tiene un índice único
        assert len(set(mapping.values())) == len(mapping)


# =============================================================================
#  Test de integración
# =============================================================================

class TestIntegration:
    """Tests que verifican la coherencia entre los módulos."""

    def test_load_three_instances(self):
        """Verifica que las tres instancias de referencia cargan sin errores."""
        for name in ["p01", "p23", "pr06"]:
            instance = load_instance(DATA_DIR / f"{name}.txt")
            assert instance.num_customers > 0
            assert instance.horizon > 0
            assert instance.capacity > 0

    def test_distance_matrix_consistent_with_id_map(self):
        """El mapeo ID->índice debe ser coherente con la matriz de distancias."""
        instance = load_instance(DATA_DIR / "p01.txt")
        matrix = build_distance_matrix(instance)
        mapping = build_id_to_index_map(instance)

        # La distancia del depósito al cliente 1 debe coincidir con la
        # distancia euclidiana calculada directamente.
        c1 = instance.get_customer(1)
        expected = euclidean_distance(instance.depot.coords, c1.coords)

        idx_depot = mapping[0]
        idx_c1 = mapping[1]
        assert matrix[idx_depot, idx_c1] == pytest.approx(expected)
