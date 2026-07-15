"""
Tests del cargador de instancias PVRP y del módulo de distancias: validan
que el parser interpreta correctamente el formato del NEO Research Group
sobre tres instancias de tamaño creciente (p01, p23, pr06).

Entrada: instancias del dataset (data/raw/p01.txt, p23.txt, pr06.txt).
Salida: aserciones pytest; no retorna valores.
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


class TestPatternDecoding:

    # 1 = 01 en 2 bits -> día 2.
    def test_horizon_2_pattern_1(self):
        assert _decode_pattern(1, 2) == (2,)

    # 2 = 10 en 2 bits -> día 1.
    def test_horizon_2_pattern_2(self):
        assert _decode_pattern(2, 2) == (1,)

    # En horizonte 4, los patrones de frecuencia 1 son 1, 2, 4, 8.
    def test_horizon_4_single_day_patterns(self):
        assert _decode_pattern(1, 4) == (4,)
        assert _decode_pattern(2, 4) == (3,)
        assert _decode_pattern(4, 4) == (2,)
        assert _decode_pattern(8, 4) == (1,)

    # 5 = 0101 -> días 2 y 4; 10 = 1010 -> días 1 y 3.
    def test_horizon_4_double_day_patterns(self):
        assert _decode_pattern(5, 4) == (2, 4)
        assert _decode_pattern(10, 4) == (1, 3)


class TestInstanceLoaderP01:

    @pytest.fixture(scope="class")
    def instance(self) -> Instance:
        return load_instance(DATA_DIR / "p01.txt")

    # Metadatos de p01: horizonte, vehículos y capacidad.
    def test_metadata(self, instance):
        assert instance.name == "p01"
        assert instance.horizon == 2
        assert instance.num_vehicles == 3
        assert instance.capacity == 160

    # Coordenadas del depósito.
    def test_depot_coordinates(self, instance):
        assert instance.depot.x == 30
        assert instance.depot.y == 40

    # p01 declara 51 clientes; el "cliente 51" es duplicado del depósito y se descarta.
    def test_customer_count(self, instance):
        assert instance.num_customers == 50

    # Datos del primer cliente, incluyendo sus patrones permitidos.
    def test_first_customer(self, instance):
        c1 = instance.get_customer(1)
        assert c1.x == 37
        assert c1.y == 52
        assert c1.demand == 7
        assert c1.frequency == 1
        assert set(c1.allowed_patterns) == {(1,), (2,)}


class TestInstanceLoaderP23:

    @pytest.fixture(scope="class")
    def instance(self) -> Instance:
        return load_instance(DATA_DIR / "p23.txt")

    # Metadatos de p23.
    def test_metadata(self, instance):
        assert instance.horizon == 4
        assert instance.num_vehicles == 6
        assert instance.capacity == 40

    # Cliente 25 tiene frecuencia 2 con patrones 5 -> (2,4) y 10 -> (1,3).
    def test_customer_with_frequency_2(self, instance):
        c = instance.get_customer(25)
        assert c.frequency == 2
        assert set(c.allowed_patterns) == {(2, 4), (1, 3)}

    # Cliente 97 tiene frecuencia 1 con los 4 patrones posibles.
    def test_customer_with_frequency_1_full_flexibility(self, instance):
        c = instance.get_customer(97)
        assert c.frequency == 1
        assert set(c.allowed_patterns) == {(1,), (2,), (3,), (4,)}

    # Todos los patrones de un cliente tienen exactamente tantos días como su frecuencia.
    def test_all_patterns_match_frequency(self, instance):
        for c in instance.customers:
            for pattern in c.allowed_patterns:
                assert len(pattern) == c.frequency


class TestInstanceLoaderPR06:

    @pytest.fixture(scope="class")
    def instance(self) -> Instance:
        return load_instance(DATA_DIR / "pr06.txt")

    # Metadatos de pr06 (instancia grande).
    def test_metadata(self, instance):
        assert instance.horizon == 4
        assert instance.num_vehicles == 12
        assert instance.capacity == 175
        assert instance.max_duration == 400

    # Número total de clientes.
    def test_customer_count(self, instance):
        assert instance.num_customers == 288

    # Ninguna demanda es negativa.
    def test_no_negative_demands(self, instance):
        for c in instance.customers:
            assert c.demand >= 0

    # Toda demanda individual cabe dentro de la capacidad del vehículo.
    def test_demands_within_capacity(self, instance):
        for c in instance.customers:
            assert c.demand <= instance.capacity


class TestDistance:

    # Distancia euclidiana entre (0,0) y (3,4) es 5 (triángulo 3-4-5).
    def test_euclidean_basic(self):
        assert euclidean_distance((0, 0), (3, 4)) == pytest.approx(5.0)

    # La distancia de un punto a sí mismo es cero.
    def test_euclidean_zero(self):
        assert euclidean_distance((1, 2), (1, 2)) == 0.0

    # La distancia euclidiana es simétrica.
    def test_euclidean_symmetric(self):
        d1 = euclidean_distance((1, 2), (4, 6))
        d2 = euclidean_distance((4, 6), (1, 2))
        assert d1 == d2

    # La matriz de distancias de p01 tiene la dimensión correcta, diagonal cero, es simétrica y no negativa.
    def test_distance_matrix_p01(self):
        instance = load_instance(DATA_DIR / "p01.txt")
        matrix = build_distance_matrix(instance)

        assert matrix.shape == (instance.num_nodes, instance.num_nodes)
        assert np.allclose(np.diag(matrix), 0)
        assert np.allclose(matrix, matrix.T)
        assert (matrix >= 0).all()

    # El mapeo id->índice ubica el depósito en 0 y da índices únicos a cada cliente.
    def test_id_to_index_map(self):
        instance = load_instance(DATA_DIR / "p01.txt")
        mapping = build_id_to_index_map(instance)

        assert mapping[0] == 0
        assert mapping[1] == 1
        assert len(set(mapping.values())) == len(mapping)


class TestIntegration:

    # Las tres instancias de referencia (p01, p23, pr06) cargan sin errores.
    def test_load_three_instances(self):
        for name in ["p01", "p23", "pr06"]:
            instance = load_instance(DATA_DIR / f"{name}.txt")
            assert instance.num_customers > 0
            assert instance.horizon > 0
            assert instance.capacity > 0

    # La distancia depósito-cliente 1 en la matriz coincide con la euclidiana directa.
    def test_distance_matrix_consistent_with_id_map(self):
        instance = load_instance(DATA_DIR / "p01.txt")
        matrix = build_distance_matrix(instance)
        mapping = build_id_to_index_map(instance)

        c1 = instance.get_customer(1)
        expected = euclidean_distance(instance.depot.coords, c1.coords)

        idx_depot = mapping[0]
        idx_c1 = mapping[1]
        assert matrix[idx_depot, idx_c1] == pytest.approx(expected)
