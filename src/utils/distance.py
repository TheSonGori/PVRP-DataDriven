"""
Cálculo de distancias euclidianas y construcción de la matriz de costos entre
nodos (depósito + clientes) de una instancia del PVRP.

Entrada: una Instance (src/data/instance.py) o pares de coordenadas (x, y).
Salida: distancias escalares, una matriz numpy (n_nodes x n_nodes) de
distancias, o un diccionario que mapea customer_id -> índice de matriz.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from src.data.instance import Instance


# Distancia euclidiana entre dos puntos (x, y).
def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx * dx + dy * dy)


# Construye la matriz (n_nodes x n_nodes) de distancias entre depósito y clientes.
def build_distance_matrix(instance: Instance) -> np.ndarray:
    n_nodes = instance.num_nodes
    matrix = np.zeros((n_nodes, n_nodes), dtype=np.float64)

    coords = [instance.depot.coords] + [c.coords for c in instance.customers]

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = euclidean_distance(coords[i], coords[j])
            matrix[i, j] = d
            matrix[j, i] = d

    return matrix


# Construye el mapeo {customer_id: índice en la matriz de distancias} (depósito = 0).
def build_id_to_index_map(instance: Instance) -> dict:
    mapping = {0: 0}
    for idx, customer in enumerate(instance.customers, start=1):
        mapping[customer.id] = idx
    return mapping
