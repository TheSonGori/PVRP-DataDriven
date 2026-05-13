"""
Cálculo de distancias y construcción de la matriz de costos para una instancia
del PVRP.

La distancia entre dos nodos se asume euclidiana, consistente con la convención
de las instancias del NEO Research Group y con el modelo matemático presentado
en la Sección 1.5.2 de la memoria (costo c_ij del arco (i, j)).
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from src.data.instance import Instance


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Calcula la distancia euclidiana entre dos puntos del plano.

    Args:
        p1: Coordenadas (x, y) del primer punto.
        p2: Coordenadas (x, y) del segundo punto.

    Returns:
        Distancia euclidiana no negativa.
    """
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx * dx + dy * dy)


def build_distance_matrix(instance: Instance) -> np.ndarray:
    """
    Construye la matriz de distancias euclidianas entre todos los nodos de una
    instancia.

    La matriz resultante tiene dimensión (n+1) x (n+1), donde n es el número
    de clientes. El índice 0 corresponde al depósito y los índices 1 a n
    corresponden a los clientes en el orden en que aparecen en
    `instance.customers`.

    IMPORTANTE: el índice en la matriz NO necesariamente coincide con el `id`
    del cliente. Para mapear entre índice de matriz e ID de cliente, usar
    `build_id_to_index_map`.

    Args:
        instance: Instancia del PVRP.

    Returns:
        Matriz cuadrada simétrica de distancias con ceros en la diagonal.
    """
    n_nodes = instance.num_nodes
    matrix = np.zeros((n_nodes, n_nodes), dtype=np.float64)

    coords = [instance.depot.coords] + [c.coords for c in instance.customers]

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = euclidean_distance(coords[i], coords[j])
            matrix[i, j] = d
            matrix[j, i] = d

    return matrix


def build_id_to_index_map(instance: Instance) -> dict:
    """
    Construye un mapeo entre el ID del cliente y su índice en la matriz de
    distancias.

    El depósito tiene siempre índice 0. Los clientes se indexan en el orden
    en que aparecen en `instance.customers`.

    Args:
        instance: Instancia del PVRP.

    Returns:
        Diccionario {customer_id: matrix_index}. El depósito se mapea con
        la clave 0 al índice 0.
    """
    mapping = {0: 0}
    for idx, customer in enumerate(instance.customers, start=1):
        mapping[customer.id] = idx
    return mapping
