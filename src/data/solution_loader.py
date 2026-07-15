"""
Carga soluciones de referencia (Best Known Solutions) del PVRP desde archivos
.res del NEO Research Group: costo total en la primera línea y, luego, una
línea por ruta (día, vehículo, costo, carga, secuencia de nodos).

Entrada: ruta a un archivo .res de solución.
Salida: un objeto Solution (src/utils/solution.py) con las rutas leídas y el
costo total reportado.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from src.utils.solution import Route, Solution


# Lee un archivo .res del NEO Research Group y construye un Solution.
def load_solution(filepath: Union[str, Path]) -> Solution:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"No se encontró el archivo de solución: {filepath}")

    with filepath.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.read().splitlines() if ln.strip()]

    if len(lines) < 1:
        raise ValueError(f"Archivo vacío: {filepath}")

    try:
        reported_cost = float(lines[0].split()[0])
    except (ValueError, IndexError) as e:
        raise ValueError(
            f"No se pudo parsear el costo total en la primera línea de {filepath}: {e}"
        )

    routes: list[Route] = []
    for ln_idx, raw_line in enumerate(lines[1:], start=2):
        tokens = raw_line.split()
        if len(tokens) < 6:
            raise ValueError(
                f"Línea {ln_idx} de {filepath} tiene formato inválido "
                f"(esperan >= 6 tokens, hay {len(tokens)})."
            )

        try:
            day = int(tokens[0])
            vehicle_id = int(tokens[1])
            nodes = [int(t) for t in tokens[4:]]
        except ValueError as e:
            raise ValueError(
                f"Error parseando la línea {ln_idx} de {filepath}: {e}"
            )

        if nodes[0] != 0 or nodes[-1] != 0:
            raise ValueError(
                f"Línea {ln_idx} de {filepath}: la ruta no empieza/termina "
                f"en el depósito (0). Nodos: {nodes}"
            )

        routes.append(Route(day=day, vehicle_id=vehicle_id, nodes=nodes))

    return Solution(routes=routes, reported_cost=reported_cost)
