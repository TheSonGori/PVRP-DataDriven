"""
Carga de soluciones del Periodic Vehicle Routing Problem (PVRP) en el formato
`.res` del NEO Research Group.

Formato del archivo `.res`:

    Línea 1:    costo_total
    Líneas n:   día  vehículo  costo_ruta  carga  nodo_0  nodo_1  ...  nodo_0

Cada línea de ruta contiene:

    - Columna 1: día del horizonte (1-indexed).
    - Columna 2: identificador del vehículo dentro del día.
    - Columna 3: costo (distancia) de la ruta.
    - Columna 4: carga total transportada (suma de demandas de clientes).
    - Columnas 5+: secuencia de nodos visitados. Comienza y termina en 0
      (el depósito).

Estas soluciones son las **Best Known Solutions (BKS)** publicadas por el
NEO Research Group y se utilizan como referencia para calcular el gap del
método propuesto en la memoria.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from src.utils.solution import Route, Solution


def load_solution(filepath: Union[str, Path]) -> Solution:
    """
    Carga una solución de referencia desde un archivo `.res` del NEO.

    Args:
        filepath: Ruta al archivo `.res`.

    Returns:
        Una instancia `Solution` con las rutas leídas y el costo total
        reportado en el archivo.

    Raises:
        FileNotFoundError: si el archivo no existe.
        ValueError: si el formato no es válido.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"No se encontró el archivo de solución: {filepath}")

    with filepath.open("r", encoding="utf-8") as f:
        # `splitlines()` elimina automáticamente el `\r` de archivos con final
        # de línea Windows (CRLF), que es el caso de los archivos del NEO.
        lines = [ln.strip() for ln in f.read().splitlines() if ln.strip()]

    if len(lines) < 1:
        raise ValueError(f"Archivo vacío: {filepath}")

    # --- Línea 1: costo total reportado ---
    try:
        reported_cost = float(lines[0].split()[0])
    except (ValueError, IndexError) as e:
        raise ValueError(
            f"No se pudo parsear el costo total en la primera línea de {filepath}: {e}"
        )

    # --- Líneas restantes: rutas ---
    routes: list[Route] = []
    for ln_idx, raw_line in enumerate(lines[1:], start=2):
        tokens = raw_line.split()
        if len(tokens) < 6:
            # Una ruta válida tiene al menos: día, vehículo, costo, carga,
            # depósito_inicial, depósito_final (= 6 tokens mínimo si solo
            # visita el depósito, lo cual no debería ocurrir, pero por
            # robustez aceptamos).
            raise ValueError(
                f"Línea {ln_idx} de {filepath} tiene formato inválido "
                f"(esperan >= 6 tokens, hay {len(tokens)})."
            )

        try:
            day = int(tokens[0])
            vehicle_id = int(tokens[1])
            # tokens[2] = costo de la ruta (lo recalcularemos al validar)
            # tokens[3] = carga total transportada (lo verificaremos al validar)
            nodes = [int(t) for t in tokens[4:]]
        except ValueError as e:
            raise ValueError(
                f"Error parseando la línea {ln_idx} de {filepath}: {e}"
            )

        # Verificación básica: la ruta debe empezar y terminar en el depósito.
        if nodes[0] != 0 or nodes[-1] != 0:
            raise ValueError(
                f"Línea {ln_idx} de {filepath}: la ruta no empieza/termina "
                f"en el depósito (0). Nodos: {nodes}"
            )

        routes.append(Route(day=day, vehicle_id=vehicle_id, nodes=nodes))

    return Solution(routes=routes, reported_cost=reported_cost)
