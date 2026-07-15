"""
Carga instancias del PVRP desde archivos de texto en el formato del NEO
Research Group (Cordeau et al.): cabecera (tipo, vehículos, clientes,
horizonte), parámetros diarios (duración, capacidad) y un nodo por línea
(depósito + clientes con sus patrones de visita codificados en binario).

Entrada: ruta a un archivo .txt de instancia.
Salida: un objeto Instance (src/data/instance.py) con depósito, clientes,
horizonte, número de vehículos, capacidad y duración máxima de ruta.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

from src.data.instance import Customer, Depot, Instance


# Convierte un patrón entero (bits = días de visita) a la tupla de días 1-indexed.
def _decode_pattern(value: int, horizon: int) -> Tuple[int, ...]:
    days: List[int] = []
    for bit_position in range(horizon):
        day = horizon - bit_position
        if value & (1 << bit_position):
            days.append(day)
    return tuple(sorted(days))


# Separa una línea en tokens, ignorando espacios extra.
def _tokenize_line(line: str) -> List[str]:
    return line.strip().split()


# Lee un archivo de instancia del NEO Research Group y construye un Instance.
def load_instance(filepath: Union[str, Path]) -> Instance:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {filepath}")

    with filepath.open("r", encoding="utf-8") as f:
        lines = [ln for ln in f.readlines() if ln.strip()]

    if len(lines) < 2:
        raise ValueError(f"Archivo demasiado corto: {filepath}")

    header = _tokenize_line(lines[0])
    if len(header) < 4:
        raise ValueError(
            f"Cabecera inválida en {filepath}: se esperan 4 valores "
            f"(Type, m, n, t), se obtuvieron {len(header)}."
        )

    _problem_type = int(header[0])
    num_vehicles = int(header[1])
    num_customers_expected = int(header[2])
    horizon = int(header[3])

    if len(lines) < 1 + horizon:
        raise ValueError(
            f"Faltan líneas de parámetros diarios en {filepath}: "
            f"se esperaban {horizon}, se encontraron {len(lines) - 1}."
        )

    daily_params: List[Tuple[float, float]] = []
    for day_idx in range(horizon):
        tokens = _tokenize_line(lines[1 + day_idx])
        if len(tokens) < 2:
            raise ValueError(
                f"Línea de parámetros del día {day_idx + 1} inválida en {filepath}."
            )
        max_duration = float(tokens[0])
        capacity = float(tokens[1])
        daily_params.append((max_duration, capacity))

    capacities = [q for _, q in daily_params]
    durations = [d for d, _ in daily_params]
    if len(set(capacities)) > 1:
        capacity = min(capacities)
    else:
        capacity = capacities[0]
    max_duration = max(durations)

    node_lines = lines[1 + horizon:]

    depot: Depot | None = None
    customers: List[Customer] = []

    for raw_line in node_lines:
        tokens = _tokenize_line(raw_line)
        if len(tokens) < 7:
            continue

        node_id = int(tokens[0])
        x = float(tokens[1])
        y = float(tokens[2])
        service_duration = float(tokens[3])
        demand = float(tokens[4])
        frequency = int(tokens[5])
        num_patterns = int(tokens[6])

        if node_id == 0:
            depot = Depot(x=x, y=y)
            continue

        if depot is not None and (x, y) == (depot.x, depot.y) and demand >= capacity:
            continue

        if len(tokens) < 7 + num_patterns:
            raise ValueError(
                f"Cliente {node_id}: se esperaban {num_patterns} patrones, "
                f"hay menos en la línea."
            )

        pattern_values = [int(tokens[7 + i]) for i in range(num_patterns)]
        allowed_patterns = tuple(
            _decode_pattern(value, horizon) for value in pattern_values
        )

        for pattern in allowed_patterns:
            if len(pattern) != frequency:
                raise ValueError(
                    f"Cliente {node_id}: patrón {pattern} tiene {len(pattern)} "
                    f"días pero la frecuencia declarada es {frequency}."
                )

        customers.append(
            Customer(
                id=node_id,
                x=x,
                y=y,
                service_duration=service_duration,
                demand=demand,
                frequency=frequency,
                allowed_patterns=allowed_patterns,
            )
        )

    if depot is None:
        raise ValueError(f"No se encontró el depósito (nodo 0) en {filepath}.")

    if len(customers) != num_customers_expected:
        pass

    return Instance(
        name=filepath.stem,
        depot=depot,
        customers=customers,
        horizon=horizon,
        num_vehicles=num_vehicles,
        capacity=capacity,
        max_duration=max_duration,
    )
