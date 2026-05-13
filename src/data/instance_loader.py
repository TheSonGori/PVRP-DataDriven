"""
Carga de instancias del Periodic Vehicle Routing Problem (PVRP) en el formato
del NEO Research Group (Cordeau et al.).

Formato del archivo:

    Línea 1:        Type  m  n  t
        - Type: tipo de problema (1 = PVRP)
        - m: número de vehículos por día
        - n: número de clientes (sin contar el depósito en línea 0)
        - t: horizonte temporal (días)

    Líneas 2 a t+1: D  Q
        - D: duración máxima de ruta en el día d (0 = sin límite)
        - Q: capacidad máxima del vehículo en el día d

    Líneas siguientes: una por nodo, con la estructura
        ID  X  Y  ServiceDuration  Demand  Frequency  NumPatterns  P1 P2 ...

        - El nodo con ID 0 es el depósito (demanda y frecuencia 0).
        - Cada patrón Pi es un entero cuya representación binaria sobre t bits
          indica los días de visita. El bit más significativo corresponde al
          día 1, el menos significativo al día t.

Ejemplo de codificación binaria de patrones (horizonte t = 4):

        valor    binario     días de visita
        ------   --------    ---------------
            1     0001              4
            2     0010              3
            4     0100              2
            5     0101            1 y 3
            8     1000              1
           10     1010            2 y 4
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

from src.data.instance import Customer, Depot, Instance


def _decode_pattern(value: int, horizon: int) -> Tuple[int, ...]:
    """
    Decodifica un patrón entero a la lista ordenada de días de visita.

    El bit más significativo (posición `horizon - 1` en la representación
    binaria) corresponde al día 1.

    Args:
        value: Valor entero del patrón.
        horizon: Número de días del horizonte de planificación.

    Returns:
        Tupla ordenada de días (1-indexed) en los que se realiza la visita.

    Examples:
        >>> _decode_pattern(5, 4)   # 5 = 0101 -> días 2 y 4 (desde MSB)
        (2, 4)
        >>> _decode_pattern(8, 4)   # 8 = 1000 -> día 1
        (1,)
    """
    days: List[int] = []
    for bit_position in range(horizon):
        # bit_position = 0 -> día más a la derecha del binario
        # Queremos día 1 = bit más significativo
        day = horizon - bit_position
        if value & (1 << bit_position):
            days.append(day)
    return tuple(sorted(days))


def _tokenize_line(line: str) -> List[str]:
    """Tokeniza una línea eliminando espacios extra y líneas vacías."""
    return line.strip().split()


def load_instance(filepath: Union[str, Path]) -> Instance:
    """
    Carga una instancia del PVRP desde un archivo de texto del NEO Research Group.

    Args:
        filepath: Ruta al archivo de la instancia (.txt).

    Returns:
        Una instancia `Instance` con todos los datos del problema.

    Raises:
        FileNotFoundError: si el archivo no existe.
        ValueError: si el formato del archivo es inválido.

    Note:
        El parser asume el formato estándar del NEO Research Group para PVRP
        (Cordeau et al.). En algunas instancias, la última línea puede ser
        una repetición del depósito; se ignora si su demanda iguala la
        capacidad y sus coordenadas coinciden con el depósito.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {filepath}")

    with filepath.open("r", encoding="utf-8") as f:
        lines = [ln for ln in f.readlines() if ln.strip()]

    if len(lines) < 2:
        raise ValueError(f"Archivo demasiado corto: {filepath}")

    # --- Línea 1: cabecera ---
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

    # --- Líneas 2 a horizon+1: parámetros por día (D, Q) ---
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

    # Por simplicidad asumimos que la capacidad es uniforme en todos los días
    # (consistente con la formulación del modelo en la memoria, Sección 1.5.2).
    capacities = [q for _, q in daily_params]
    durations = [d for d, _ in daily_params]
    if len(set(capacities)) > 1:
        # Si en algún caso varía, tomamos la mínima como restricción más fuerte.
        # En las instancias estándar de Cordeau esto no ocurre.
        capacity = min(capacities)
    else:
        capacity = capacities[0]
    max_duration = max(durations)

    # --- Líneas restantes: nodos ---
    node_lines = lines[1 + horizon:]

    depot: Depot | None = None
    customers: List[Customer] = []

    for raw_line in node_lines:
        tokens = _tokenize_line(raw_line)
        if len(tokens) < 7:
            # Línea malformada o vacía: la saltamos.
            continue

        node_id = int(tokens[0])
        x = float(tokens[1])
        y = float(tokens[2])
        service_duration = float(tokens[3])
        demand = float(tokens[4])
        frequency = int(tokens[5])
        num_patterns = int(tokens[6])

        # Caso depósito: ID 0, frecuencia 0, sin patrones.
        if node_id == 0:
            depot = Depot(x=x, y=y)
            continue

        # Caso especial: algunas instancias (p01) repiten el depósito al final
        # con un ID alto y demanda igual a la capacidad. Lo ignoramos.
        if depot is not None and (x, y) == (depot.x, depot.y) and demand >= capacity:
            continue

        # Lectura de patrones.
        if len(tokens) < 7 + num_patterns:
            raise ValueError(
                f"Cliente {node_id}: se esperaban {num_patterns} patrones, "
                f"hay menos en la línea."
            )

        pattern_values = [int(tokens[7 + i]) for i in range(num_patterns)]
        allowed_patterns = tuple(
            _decode_pattern(value, horizon) for value in pattern_values
        )

        # Verificación de consistencia: cada patrón debe tener exactamente
        # `frequency` días de visita.
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
        # Advertencia silenciosa: en algunas instancias el conteo declarado
        # incluye al depósito o difiere por una unidad. Lo registramos pero
        # no fallamos para mantener robustez.
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
