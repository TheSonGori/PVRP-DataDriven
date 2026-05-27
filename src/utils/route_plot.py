"""
Visualización de rutas del PVRP (Día 15, Fase E).

Dibuja las rutas de una solución sobre el plano de coordenadas: el depósito
destacado, los clientes como puntos, y cada vehículo/ruta en un color
distinto. Genera una figura por día del horizonte de planificación.

Es agnóstico al origen de la solución: funciona igual con la solución del
agente RL, del VNS, del Greedy o con la BKS, lo que permite comparaciones
visuales homogéneas. Alimenta el Índice de Figuras de la memoria.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # backend sin ventana: solo guarda archivos
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.data.instance import Instance
from src.utils.solution import Solution


# Paleta de colores legible y distinguible para las rutas.
_PALETTE = [
    "#2E75B6", "#C0392B", "#27AE60", "#E67E22", "#8E44AD",
    "#16A085", "#D4AC0D", "#7F8C8D", "#2C3E50", "#E84393",
]


def _node_xy(instance: Instance, node_id: int) -> tuple[float, float]:
    """Devuelve (x, y) de un nodo. El nodo 0 es el depósito."""
    if node_id == 0:
        return instance.depot.x, instance.depot.y
    c = instance.get_customer(node_id)
    return c.x, c.y


def plot_solution_day(
    instance: Instance,
    solution: Solution,
    day: int,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Dibuja las rutas de un día concreto sobre un eje matplotlib.

    Args:
        instance: Instancia del PVRP (para coordenadas).
        solution: Solución a graficar.
        day: Día del horizonte a dibujar (1-indexado).
        ax: Eje donde dibujar. Si es None, crea una figura nueva.
        title: Título del subplot.

    Returns:
        El eje con el gráfico.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    routes = solution.routes_by_day(day)

    # 1. Dibujar TODOS los clientes como fondo gris claro (contexto).
    xs = [c.x for c in instance.customers]
    ys = [c.y for c in instance.customers]
    ax.scatter(xs, ys, c="#D5D8DC", s=35, zorder=1, edgecolors="none")

    # 2. Dibujar cada ruta del día con su color.
    for i, route in enumerate(routes):
        color = _PALETTE[i % len(_PALETTE)]
        path = [_node_xy(instance, n) for n in route.nodes]
        rx = [p[0] for p in path]
        ry = [p[1] for p in path]
        # Líneas de la ruta (depósito → clientes → depósito).
        ax.plot(rx, ry, "-", color=color, linewidth=1.6, alpha=0.8, zorder=2)
        # Marcar los clientes visitados en esta ruta.
        ax.scatter(rx[1:-1], ry[1:-1], c=color, s=55, zorder=3,
                   edgecolors="white", linewidths=0.8)

    # 3. Depósito destacado encima de todo.
    dx, dy = instance.depot.x, instance.depot.y
    ax.scatter([dx], [dy], marker="s", c="#1A1A1A", s=180, zorder=5,
               edgecolors="white", linewidths=1.5, label="Depósito")
    ax.annotate("Depósito", (dx, dy), textcoords="offset points",
                xytext=(0, 12), ha="center", fontsize=9, fontweight="bold")

    ax.set_title(title or f"Día {day}  ({len(routes)} rutas)", fontsize=12)
    ax.set_xlabel("Coordenada X")
    ax.set_ylabel("Coordenada Y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", alpha=0.4)
    return ax


def plot_solution(
    instance: Instance,
    solution: Solution,
    save_path: Optional[Path] = None,
    suptitle: Optional[str] = None,
) -> plt.Figure:
    """
    Dibuja la solución completa: un subplot por cada día del horizonte.

    Args:
        instance: Instancia del PVRP.
        solution: Solución a graficar.
        save_path: Si se indica, guarda la figura (PNG) en esa ruta.
        suptitle: Título general de la figura.

    Returns:
        La figura matplotlib.
    """
    horizon = instance.horizon
    # Una fila de subplots, uno por día.
    fig, axes = plt.subplots(1, horizon, figsize=(7 * horizon, 6), squeeze=False)
    axes = axes[0]  # squeeze=False -> matriz 1xN, tomamos la fila

    for day in range(1, horizon + 1):
        plot_solution_day(instance, solution, day, ax=axes[day - 1])

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold")

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig