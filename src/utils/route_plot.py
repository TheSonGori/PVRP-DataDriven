"""
Genera figuras de las rutas de una solución del PVRP sobre el plano de
coordenadas (depósito destacado, clientes como puntos, cada ruta en un color
distinto), con un subplot por día del horizonte. Es agnóstico al origen de la
solución (agente RL, VNS, Greedy o BKS), lo que permite comparaciones visuales
homogéneas.

Entrada: una Instance y una Solution (src/data/instance.py,
src/utils/solution.py), y opcionalmente una ruta donde guardar la figura.
Salida: objetos matplotlib Axes/Figure y, si se indica save_path, un PNG
guardado en disco.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.data.instance import Instance
from src.utils.solution import Solution


_PALETTE = [
    "#2E75B6", "#C0392B", "#27AE60", "#E67E22", "#8E44AD",
    "#16A085", "#D4AC0D", "#7F8C8D", "#2C3E50", "#E84393",
]


# Coordenadas (x, y) de un nodo; el nodo 0 es el depósito.
def _node_xy(instance: Instance, node_id: int) -> tuple[float, float]:
    if node_id == 0:
        return instance.depot.x, instance.depot.y
    c = instance.get_customer(node_id)
    return c.x, c.y


# Dibuja las rutas de un día concreto sobre un eje matplotlib.
def plot_solution_day(
    instance: Instance,
    solution: Solution,
    day: int,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    routes = solution.routes_by_day(day)

    xs = [c.x for c in instance.customers]
    ys = [c.y for c in instance.customers]
    ax.scatter(xs, ys, c="#D5D8DC", s=35, zorder=1, edgecolors="none")

    for i, route in enumerate(routes):
        color = _PALETTE[i % len(_PALETTE)]
        path = [_node_xy(instance, n) for n in route.nodes]
        rx = [p[0] for p in path]
        ry = [p[1] for p in path]
        ax.plot(rx, ry, "-", color=color, linewidth=1.6, alpha=0.8, zorder=2)
        ax.scatter(rx[1:-1], ry[1:-1], c=color, s=55, zorder=3,
                   edgecolors="white", linewidths=0.8)

    dx, dy = instance.depot.x, instance.depot.y
    ax.scatter([dx], [dy], marker="s", c="#1A1A1A", s=180, zorder=5,
               edgecolors="white", linewidths=1.5, label="Depósito")
    ax.annotate("Depósito", (dx, dy), textcoords="offset points",
                xytext=(0, 12), ha="center", fontsize=9, fontweight="bold")

    ax.set_title(title or f"Día {day}  ({len(routes)} rutas)", fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", alpha=0.4)
    return ax


# Dibuja la solución completa: un subplot por cada día del horizonte, y guarda el PNG si se pide.
def plot_solution(
    instance: Instance,
    solution: Solution,
    save_path: Optional[Path] = None,
    suptitle: Optional[str] = None,
    max_cols: int = 3,
) -> plt.Figure:
    import math

    horizon = instance.horizon
    n_cols = min(horizon, max_cols)
    n_rows = math.ceil(horizon / max_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(7 * n_cols, 6 * n_rows),
        squeeze=False,
    )

    axes_flat = axes.flatten()

    for day in range(1, horizon + 1):
        plot_solution_day(instance, solution, day, ax=axes_flat[day - 1])

    for k in range(horizon, len(axes_flat)):
        axes_flat[k].axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=16, fontweight="bold", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
