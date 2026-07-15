"""
Funciones de visualización (matplotlib) para instancias y soluciones del PVRP:
disposición geográfica de clientes/depósito, distribución de demandas y
frecuencias, y rutas de una solución separadas por día.

Entrada: una Instance y, opcionalmente, una Solution (src/data/instance.py,
src/utils/solution.py).
Salida: objetos Figure de matplotlib listos para mostrar o guardar.
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.data.instance import Instance
from src.utils.solution import Solution


_DAY_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
]


# Color cíclico asociado a un día del horizonte.
def _day_color(day: int) -> str:
    return _DAY_PALETTE[(day - 1) % len(_DAY_PALETTE)]


# Dibuja la disposición geográfica de clientes y depósito de una instancia.
def plot_instance(
    instance: Instance,
    *,
    ax: Optional[Axes] = None,
    color_by: str = "frequency",
    show_ids: bool = False,
    title: Optional[str] = None,
) -> Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 9))
    else:
        fig = ax.figure

    xs = np.array([c.x for c in instance.customers])
    ys = np.array([c.y for c in instance.customers])

    if color_by == "frequency":
        freqs = np.array([c.frequency for c in instance.customers])
        scatter = ax.scatter(
            xs, ys,
            c=freqs, cmap="viridis",
            s=60, edgecolors="black", linewidths=0.5,
            label="Clientes",
        )
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.7, label="Frecuencia de visita")
        cbar.set_ticks(sorted(set(freqs.tolist())))
    elif color_by == "demand":
        demands = np.array([c.demand for c in instance.customers])
        scatter = ax.scatter(
            xs, ys,
            c=demands, cmap="plasma",
            s=60, edgecolors="black", linewidths=0.5,
        )
        fig.colorbar(scatter, ax=ax, shrink=0.7, label="Demanda")
    else:
        ax.scatter(
            xs, ys,
            color="steelblue", s=50, edgecolors="black", linewidths=0.5,
            label="Clientes",
        )

    ax.scatter(
        [instance.depot.x], [instance.depot.y],
        marker="s", color="red", s=180,
        edgecolors="black", linewidths=1.2, zorder=5,
        label="Depósito",
    )

    if show_ids:
        for c in instance.customers:
            ax.annotate(
                str(c.id),
                (c.x, c.y),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
            )

    ax.set_xlabel("Coordenada X")
    ax.set_ylabel("Coordenada Y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", framealpha=0.9)

    if title is None:
        title = (
            f"Instancia {instance.name} "
            f"({instance.num_customers} clientes, horizonte {instance.horizon} días)"
        )
    ax.set_title(title)

    return fig


# Histograma de demandas de clientes con la capacidad vehicular marcada.
def plot_demand_distribution(
    instance: Instance,
    *,
    ax: Optional[Axes] = None,
) -> Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    demands = [c.demand for c in instance.customers]

    ax.hist(demands, bins=20, color="steelblue", edgecolor="black", alpha=0.8)
    ax.axvline(
        instance.capacity, color="red", linestyle="--", linewidth=2,
        label=f"Capacidad vehicular (Q = {instance.capacity:g})",
    )

    ax.set_xlabel("Demanda por visita")
    ax.set_ylabel("Número de clientes")
    ax.set_title(f"Distribución de demandas en {instance.name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


# Gráfico de barras con la cantidad de clientes por frecuencia de visita.
def plot_frequency_distribution(
    instance: Instance,
    *,
    ax: Optional[Axes] = None,
) -> Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    freqs = [c.frequency for c in instance.customers]
    unique_freqs = sorted(set(freqs))
    counts = [freqs.count(f) for f in unique_freqs]

    bars = ax.bar(
        [str(f) for f in unique_freqs], counts,
        color="steelblue", edgecolor="black", alpha=0.8,
    )

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            str(count),
            ha="center", va="bottom", fontsize=10,
        )

    ax.set_xlabel("Frecuencia requerida (visitas en el horizonte)")
    ax.set_ylabel("Número de clientes")
    ax.set_title(f"Distribución de frecuencias en {instance.name}")
    ax.grid(True, alpha=0.3, axis="y")

    return fig


# Visualiza una solución completa, un subplot por día con sus rutas coloreadas.
def plot_solution(
    instance: Instance,
    solution: Solution,
    *,
    days: Optional[Sequence[int]] = None,
    title: Optional[str] = None,
) -> Figure:
    if days is None:
        days = list(range(1, instance.horizon + 1))

    n_days = len(days)
    ncols = min(n_days, 2)
    nrows = (n_days + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(7 * ncols, 6 * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    cust_x = [c.x for c in instance.customers]
    cust_y = [c.y for c in instance.customers]
    cust_by_id = {c.id: c for c in instance.customers}

    for i, day in enumerate(days):
        ax = axes_flat[i]

        ax.scatter(cust_x, cust_y, color="lightgray", s=30, zorder=1)

        ax.scatter(
            [instance.depot.x], [instance.depot.y],
            marker="s", color="red", s=140,
            edgecolors="black", linewidths=1.2, zorder=5,
        )

        day_routes = solution.routes_by_day(day)
        color = _day_color(day)

        for r in day_routes:
            coords_x = []
            coords_y = []
            for node_id in r.nodes:
                if node_id == 0:
                    coords_x.append(instance.depot.x)
                    coords_y.append(instance.depot.y)
                else:
                    cust = cust_by_id[node_id]
                    coords_x.append(cust.x)
                    coords_y.append(cust.y)

            ax.plot(coords_x, coords_y, "-", color=color, linewidth=1.5, zorder=2)
            visited_x = [coords_x[k] for k in range(len(r.nodes)) if r.nodes[k] != 0]
            visited_y = [coords_y[k] for k in range(len(r.nodes)) if r.nodes[k] != 0]
            ax.scatter(
                visited_x, visited_y,
                color=color, s=55, edgecolors="black", linewidths=0.6, zorder=3,
            )

        ax.set_title(f"Día {day} — {len(day_routes)} ruta(s)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)

    for j in range(n_days, len(axes_flat)):
        axes_flat[j].axis("off")

    if title is None:
        cost = solution.total_cost(instance)
        title = f"Solución sobre {instance.name} (costo total = {cost:.2f})"
    fig.suptitle(title, fontsize=14, y=1.00)
    fig.tight_layout()

    return fig
