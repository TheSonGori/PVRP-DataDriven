"""
Funciones de visualización para instancias y soluciones del PVRP.

Estas funciones están diseñadas para producir figuras de calidad académica que
pueden incluirse directamente en la memoria (Capítulos 1, 4 y 5). Se separan
del notebook de exploración para que sean reutilizables desde otros contextos
(scripts de experimentación, generación automática de figuras del Capítulo 4).

Convenciones de estilo:

    - Depósito: cuadrado rojo grande.
    - Clientes: puntos azules; tamaño/color pueden codificar frecuencia o demanda.
    - Rutas: líneas con colores distintivos según día.
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.data.instance import Instance
from src.utils.solution import Solution


# =============================================================================
#  Paletas y constantes de estilo
# =============================================================================

_DAY_PALETTE = [
    "#1f77b4",  # azul
    "#ff7f0e",  # naranja
    "#2ca02c",  # verde
    "#d62728",  # rojo
    "#9467bd",  # morado
    "#8c564b",  # marrón
    "#e377c2",  # rosa
]


def _day_color(day: int) -> str:
    """Color asociado a un día del horizonte (cíclico)."""
    return _DAY_PALETTE[(day - 1) % len(_DAY_PALETTE)]


# =============================================================================
#  Visualización de instancias
# =============================================================================

def plot_instance(
    instance: Instance,
    *,
    ax: Optional[Axes] = None,
    color_by: str = "frequency",
    show_ids: bool = False,
    title: Optional[str] = None,
) -> Figure:
    """
    Dibuja la disposición geográfica de los nodos de una instancia.

    Args:
        instance: Instancia a visualizar.
        ax: Eje matplotlib opcional. Si no se provee, se crea una figura nueva.
        color_by: Atributo que codifica el color de los clientes.
            "frequency" -> color discreto por frecuencia de visita.
            "demand"    -> mapa de calor continuo según demanda.
            "none"      -> todos los clientes del mismo color.
        show_ids: Si True, etiqueta cada cliente con su ID.
        title: Título personalizado. Si es None, se usa el nombre de la instancia.

    Returns:
        La figura de matplotlib creada o utilizada.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 9))
    else:
        fig = ax.figure

    xs = np.array([c.x for c in instance.customers])
    ys = np.array([c.y for c in instance.customers])

    # Clientes
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

    # Depósito
    ax.scatter(
        [instance.depot.x], [instance.depot.y],
        marker="s", color="red", s=180,
        edgecolors="black", linewidths=1.2, zorder=5,
        label="Depósito",
    )

    # Etiquetas de IDs (opcional)
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


def plot_demand_distribution(
    instance: Instance,
    *,
    ax: Optional[Axes] = None,
) -> Figure:
    """
    Dibuja un histograma de las demandas de los clientes con la capacidad
    vehicular marcada como referencia.

    Args:
        instance: Instancia a analizar.
        ax: Eje matplotlib opcional.

    Returns:
        Figura matplotlib.
    """
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


def plot_frequency_distribution(
    instance: Instance,
    *,
    ax: Optional[Axes] = None,
) -> Figure:
    """
    Dibuja un gráfico de barras con la cantidad de clientes por frecuencia
    de visita.

    Args:
        instance: Instancia a analizar.
        ax: Eje matplotlib opcional.

    Returns:
        Figura matplotlib.
    """
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


# =============================================================================
#  Visualización de soluciones
# =============================================================================

def plot_solution(
    instance: Instance,
    solution: Solution,
    *,
    days: Optional[Sequence[int]] = None,
    title: Optional[str] = None,
) -> Figure:
    """
    Visualiza una solución del PVRP separando las rutas por día.

    Cada día se dibuja en un subplot independiente, con sus rutas en colores
    distintos. Esta figura es ideal para incluir en el Capítulo 4 de la
    memoria al mostrar resultados cualitativos.

    Args:
        instance: Instancia correspondiente a la solución.
        solution: Solución a visualizar.
        days: Días específicos a graficar. Si es None, se grafican todos.
        title: Título global de la figura. Si es None, se genera automáticamente.

    Returns:
        Figura matplotlib con un subplot por día.
    """
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

    # Coordenadas de todos los clientes para dibujarlos como fondo
    cust_x = [c.x for c in instance.customers]
    cust_y = [c.y for c in instance.customers]
    cust_by_id = {c.id: c for c in instance.customers}

    for i, day in enumerate(days):
        ax = axes_flat[i]

        # Fondo: todos los clientes (gris claro)
        ax.scatter(cust_x, cust_y, color="lightgray", s=30, zorder=1)

        # Depósito
        ax.scatter(
            [instance.depot.x], [instance.depot.y],
            marker="s", color="red", s=140,
            edgecolors="black", linewidths=1.2, zorder=5,
        )

        # Rutas del día
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
            # Resaltar clientes visitados ese día
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

    # Ocultar subplots no usados
    for j in range(n_days, len(axes_flat)):
        axes_flat[j].axis("off")

    if title is None:
        cost = solution.total_cost(instance)
        title = f"Solución sobre {instance.name} (costo total = {cost:.2f})"
    fig.suptitle(title, fontsize=14, y=1.00)
    fig.tight_layout()

    return fig
