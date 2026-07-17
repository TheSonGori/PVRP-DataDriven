"""
Multi-semilla del VNS con max_iterations=250 sobre las 9 instancias del
conjunto NEO evaluadas.

Este script reemplaza los resultados del multi-semilla anterior (con
max_iterations=100), que resultaron insuficientes según el barrido
de iteraciones del script vns_barrido_iteraciones.py: en p01 el VNS
con 100 iteraciones no había convergido (gap +31,70%), mientras que
con 250 iteraciones el gap baja a +22,27% y se estanca ahí incluso
con 1000 iteraciones. Esto demuestra que 250 iteraciones es el punto
de convergencia empírica del método sobre la instancia más pequeña.

Se adopta max_iterations=250 como valor único para todas las instancias,
en línea con el pedido de la profesora guía de "no perjudicar al método
de referencia" (VNS) en la comparación.

    python scripts/vns_multi_seed_250.py

Tiempo estimado: ~15-20 minutos totales (VNS es rápido; en las instancias
mayores como p07 y p08 cada corrida puede tardar 30-60 segundos).

Entrada: ninguna (usa las 9 instancias y semillas fijas definidas en main()).
Salida: tabla consolidada de gap/factibilidad/tiempo por instancia, un bloque
LaTeX copiable y una comparación informativa contra la corrida de 100
iteraciones; todo impreso en consola.
"""

from __future__ import annotations
import sys
import time
from pathlib import Path
from statistics import mean, stdev

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.vns import vns_solve
from src.data.instance_loader import load_instance
from src.data.solution_loader import load_solution


DATA_DIR = PROJECT_ROOT / "data" / "raw"
MAX_ITERATIONS = 250  # <-- único cambio vs vns_multi_seed.py (era 100)


def _bks(name: str):
    p = DATA_DIR / f"{name}.res"
    if p.exists():
        try:
            return load_solution(p).reported_cost
        except Exception:
            return None
    return None


# Ejecuta el VNS sobre una instancia con las semillas dadas.
def evaluar_instancia(nombre, seeds, max_iterations=MAX_ITERATIONS, k_max=3, verbose=True):
    instance = load_instance(DATA_DIR / f"{nombre}.txt")
    bks = _bks(nombre)

    resultados = []

    if verbose:
        print(f"\n--- {nombre} (BKS: {bks:.2f}) ---")

    for seed in seeds:
        t0 = time.time()
        res = vns_solve(
            instance,
            max_iterations=max_iterations,
            k_max=k_max,
            seed=seed,
            verbose=False,
        )
        elapsed = time.time() - t0

        cost = (getattr(res, "best_cost", None)
                or getattr(res, "cost", None)
                or res.solution.total_cost(instance))
        solution = (getattr(res, "solution", None)
                    or getattr(res, "best_solution", None))
        feasible, _ = solution.is_feasible(instance) if solution else (False, None)

        gap = (cost - bks) / bks * 100 if bks else None

        resultados.append({
            "seed": seed,
            "cost": cost,
            "gap": gap,
            "feasible": feasible,
            "time": elapsed,
        })

        if verbose:
            gap_s = f"{gap:+.2f}%" if gap is not None else "N/A"
            feas_s = "sí" if feasible else "NO"
            print(f"  seed={seed}: costo={cost:.2f}  gap={gap_s}  "
                  f"factible={feas_s}  t={elapsed:.3f}s")

    return {"instance": nombre, "bks": bks, "resultados": resultados}


# Calcula estadísticos agregados (gap, costo, tiempo) sobre las semillas factibles.
def resumen_instancia(data):
    factibles = [r for r in data["resultados"] if r["feasible"]]
    n_feas = len(factibles)
    n_total = len(data["resultados"])

    if n_feas == 0:
        return {
            "instance": data["instance"],
            "bks": data["bks"],
            "gap_mean": None, "gap_std": None,
            "gap_min": None, "gap_max": None,
            "cost_mean": None, "cost_std": None,
            "time_mean": mean([r["time"] for r in data["resultados"]]),
            "feas_ratio": f"{n_feas}/{n_total}",
        }

    gaps = [r["gap"] for r in factibles]
    costs = [r["cost"] for r in factibles]
    times = [r["time"] for r in factibles]

    return {
        "instance": data["instance"],
        "bks": data["bks"],
        "gap_mean": mean(gaps),
        "gap_std": stdev(gaps) if len(gaps) > 1 else 0.0,
        "gap_min": min(gaps),
        "gap_max": max(gaps),
        "cost_mean": mean(costs),
        "cost_std": stdev(costs) if len(costs) > 1 else 0.0,
        "time_mean": mean(times),
        "feas_ratio": f"{n_feas}/{n_total}",
    }


# Corre las 9 instancias con 5 semillas cada una y consolida resumen + bloque LaTeX.
def main():
    instancias = ["p01", "p02", "p03", "p04", "p05", "p06", "p07", "p08", "p09"]
    seeds = [0, 1, 2, 3, 4]

    print(f"\n{'='*84}")
    print("  VNS MULTI-SEMILLA — 9 instancias del conjunto NEO")
    print(f"  Parámetros: max_iterations={MAX_ITERATIONS}, k_max=3, seeds={seeds}")
    print(f"  Justificación: punto de convergencia empírica según barrido de iteraciones")
    print(f"{'='*84}")

    resumenes = []
    for nombre in instancias:
        data = evaluar_instancia(nombre, seeds, verbose=True)
        resumenes.append(resumen_instancia(data))

    # Resumen consolidado
    print(f"\n{'='*84}")
    print(f"  RESUMEN CONSOLIDADO — VNS con 5 semillas y max_iterations={MAX_ITERATIONS}")
    print(f"{'='*84}")
    print(f"  {'Instancia':<10} {'BKS':>10} {'Gap medio':>16} {'Rango':>22} "
          f"{'Factib.':>9} {'T medio':>10}")
    print(f"  {'-'*84}")

    for r in resumenes:
        bks_s = f"{r['bks']:.2f}" if r['bks'] else "N/A"
        if r['gap_mean'] is not None:
            gap_mean_s = f"{r['gap_mean']:+.2f}% ± {r['gap_std']:.2f}%"
            rango_s = f"[{r['gap_min']:+.2f}, {r['gap_max']:+.2f}]"
        else:
            gap_mean_s = "N/A"
            rango_s = "N/A"
        t_s = f"{r['time_mean']:.3f}s"
        print(f"  {r['instance']:<10} {bks_s:>10} {gap_mean_s:>16} "
              f"{rango_s:>22} {r['feas_ratio']:>9} {t_s:>10}")

    # LaTeX helper
    print(f"\n{'='*84}")
    print("  BLOQUE PARA TABLA LATEX (copiable)")
    print(f"{'='*84}")
    print("  % Formato: Instancia & BKS & Gap medio & Desv. std & Factibilidad & T medio \\\\")
    for r in resumenes:
        if r['gap_mean'] is not None:
            line = (f"    {r['instance']} & {r['bks']:.2f} & "
                    f"$+{r['gap_mean']:.2f}\\%$ & $\\pm{r['gap_std']:.2f}\\%$ & "
                    f"{r['feas_ratio']} & {r['time_mean']:.3f}s \\\\")
        else:
            line = (f"    {r['instance']} & {r['bks']:.2f} & "
                    f"infactible & --- & {r['feas_ratio']} & {r['time_mean']:.3f}s \\\\")
        print(line)
    print()

    # Comparación explícita con la versión de 100 iteraciones (referencia informativa)
    print(f"{'='*84}")
    print("  COMPARACIÓN INFORMATIVA CON max_iterations=100")
    print(f"{'='*84}")
    print("  Referencia (100 iter):")
    print("    p01: +31.26% ± 0.90%   p02: +20.63% ± 0.84%   p03: +82.64% ± 3.28%")
    print("    p04: +23.60% ± 0.49%   p05: +17.52% ± 0.64%   p06: +105.83% ± 2.88%")
    print("    p07: +26.63% ± 0.50%   p08: +17.82% ± 0.60%   p09: +125.55% ± 3.54%")
    print(f"\n  Nueva ({MAX_ITERATIONS} iter):")
    for r in resumenes:
        if r['gap_mean'] is not None:
            print(f"    {r['instance']}: {r['gap_mean']:+.2f}% ± {r['gap_std']:.2f}%")
    print(f"{'='*84}\n")


if __name__ == "__main__":
    main()