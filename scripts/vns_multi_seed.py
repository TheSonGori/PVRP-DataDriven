"""
Multi-semilla del VNS sobre las instancias del conjunto NEO evaluadas.

Ejecuta el VNS con varias semillas fijas y reproducibles sobre cada instancia
y reporta, por instancia, la media y la desviación estándar del costo y del
gap respecto al BKS, el rango observado, la tasa de factibilidad y el tiempo
medio de ejecución. Todos los valores del resumen se calculan a partir de las
corridas de esta misma ejecución: el script no contiene resultados de
referencia precargados.

Uso:
    python scripts/vns_multi_seed.py
    python scripts/vns_multi_seed.py --instancias p03 --iters 250
    python scripts/vns_multi_seed.py --iters 1000 --seeds 0 1 2 3 4

Entrada: instancias en data/raw/*.txt y sus BKS en data/raw/*.res.
Salida: resumen por consola y bloque LaTeX copiable con los valores medidos.
"""

from __future__ import annotations

import argparse
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


# Mejor solución conocida (BKS) de una instancia, leída de su archivo .res.
def _bks(name: str):
    p = DATA_DIR / f"{name}.res"
    if not p.exists():
        return None
    try:
        return load_solution(p).reported_cost
    except Exception:
        return None


# Ejecuta el VNS sobre una instancia con todas las semillas indicadas.
def evaluar_instancia(nombre, seeds, max_iterations, k_max, verbose=True):
    instance = load_instance(DATA_DIR / f"{nombre}.txt")
    bks = _bks(nombre)

    if verbose:
        bks_s = f"{bks:.2f}" if bks else "N/A"
        print(f"\n--- {nombre} (BKS: {bks_s}) ---")

    resultados = []
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

        cost = res.final_cost
        feasible, violaciones = res.solution.is_feasible(instance)
        gap = (cost - bks) / bks * 100 if bks else None

        resultados.append({
            "seed": seed,
            "cost": cost,
            "gap": gap,
            "feasible": feasible,
            "time": elapsed,
            "violaciones": violaciones,
        })

        if verbose:
            gap_s = f"{gap:+.2f}%" if gap is not None else "N/A"
            feas_s = "sí" if feasible else "NO"
            print(f"  seed={seed}: costo={cost:.2f}  gap={gap_s}  "
                  f"factible={feas_s}  t={elapsed:.3f}s")
            if not feasible:
                for v in violaciones[:3]:
                    print(f"      violación: {v}")

    return {"instance": nombre, "bks": bks, "resultados": resultados}


# Estadísticos agregados sobre las semillas factibles de una instancia.
def resumen_instancia(data):
    factibles = [r for r in data["resultados"] if r["feasible"]]
    n_feas, n_total = len(factibles), len(data["resultados"])
    base = {
        "instance": data["instance"],
        "bks": data["bks"],
        "feas_ratio": f"{n_feas}/{n_total}",
        "time_mean": mean([r["time"] for r in data["resultados"]]),
    }

    if n_feas == 0:
        base.update({
            "gap_mean": None, "gap_std": None, "gap_min": None, "gap_max": None,
            "cost_mean": None, "cost_std": None,
        })
        return base

    gaps = [r["gap"] for r in factibles]
    costs = [r["cost"] for r in factibles]
    base.update({
        "gap_mean": mean(gaps) if gaps[0] is not None else None,
        "gap_std": stdev(gaps) if len(gaps) > 1 and gaps[0] is not None else 0.0,
        "gap_min": min(gaps) if gaps[0] is not None else None,
        "gap_max": max(gaps) if gaps[0] is not None else None,
        "cost_mean": mean(costs),
        "cost_std": stdev(costs) if len(costs) > 1 else 0.0,
        "time_mean": mean([r["time"] for r in factibles]),
    })
    return base


# Corre las instancias pedidas con las semillas dadas y consolida resumen + bloque LaTeX.
def main():
    ap = argparse.ArgumentParser(description="Multi-semilla del VNS sobre instancias NEO.")
    ap.add_argument("--instancias", nargs="+",
                    default=["p01", "p02", "p03", "p04", "p05", "p06", "p07", "p08", "p09"])
    ap.add_argument("--iters", type=int, default=250, help="max_iterations del VNS")
    ap.add_argument("--k-max", type=int, default=6, dest="k_max")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    print(f"\n{'=' * 92}")
    print("  VNS MULTI-SEMILLA — conjunto NEO")
    print(f"  Parámetros: max_iterations={args.iters}, k_max={args.k_max}, seeds={args.seeds}")
    print(f"{'=' * 92}")

    resumenes = []
    for nombre in args.instancias:
        data = evaluar_instancia(nombre, args.seeds, args.iters, args.k_max)
        resumenes.append(resumen_instancia(data))

    print(f"\n{'=' * 92}")
    print(f"  RESUMEN — {len(args.seeds)} semillas, max_iterations={args.iters}, k_max={args.k_max}")
    print(f"{'=' * 92}")
    print(f"  {'Inst.':<7} {'BKS':>9} {'Costo medio':>18} {'Gap medio':>17} "
          f"{'Rango gap':>21} {'Factib.':>8} {'T medio':>9}")
    print(f"  {'-' * 90}")
    for r in resumenes:
        bks_s = f"{r['bks']:.2f}" if r["bks"] else "N/A"
        if r["cost_mean"] is not None:
            costo_s = f"{r['cost_mean']:.2f} ± {r['cost_std']:.2f}"
        else:
            costo_s = "infactible"
        if r["gap_mean"] is not None:
            gap_s = f"{r['gap_mean']:+.2f}% ± {r['gap_std']:.2f}%"
            rango_s = f"[{r['gap_min']:+.2f}, {r['gap_max']:+.2f}]"
        else:
            gap_s, rango_s = "N/A", "N/A"
        print(f"  {r['instance']:<7} {bks_s:>9} {costo_s:>18} {gap_s:>17} "
              f"{rango_s:>21} {r['feas_ratio']:>8} {r['time_mean']:>8.3f}s")

    print(f"\n{'=' * 92}")
    print("  BLOQUE PARA TABLA LATEX (copiable)")
    print(f"{'=' * 92}")
    print("  % Instancia & BKS & Costo medio & Gap medio & Factib. & T medio \\\\")
    for r in resumenes:
        bks_s = f"{r['bks']:.2f}" if r["bks"] else "---"
        if r["gap_mean"] is not None:
            print(f"    {r['instance']} & ${bks_s}$ & "
                  f"${r['cost_mean']:.2f} \\pm {r['cost_std']:.2f}$ & "
                  f"${r['gap_mean']:+.1f} \\pm {r['gap_std']:.1f}\\%$ & "
                  f"${r['feas_ratio']}$ & ${r['time_mean']:.1f}$~s \\\\")
        else:
            print(f"    {r['instance']} & ${bks_s}$ & infactible & --- & "
                  f"${r['feas_ratio']}$ & ${r['time_mean']:.1f}$~s \\\\")
    print()


if __name__ == "__main__":
    main()
