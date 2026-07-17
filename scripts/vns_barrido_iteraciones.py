"""
Barrido de max_iterations del VNS para justificar el valor adoptado.

Responde a la objeción "¿100 iteraciones es lo indicado por los autores?
¿No estás perjudicando al VNS con un presupuesto bajo?". Este script ejecuta
VNS con distintos presupuestos de iteraciones (100, 250, 500, 1000) sobre
tres instancias representativas de las tres familias estructurales
(p01, p04, p07), con semilla fija. El objetivo es demostrar empíricamente
si el VNS ya converge con 100 iteraciones o si mejora significativamente
al aumentar el presupuesto.

    python scripts/vns_barrido_iteraciones.py

Tiempo estimado: 5-15 minutos (VNS es rápido, pero 1000 iteraciones tardan
más que 100).

Entrada: ninguna (usa las instancias y presupuestos fijos definidos en main()).
Salida: tabla consolidada de costo/gap/factibilidad/tiempo por instancia y
presupuesto, más una interpretación de la mejora al 10x iteraciones; todo
impreso en consola.
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.vns import vns_solve
from src.data.instance_loader import load_instance
from src.data.solution_loader import load_solution


DATA_DIR = PROJECT_ROOT / "data" / "raw"


def _bks(name: str):
    p = DATA_DIR / f"{name}.res"
    if p.exists():
        try:
            return load_solution(p).reported_cost
        except Exception:
            return None
    return None


# Ejecuta el VNS sobre una instancia con distintos presupuestos de iteraciones.
def evaluar_iteraciones(nombre, iteraciones_lista, seed=0, k_max=3):
    instance = load_instance(DATA_DIR / f"{nombre}.txt")
    bks = _bks(nombre)

    print(f"\n--- {nombre} (BKS: {bks:.2f}, seed={seed}) ---")

    resultados = []
    for max_it in iteraciones_lista:
        t0 = time.time()
        res = vns_solve(
            instance,
            max_iterations=max_it,
            k_max=k_max,
            seed=seed,
            verbose=False,
        )
        elapsed = time.time() - t0

        cost = getattr(res, "best_cost", None) or getattr(res, "cost", None) or \
               res.solution.total_cost(instance)
        solution = getattr(res, "solution", None) or getattr(res, "best_solution", None)
        feasible, _ = solution.is_feasible(instance) if solution else (False, None)
        gap = (cost - bks) / bks * 100 if bks else None

        resultados.append({
            "max_iter": max_it,
            "cost": cost,
            "gap": gap,
            "feasible": feasible,
            "time": elapsed,
        })

        gap_s = f"{gap:+.2f}%" if gap is not None else "N/A"
        feas_s = "sí" if feasible else "NO"
        print(f"  max_iter={max_it:>5}: costo={cost:.2f}  gap={gap_s}  "
              f"factible={feas_s}  t={elapsed:.3f}s")

    return {"instance": nombre, "bks": bks, "resultados": resultados}


# Corre el barrido sobre las tres instancias y consolida un resumen con interpretación.
def main():
    instancias = ["p01", "p04", "p07"]
    iteraciones_lista = [100, 250, 500, 1000]

    print(f"\n{'='*84}")
    print("  BARRIDO DE ITERACIONES DEL VNS")
    print(f"  Instancias: {instancias}")
    print(f"  Presupuestos: {iteraciones_lista}")
    print(f"  seed=0, k_max=3")
    print(f"{'='*84}")

    todos = []
    for nombre in instancias:
        data = evaluar_iteraciones(nombre, iteraciones_lista)
        todos.append(data)

    # Resumen consolidado
    print(f"\n{'='*84}")
    print("  RESUMEN CONSOLIDADO")
    print(f"{'='*84}")
    print(f"  {'Instancia':<10} {'max_iter':>10} {'Gap':>12} {'Costo':>12} "
          f"{'Factib.':>10} {'Tiempo':>10} {'Δgap vs 100':>14}")
    print(f"  {'-'*82}")

    for data in todos:
        gap_100 = data["resultados"][0]["gap"] if data["resultados"][0]["gap"] is not None else 0
        for r in data["resultados"]:
            gap_s = f"{r['gap']:+.2f}%" if r['gap'] is not None else "N/A"
            feas_s = "sí" if r["feasible"] else "NO"
            delta = r['gap'] - gap_100 if r['gap'] is not None else 0
            delta_s = f"{delta:+.2f}%" if r['max_iter'] != 100 else "  (ref.)"
            print(f"  {data['instance']:<10} {r['max_iter']:>10} {gap_s:>12} "
                  f"{r['cost']:>12.2f} {feas_s:>10} {r['time']:>9.3f}s {delta_s:>14}")
        print(f"  {'-'*82}")

    # Interpretación automática
    print(f"\n{'='*84}")
    print("  INTERPRETACIÓN")
    print(f"{'='*84}")
    for data in todos:
        gap_100 = data["resultados"][0]["gap"]
        gap_1000 = data["resultados"][-1]["gap"]
        if gap_100 is not None and gap_1000 is not None:
            mejora = gap_100 - gap_1000
            print(f"  {data['instance']}: gap con 100 iter = {gap_100:+.2f}%, "
                  f"gap con 1000 iter = {gap_1000:+.2f}%. "
                  f"Mejora al 10× iteraciones: {mejora:+.2f} puntos porcentuales.")
    print()


if __name__ == "__main__":
    main()