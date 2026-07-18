"""
Multi-semilla del VNS sobre las instancias del conjunto NEO evaluadas.

Ejecuta el VNS con varias semillas fijas y reproducibles sobre cada instancia
y reporta, por instancia, la media y la desviación estándar del costo y del
gap respecto al BKS, el rango observado, la tasa de factibilidad y el tiempo
medio de ejecución. Todos los valores del resumen se calculan a partir de las
corridas de esta misma ejecución: el script no contiene resultados de
referencia precargados.

El punto de partida del VNS es configurable. Por defecto es la heurística
Greedy, que es el comportamiento reportado en la memoria. Con --init rl, el
VNS arranca desde la solución que construye el agente entrenado con la misma
semilla, lo que permite medir si un mejor punto de partida se traduce en
mejores resultados bajo presupuesto acotado. En ambos casos, vns_solve aplica
búsqueda local sobre la solución inicial antes de comenzar a iterar, de modo
que las dos ramas se comparan en igualdad de condiciones.

Uso:
    python scripts/vns_multi_seed.py
    python scripts/vns_multi_seed.py --instancias p03 --iters 250
    python scripts/vns_multi_seed.py --instancias p01 --iters 500 --init rl

Entrada: instancias en data/raw/*.txt y sus BKS en data/raw/*.res. Con
--init rl, además, los modelos de results/models/ppo_<inst>_seed<n>.zip.
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

from sb3_contrib import MaskablePPO

from src.agent.multi_seed import model_path
from src.agent.train import build_env
from src.baselines.vns import vns_solve
from src.data.instance_loader import load_instance
from src.data.solution_loader import load_solution

DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "results" / "models"

# Semilla del entorno en la evaluación determinística del agente. Debe coincidir
# con la de src/agent/evaluate.py para reproducir las soluciones reportadas.
EVAL_SEED = 42


# Mejor solución conocida (BKS) de una instancia, leída de su archivo .res.
def _bks(name: str):
    p = DATA_DIR / f"{name}.res"
    if not p.exists():
        return None
    try:
        return load_solution(p).reported_cost
    except Exception:
        return None


# Solución construida por el agente entrenado con la semilla indicada.
# Se usa como punto de partida del VNS cuando --init rl.
def _solucion_del_agente(instance, seed):
    path = model_path(MODELS_DIR, instance.name, seed)
    if not Path(str(path) + ".zip").exists():
        raise FileNotFoundError(
            f"No existe el modelo {path}.zip, necesario para --init rl."
        )
    model = MaskablePPO.load(str(path))
    env = build_env(instance, seed=EVAL_SEED)
    obs, _ = env.reset()
    terminated = False
    steps = 0
    while not terminated and steps < 5000:
        mask = env.action_masks()
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        obs, _, terminated, _, _ = env.step(int(action))
        steps += 1
    sol = env.unwrapped.get_solution()
    feasible, violaciones = sol.is_feasible(instance)
    if not feasible:
        raise ValueError(
            f"El agente no produce solución factible en {instance.name} con la "
            f"semilla {seed}, de modo que no puede servir de punto de partida "
            f"del VNS. Primera violación: {violaciones[0] if violaciones else 'N/A'}"
        )
    return sol


# Ejecuta el VNS sobre una instancia con todas las semillas indicadas.
def evaluar_instancia(nombre, seeds, max_iterations, k_max, init="greedy", verbose=True):
    instance = load_instance(DATA_DIR / f"{nombre}.txt")
    bks = _bks(nombre)

    if verbose:
        bks_s = f"{bks:.2f}" if bks else "N/A"
        print(f"\n--- {nombre} (BKS: {bks_s}, arranque: {init}) ---")

    resultados = []
    for seed in seeds:
        inicial = _solucion_del_agente(instance, seed) if init == "rl" else None
        t0 = time.time()
        res = vns_solve(
            instance,
            max_iterations=max_iterations,
            k_max=k_max,
            seed=seed,
            initial_solution=inicial,
            verbose=False,
        )
        elapsed = time.time() - t0

        cost = res.final_cost
        feasible, violaciones = res.solution.is_feasible(instance)
        gap = (cost - bks) / bks * 100 if bks else None

        gap_ini = (res.initial_cost - bks) / bks * 100 if bks else None

        resultados.append({
            "seed": seed,
            "cost": cost,
            "gap": gap,
            "feasible": feasible,
            "time": elapsed,
            "violaciones": violaciones,
            "gap_inicial": gap_ini,
        })

        if verbose:
            gap_s = f"{gap:+.2f}%" if gap is not None else "N/A"
            ini_s = f"{gap_ini:+.2f}%" if gap_ini is not None else "N/A"
            feas_s = "sí" if feasible else "NO"
            print(f"  seed={seed}: arranque={ini_s} -> final={gap_s}  "
                  f"costo={cost:.2f}  factible={feas_s}  t={elapsed:.3f}s")
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
        "gap_inicial_mean": (
            mean([r["gap_inicial"] for r in data["resultados"]])
            if data["resultados"][0]["gap_inicial"] is not None else None
        ),
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


def main():
    ap = argparse.ArgumentParser(description="Multi-semilla del VNS sobre instancias NEO.")
    ap.add_argument("--instancias", nargs="+",
                    default=["p01", "p02", "p03", "p04", "p05", "p06", "p07", "p08", "p09"])
    ap.add_argument("--iters", type=int, default=250, help="max_iterations del VNS")
    ap.add_argument("--k-max", type=int, default=6, dest="k_max")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--init", type=str, default="greedy", choices=["greedy", "rl"],
                    help="Punto de partida del VNS. 'greedy' es el comportamiento "
                         "reportado en la memoria; 'rl' arranca desde la solución "
                         "del agente entrenado con la misma semilla.")
    args = ap.parse_args()

    print(f"\n{'=' * 92}")
    print("  VNS MULTI-SEMILLA — conjunto NEO")
    print(f"  Parámetros: max_iterations={args.iters}, k_max={args.k_max}, "
          f"seeds={args.seeds}, arranque={args.init}")
    print(f"{'=' * 92}")

    resumenes = []
    for nombre in args.instancias:
        data = evaluar_instancia(nombre, args.seeds, args.iters, args.k_max,
                                 init=args.init)
        resumenes.append(resumen_instancia(data))

    print(f"\n{'=' * 92}")
    print(f"  RESUMEN — {len(args.seeds)} semillas, max_iterations={args.iters}, "
          f"k_max={args.k_max}, arranque={args.init}")
    print(f"{'=' * 92}")
    print(f"  {'Inst.':<7} {'BKS':>9} {'Gap arranque':>13} {'Costo medio':>18} "
          f"{'Gap medio':>17} {'Rango gap':>21} {'Factib.':>8} {'T medio':>9}")
    print(f"  {'-' * 104}")
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
        ini_s = (f"{r['gap_inicial_mean']:+.2f}%"
                 if r["gap_inicial_mean"] is not None else "N/A")
        print(f"  {r['instance']:<7} {bks_s:>9} {ini_s:>13} {costo_s:>18} {gap_s:>17} "
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
