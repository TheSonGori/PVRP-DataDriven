"""
Comparación de métodos a presupuesto de tiempo equiparado.

Las comparaciones habituales de este trabajo enfrentan al agente contra el VNS
en su configuración completa (8000 iteraciones), donde el VNS dispone de entre
95 y 2415 segundos por corrida y el agente responde en centésimas de segundo.
Este script responde la pregunta complementaria: dado el presupuesto de tiempo
que consume el agente, ¿qué es lo mejor que entrega un método clásico?

Para cada instancia se reportan, con su tiempo medido:

  * Greedy                    construcción por vecino más cercano.
  * Greedy + 2opt             más reordenamiento intra-ruta.
  * Greedy + intra_day        más movimientos entre rutas de un mismo día.
  * Solución inicial del VNS  Greedy más la búsqueda local completa del VNS,
                              incluida la reubicación entre días. Es lo que el
                              VNS produce antes de su primera iteración, de modo
                              que ningún presupuesto de tiempo menor puede
                              mejorarlo. Se obtiene con vns_solve(max_iterations=0).
  * RL                        política determinística, por semilla.
  * RL + 2opt / RL + intra_day

Todos los tiempos son de pared, medidos alrededor de cada llamada. El tiempo
del RL no incluye el entrenamiento, que es un costo previo y único por
instancia, y se reporta por separado en la memoria.

Uso:
    python scripts/comparacion_tiempo.py
    python scripts/comparacion_tiempo.py --instancias p01 p03 --seeds 0 1 2 3 4

Requiere los modelos guardados en results/models/ (entrenar con
scripts/multi_seed.py --save-models).
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sb3_contrib import MaskablePPO

from src.agent.evaluate import apply_local_search
from src.agent.multi_seed import model_path
from src.agent.train import build_env
from src.baselines.greedy import greedy_solve
from src.baselines.vns import vns_solve
from src.data.instance_loader import load_instance
from src.data.solution_loader import load_solution

DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "results" / "models"

INSTANCIAS_POR_DEFECTO = ["p01", "p03", "p04", "p06", "p07", "p09"]


# Mejor solución conocida (BKS) de una instancia, o None si no hay .res.
def _bks(name: str):
    p = DATA_DIR / f"{name}.res"
    if not p.exists():
        return None
    try:
        return load_solution(p).reported_cost
    except Exception:
        return None


# Gap porcentual respecto al BKS.
def _gap(cost, bks):
    return (cost - bks) / bks * 100 if bks else None


# Ejecuta un episodio determinístico con el modelo y devuelve la solución.
def _run_episode(model, env, max_steps: int = 5000):
    obs, _ = env.reset()
    terminated = False
    steps = 0
    while not terminated and steps < max_steps:
        mask = env.action_masks()
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        steps += 1
    return env.unwrapped.get_solution()


# Métodos clásicos bajo presupuesto reducido. Devuelve una lista de
# (etiqueta, gap, tiempo, factible).
def _clasicos(instance, bks):
    filas = []

    t0 = time.time()
    sol = greedy_solve(instance)
    t_greedy = time.time() - t0
    feas, _ = sol.is_feasible(instance)
    filas.append(("Greedy", _gap(sol.total_cost(instance), bks), t_greedy, feas))

    for nivel in ("2opt", "intra_day"):
        t0 = time.time()
        base = greedy_solve(instance)
        refinada = apply_local_search(base, instance, level=nivel)
        t = time.time() - t0
        feas, _ = refinada.is_feasible(instance)
        filas.append((f"Greedy + {nivel}", _gap(refinada.total_cost(instance), bks), t, feas))

    # Solución inicial del VNS: Greedy más su búsqueda local completa, sin
    # ejecutar ninguna iteración de shaking. Es el piso temporal del VNS.
    t0 = time.time()
    res = vns_solve(instance, max_iterations=0, seed=0)
    t = time.time() - t0
    feas, _ = res.solution.is_feasible(instance)
    filas.append(("VNS (solución inicial)", _gap(res.final_cost, bks), t, feas))

    return filas


# Variantes del agente sobre las semillas dadas. Devuelve una lista de
# (etiqueta, gap medio, desv., tiempo medio, factibles/total).
def _agente(instance, bks, seeds):
    filas = []
    for nivel in ("none", "2opt", "intra_day"):
        gaps, tiempos, factibles = [], [], 0
        for seed in seeds:
            path = model_path(MODELS_DIR, instance.name, seed)
            if not Path(str(path) + ".zip").exists():
                raise FileNotFoundError(f"Falta el modelo {path}.zip")
            model = MaskablePPO.load(str(path))
            env = build_env(instance, seed=42)

            t0 = time.time()
            sol = _run_episode(model, env)
            if nivel != "none":
                sol = apply_local_search(sol, instance, level=nivel)
            t = time.time() - t0

            feas, _ = sol.is_feasible(instance)
            if feas:
                factibles += 1
                gaps.append(_gap(sol.total_cost(instance), bks))
            tiempos.append(t)

        etiqueta = "RL" if nivel == "none" else f"RL + {nivel}"
        media = st.mean(gaps) if gaps else None
        desv = st.stdev(gaps) if len(gaps) > 1 else 0.0
        filas.append((etiqueta, media, desv, st.mean(tiempos),
                      f"{factibles}/{len(seeds)}"))
    return filas


# Corre los métodos clásicos y las variantes del agente sobre cada instancia pedida.
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Comparación de métodos a presupuesto de tiempo equiparado."
    )
    ap.add_argument("--instancias", nargs="+", default=INSTANCIAS_POR_DEFECTO)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    print(f"\n{'=' * 84}")
    print("  COMPARACIÓN A PRESUPUESTO DE TIEMPO EQUIPARADO")
    print(f"  Semillas del agente: {args.seeds}   |   tiempos de pared medidos")
    print(f"{'=' * 84}")

    for nombre in args.instancias:
        instance = load_instance(DATA_DIR / f"{nombre}.txt")
        bks = _bks(nombre)
        bks_s = f"{bks:.2f}" if bks else "N/A"

        print(f"\n--- {nombre} (BKS: {bks_s}, {instance.num_customers} clientes, "
              f"T={instance.horizon}, K={instance.num_vehicles}) ---")
        print(f"  {'Método':<26} {'Gap':>16} {'Tiempo':>11} {'Factible':>10}")
        print(f"  {'-' * 66}")

        for etiqueta, gap, t, feas in _clasicos(instance, bks):
            g = f"{gap:+.2f}%" if gap is not None else "N/A"
            print(f"  {etiqueta:<26} {g:>16} {t:>10.4f}s {('sí' if feas else 'NO'):>10}")

        print(f"  {'-' * 66}")

        for etiqueta, media, desv, t, factib in _agente(instance, bks, args.seeds):
            g = f"{media:+.2f} ± {desv:.2f}%" if media is not None else "infactible"
            print(f"  {etiqueta:<26} {g:>16} {t:>10.4f}s {factib:>10}")

    print(f"\n{'=' * 84}")
    print("  Nota: la fila 'VNS (solución inicial)' es lo que el VNS entrega antes")
    print("  de su primera iteración de shaking. Ningún presupuesto de tiempo")
    print("  inferior a ese permite al VNS producir nada mejor.")
    print(f"{'=' * 84}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
