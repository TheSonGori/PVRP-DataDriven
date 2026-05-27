"""
Evaluación de baselines (Greedy y VNS) sobre varias instancias (Fase E).

Complementa el análisis multi-semilla del agente RL: produce los costos de
los métodos de referencia sobre las mismas instancias, de modo que la tabla
consolidada del Capítulo 4 sea homogénea y comparable.

    python scripts/baselines_all.py --instances p01 p02 p03 --vns-iters 100

A diferencia del RL, estos métodos no entrenan: corren en segundos.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.evaluate import evaluate_greedy, evaluate_vns
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Baselines sobre varias instancias.")
    parser.add_argument("--instances", type=str, nargs="+",
                        default=["p01", "p02", "p03"])
    parser.add_argument("--vns-iters", type=int, default=100)
    args = parser.parse_args()

    print(f"\n{'='*72}")
    print(f"  BASELINES — Greedy y VNS")
    print(f"{'='*72}")
    print(f"  {'Instancia':<10} {'Método':<10} {'Costo':>10} {'BKS':>10} "
          f"{'Gap':>9} {'Tiempo':>9} {'Factible':>10}")
    print(f"  {'-'*68}")

    for name in args.instances:
        instance = load_instance(DATA_DIR / f"{name}.txt")
        bks = _bks(name)

        g = evaluate_greedy(instance, bks_cost=bks)
        v = evaluate_vns(instance, max_iterations=args.vns_iters, bks_cost=bks)

        for label, r in [("Greedy", g), ("VNS", v)]:
            gap = f"{r.gap_pct:+.1f}%" if r.gap_pct is not None else "N/A"
            bks_s = f"{bks:.2f}" if bks is not None else "N/A"
            feas = "sí" if r.feasible else "NO"
            print(f"  {name:<10} {label:<10} {r.cost:>10.2f} {bks_s:>10} "
                  f"{gap:>9} {r.elapsed_time:>8.3f}s {feas:>10}")
        print(f"  {'-'*68}")

    print(f"{'='*72}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())