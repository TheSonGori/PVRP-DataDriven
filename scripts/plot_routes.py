"""
CLI de visualización de rutas: carga un agente entrenado, resuelve una
instancia y dibuja sus rutas; opcionalmente dibuja también la BKS para
comparación visual.

    # Semilla explícita del protocolo multi-semilla
    python scripts/plot_routes.py --instance p03 --seed 1 --compare-bks

    # Ruta de modelo arbitraria (por ejemplo, un modelo sin sufijo de semilla)
    python scripts/plot_routes.py --instance p01 --model results/models/ppo_p01

Al terminar, el script imprime el costo, el gap respecto a la BKS y la
factibilidad de la solución dibujada, junto con la línea de \\caption{} de
LaTeX correspondiente. Los valores de los captions deben copiarse desde esa
salida y no transcribirse a mano: es la única forma de garantizar que la
figura y su descripción correspondan a la misma corrida.

Entrada: argumentos --instance, --seed o --model, y --compare-bks.
Salida: archivos PNG en results/figures/ (rutas_<instance>_RL.png y,
si se pide, rutas_<instance>_BKS.png). Código de salida 0/1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sb3_contrib import MaskablePPO

from src.agent.train import build_env
from src.data.instance_loader import load_instance
from src.data.solution_loader import load_solution
from src.utils.route_plot import plot_solution


DATA_DIR = PROJECT_ROOT / "data" / "raw"
FIG_DIR = PROJECT_ROOT / "results" / "figures"

# Semilla del entorno en la evaluación determinística. Debe coincidir con la de
# src/agent/evaluate.py para que la figura reproduzca el resultado reportado.
EVAL_SEED = 42


# Ejecuta el agente sobre la instancia y devuelve la solución construida.
def _solve_with_agent(model, instance, seed: int = EVAL_SEED):
    env = build_env(instance, seed=seed)
    obs, _ = env.reset()
    terminated = False
    steps = 0
    while not terminated and steps < 5000:
        mask = env.action_masks()
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        obs, _, terminated, _, _ = env.step(int(action))
        steps += 1
    return env.unwrapped.get_solution()


# Costo de la BKS de una instancia, o None si no hay .res disponible.
def _bks_cost(name: str):
    p = DATA_DIR / f"{name}.res"
    if not p.exists():
        return None
    try:
        return load_solution(p).reported_cost
    except Exception:
        return None


# Formatea un número al estilo de la memoria: coma decimal dentro de $...$.
def _tex(x: float, dec: int = 1) -> str:
    return f"${x:.{dec}f}$".replace(".", "{,}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualización de rutas.")
    parser.add_argument("--instance", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None,
                        help="Semilla del protocolo multi-semilla. Carga "
                             "results/models/ppo_<instance>_seed<seed>.")
    parser.add_argument("--model", type=str, default=None,
                        help="Ruta al modelo (sin .zip). Alternativa a --seed.")
    parser.add_argument("--compare-bks", action="store_true",
                        help="Generar también la figura de la BKS.")
    args = parser.parse_args()

    if (args.seed is None) == (args.model is None):
        print("ERROR: indica exactamente uno de --seed o --model.")
        print("  Ejemplo: python scripts/plot_routes.py --instance p03 --seed 1 --compare-bks")
        return 1

    instance = load_instance(DATA_DIR / f"{args.instance}.txt")

    if args.seed is not None:
        model_path = PROJECT_ROOT / "results" / "models" / \
            f"ppo_{args.instance}_seed{args.seed}"
        etiqueta = f"semilla ${args.seed}$"
    else:
        model_path = PROJECT_ROOT / args.model
        etiqueta = None

    if not Path(str(model_path) + ".zip").exists():
        print(f"ERROR: no existe el modelo {model_path}.zip")
        return 1

    print(f"Cargando agente: {model_path}")
    model = MaskablePPO.load(str(model_path))
    sol_rl = _solve_with_agent(model, instance)
    cost_rl = sol_rl.total_cost(instance)
    feasible, violaciones = sol_rl.is_feasible(instance)
    bks = _bks_cost(args.instance)
    gap = (cost_rl - bks) / bks * 100 if bks else None

    out_rl = FIG_DIR / f"rutas_{args.instance}_RL.png"
    plot_solution(instance, sol_rl, save_path=out_rl)
    print(f"Figura RL guardada: {out_rl}")

    print()
    print(f"  Instancia   : {args.instance} ({instance.num_customers} clientes, "
          f"T={instance.horizon}, K={instance.num_vehicles})")
    print(f"  Costo       : {cost_rl:.2f}")
    if bks:
        print(f"  BKS         : {bks:.2f}")
        print(f"  Gap         : {gap:+.2f}%")
    print(f"  Factible    : {'sí' if feasible else 'NO'}")
    if not feasible:
        for v in violaciones[:3]:
            print(f"     violación: {v}")

    partes = ["Solución generada por el agente RL"]
    if etiqueta:
        partes.append(etiqueta)
    detalle = ", ".join(partes)
    gap_s = f", gap {_tex(gap)}\\%" if gap is not None else ""
    feas_s = "factible" if feasible else "\\textbf{infactible}"
    print()
    print("  Caption para LaTeX (copiar tal cual):")
    print(f"    \\caption{{{detalle}: costo {_tex(cost_rl)}{gap_s}, {feas_s}.}}")

    if args.compare_bks:
        bks_path = DATA_DIR / f"{args.instance}.res"
        if bks_path.exists():
            sol_bks = load_solution(bks_path)
            out_bks = FIG_DIR / f"rutas_{args.instance}_BKS.png"
            plot_solution(instance, sol_bks, save_path=out_bks)
            print(f"\nFigura BKS guardada: {out_bks}")
            print(f"    \\caption{{Mejor solución conocida BKS: costo {_tex(bks)}.}}")
        else:
            print(f"Aviso: no existe {bks_path}, se omite la BKS.")

    return 0


if __name__ == "__main__":
    sys.exit(main())