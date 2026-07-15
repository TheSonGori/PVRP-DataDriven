"""
Script de visualización de rutas (Día 15, Fase E).

Carga un agente entrenado, resuelve una instancia y dibuja sus rutas.
Opcionalmente dibuja también la BKS para comparación visual.

    python scripts/plot_routes.py --instance p01 \
        --model results/models/ppo_p01 --compare-bks

Genera archivos PNG en results/figures/.
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


def _solve_with_agent(model, instance, seed: int = 42):
    """Ejecuta el agente y devuelve la solución construida."""
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualización de rutas.")
    parser.add_argument("--instance", type=str, default="p01")
    parser.add_argument("--model", type=str,
                        default="results/models/ppo_p01")
    parser.add_argument("--compare-bks", action="store_true",
                        help="Generar también la figura de la BKS.")
    args = parser.parse_args()

    instance = load_instance(DATA_DIR / f"{args.instance}.txt")

    # --- Solución del agente RL ---
    model_path = PROJECT_ROOT / args.model
    if not Path(str(model_path) + ".zip").exists():
        print(f"ERROR: no existe el modelo {model_path}.zip")
        return 1

    print(f"Cargando agente: {model_path}")
    model = MaskablePPO.load(str(model_path))
    sol_rl = _solve_with_agent(model, instance)
    cost_rl = sol_rl.total_cost(instance)
    feas, _ = sol_rl.is_feasible(instance)

    out_rl = FIG_DIR / f"rutas_{args.instance}_RL.png"
    plot_solution(
        instance, sol_rl, save_path=out_rl,
    )
    print(f"Figura RL guardada: {out_rl}")

    # --- BKS opcional ---
    if args.compare_bks:
        bks_path = DATA_DIR / f"{args.instance}.res"
        if bks_path.exists():
            sol_bks = load_solution(bks_path)
            out_bks = FIG_DIR / f"rutas_{args.instance}_BKS.png"
            plot_solution(
                instance, sol_bks, save_path=out_bks,
            )
            print(f"Figura BKS guardada: {out_bks}")
        else:
            print(f"Aviso: no existe {bks_path}, se omite la BKS.")

    return 0


if __name__ == "__main__":
    sys.exit(main())