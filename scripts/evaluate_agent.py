"""
CLI que carga un agente PVRP-RL ya entrenado y lo compara contra Greedy y
VNS (uso: `python scripts/evaluate_agent.py --instance p01 --stochastic-runs
30 --vns-iters 150`).

Entrada: argumentos --instance, --model-path (opcional, por defecto
results/models/ppo_<instance>.zip), --stochastic-runs y --vns-iters.
Salida: tabla comparativa impresa en consola (compare_methods +
print_comparison); código de salida 0 si el modelo existe, 1 si no.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sb3_contrib import MaskablePPO

from src.agent.evaluate import compare_methods, print_comparison
from src.data.instance_loader import load_instance


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluar agente PVRP-RL y comparar con baselines."
    )
    parser.add_argument("--instance", type=str, default="p01")
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Ruta al modelo. Por defecto: results/models/ppo_<instance>.zip",
    )
    parser.add_argument("--stochastic-runs", type=int, default=20)
    parser.add_argument("--vns-iters", type=int, default=100)
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / "data" / "raw"
    instance = load_instance(data_dir / f"{args.instance}.txt")

    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = PROJECT_ROOT / "results" / "models" / f"ppo_{args.instance}"

    if not Path(str(model_path) + ".zip").exists() and not model_path.exists():
        print(f"ERROR: no se encontró el modelo en {model_path}.zip")
        print("Primero entrena el agente con scripts/train_agent.py")
        return 1

    print(f"Cargando modelo: {model_path}")
    model = MaskablePPO.load(str(model_path))

    print(f"Evaluando sobre {instance.name} "
          f"({instance.num_customers} clientes, {instance.horizon} días)...")
    print("Esto puede tardar (VNS + muestreos estocásticos)...")

    results = compare_methods(
        model, instance, data_dir,
        n_stochastic_runs=args.stochastic_runs,
        vns_iterations=args.vns_iters,
    )

    print_comparison(results, instance.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
