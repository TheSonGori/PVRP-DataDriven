"""
CLI de análisis multi-semilla: entrena el agente N veces con semillas
distintas sobre una instancia y reporta media ± desviación del gap (uso:
`python scripts/multi_seed.py --instance p01 --seeds 0 1 2 3 4 --timesteps
300000 --save-models`; con 5 semillas y 300k pasos toma ~47 min en CPU).

Entrada: argumentos --instance, --seeds, --timesteps, --ent-coef,
--net-arch, --save-models.
Salida: resumen impreso en consola (run_multi_seed + print_multi_seed); si
--save-models, guarda cada modelo en results/models/. Código de salida 0.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.multi_seed import run_multi_seed, print_multi_seed
from src.agent.policy_config import PPOConfig
from src.data.instance_loader import load_instance
from src.data.solution_loader import load_solution


DATA_DIR = PROJECT_ROOT / "data" / "raw"


# Costo de la BKS de una instancia, o None si no hay .res disponible.
def _bks(name: str):
    p = DATA_DIR / f"{name}.res"
    if p.exists():
        try:
            return load_solution(p).reported_cost
        except Exception:
            return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Análisis multi-semilla.")
    parser.add_argument("--instance", type=str, default="p01")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--ent-coef", type=float, default=0.05)
    parser.add_argument("--net-arch", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--save-models", action="store_true",
                        help="Guardar cada modelo en results/models/.")
    args = parser.parse_args()

    instance = load_instance(DATA_DIR / f"{args.instance}.txt")
    bks = _bks(args.instance)

    base_config = PPOConfig(
        total_timesteps=args.timesteps,
        ent_coef=args.ent_coef,
        policy_kwargs={"net_arch": list(args.net_arch)},
        verbose=0,
    )

    models_dir = (PROJECT_ROOT / "results" / "models") if args.save_models else None

    print(f"Instancia: {instance.name}  |  semillas: {args.seeds}  "
          f"|  {args.timesteps:,} pasos c/u")
    print(f"Tiempo estimado: ~{len(args.seeds) * args.timesteps / 300_000 * 9.4:.0f} min")

    result = run_multi_seed(
        instance,
        seeds=args.seeds,
        base_config=base_config,
        bks_cost=bks,
        models_dir=models_dir,
        verbose=True,
    )
    print_multi_seed(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
