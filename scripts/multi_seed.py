"""
CLI de análisis multi-semilla: entrena (o carga) el agente N veces con
semillas distintas sobre una instancia y reporta media ± desviación del gap.

    # Entrenar y guardar (uso original; ~47 min con 5 semillas y 300k pasos)
    python scripts/multi_seed.py --instance p01 --seeds 0 1 2 3 4 --save-models

    # Re-evaluar los modelos ya guardados, sin reentrenar (segundos)
    python scripts/multi_seed.py --instance p01 --load-models

    # Ablación: cuánto del gap recupera una búsqueda local sobre lo que
    # construyó el agente
    python scripts/multi_seed.py --instance p01 --load-models --local-search 2opt
    python scripts/multi_seed.py --instance p01 --load-models --local-search intra_day

Entrada: argumentos --instance, --seeds, --timesteps, --ent-coef,
--net-arch, --save-models, --load-models, --local-search.
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

from src.agent.evaluate import LOCAL_SEARCH_LEVELS
from src.agent.multi_seed import run_multi_seed, print_multi_seed
from src.agent.policy_config import PPOConfig
from src.data.instance_loader import load_instance
from src.data.solution_loader import load_solution


DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "results" / "models"


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
                        help="Guardar cada modelo entrenado en results/models/.")
    parser.add_argument("--load-models", action="store_true",
                        help="Cargar los modelos de results/models/ en vez de "
                             "entrenar. Requiere haber entrenado con --save-models.")
    parser.add_argument("--local-search", type=str, default="none",
                        choices=sorted(LOCAL_SEARCH_LEVELS),
                        help="Búsqueda local a aplicar sobre las soluciones del "
                             "agente: none, 2opt (reordenamiento intra-ruta) o "
                             "intra_day (agrega movimientos entre rutas del "
                             "mismo día).")
    args = parser.parse_args()

    if args.save_models and args.load_models:
        print("ERROR: --save-models y --load-models son mutuamente excluyentes.")
        return 1

    instance = load_instance(DATA_DIR / f"{args.instance}.txt")
    bks = _bks(args.instance)

    base_config = PPOConfig(
        total_timesteps=args.timesteps,
        ent_coef=args.ent_coef,
        policy_kwargs={"net_arch": list(args.net_arch)},
        verbose=0,
    )

    models_dir = MODELS_DIR if (args.save_models or args.load_models) else None

    print(f"Instancia: {instance.name}  |  semillas: {args.seeds}")
    if args.load_models:
        print(f"Cargando modelos desde {MODELS_DIR}  |  sin reentrenamiento")
    else:
        print(f"{args.timesteps:,} pasos c/u")
        print(f"Tiempo estimado: "
              f"~{len(args.seeds) * args.timesteps / 300_000 * 9.4:.0f} min")
    if args.local_search != "none":
        print(f"Búsqueda local sobre las soluciones del agente: {args.local_search}")

    result = run_multi_seed(
        instance,
        seeds=args.seeds,
        base_config=base_config,
        bks_cost=bks,
        models_dir=models_dir,
        load_models=args.load_models,
        local_search=args.local_search,
        verbose=True,
    )
    print_multi_seed(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
