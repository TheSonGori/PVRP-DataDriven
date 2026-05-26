"""
Script de experimentos de generalización (Día 13).

Dos experimentos:

  1. CERO-SHOT: carga un agente ya entrenado en una instancia y lo evalúa
     en otras instancias compatibles SIN reentrenar. Mide si el agente
     generaliza o solo memorizó la instancia de entrenamiento.

         python scripts/generalization.py zero-shot \
             --train-instance p01 --test-instances p02 p03

  2. MULTI-INSTANCIA: entrena un agente nuevo rotando entre varias
     instancias y lo evalúa en todas ellas.

         python scripts/generalization.py multi \
             --instances p01 p02 p03 --timesteps 300000 --tensorboard

Requisito: todas las instancias deben tener el mismo número de clientes
(mismo state_dim y action_dim).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sb3_contrib import MaskablePPO

from src.agent.evaluate import evaluate_deterministic, evaluate_stochastic
from src.agent.policy_config import PPOConfig
from src.agent.train import train_agent_multi
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


def run_zero_shot(args) -> int:
    """Evalúa un modelo entrenado en una instancia sobre otras."""
    model_path = (
        Path(args.model_path)
        if args.model_path
        else PROJECT_ROOT / "results" / "models" / f"ppo_{args.train_instance}"
    )
    if not Path(str(model_path) + ".zip").exists():
        print(f"ERROR: no existe el modelo {model_path}.zip")
        print("Entrena primero con: python scripts/train_agent.py "
              f"--instance {args.train_instance} --timesteps 300000")
        return 1

    print(f"Cargando agente entrenado en {args.train_instance}: {model_path}")
    model = MaskablePPO.load(str(model_path))

    test_list = args.test_instances or []
    all_instances = [args.train_instance] + test_list

    print(f"\n{'='*60}")
    print(f"  EXPERIMENTO CERO-SHOT (entrenado en {args.train_instance})")
    print(f"{'='*60}")
    print(f"  {'Instancia':<8} {'Costo':>9} {'BKS':>9} {'Gap':>8}  {'Condición':>12}")
    print(f"  {'-'*54}")
    for name in all_instances:
        inst = load_instance(DATA_DIR / f"{name}.txt")
        bks = _bks(name)
        res = evaluate_deterministic(model, inst, bks_cost=bks)
        gap = f"{res.gap_pct:+.1f}%" if res.gap_pct is not None else "N/A"
        cond = "entrenado" if name == args.train_instance else "CERO-SHOT"
        feas = "" if res.feasible else " [INFACTIBLE]"
        print(f"  {name:<8} {res.cost:>9.2f} {bks:>9.2f} {gap:>8}  {cond:>12}{feas}")
    print(f"{'='*60}\n")
    return 0


def run_multi(args) -> int:
    """Entrena un agente multi-instancia y lo evalúa en todas."""
    names = args.instances
    instances = [load_instance(DATA_DIR / f"{n}.txt") for n in names]

    config = PPOConfig(
        total_timesteps=args.timesteps,
        ent_coef=args.ent_coef,
        policy_kwargs={"net_arch": list(args.net_arch)},
        seed=args.seed,
        verbose=args.verbose,
    )

    results_dir = PROJECT_ROOT / "results"
    model_path = results_dir / "models" / f"ppo_multi_{'_'.join(names)}"
    tb_path = results_dir / "tensorboard" if args.tensorboard else None

    print(f"Entrenando agente multi-instancia sobre: {', '.join(names)}")
    print(f"({args.timesteps:,} pasos, selección {args.selection})")
    if tb_path:
        print(f"TensorBoard: tensorboard --logdir {tb_path}")

    model = train_agent_multi(
        instances,
        config=config,
        selection=args.selection,
        save_path=model_path,
        tensorboard_log=tb_path,
    )
    print(f"\nModelo guardado en: {model_path}.zip")

    print(f"\n{'='*60}")
    print(f"  EVALUACIÓN DEL AGENTE MULTI-INSTANCIA")
    print(f"{'='*60}")
    print(f"  {'Instancia':<8} {'Costo':>9} {'BKS':>9} {'Gap':>8}")
    print(f"  {'-'*40}")
    for inst in instances:
        bks = _bks(inst.name)
        res = evaluate_deterministic(model, inst, bks_cost=bks)
        gap = f"{res.gap_pct:+.1f}%" if res.gap_pct is not None else "N/A"
        feas = "" if res.feasible else " [INFACTIBLE]"
        print(f"  {inst.name:<8} {res.cost:>9.2f} {bks:>9.2f} {gap:>8}{feas}")
    print(f"{'='*60}\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Experimentos de generalización.")
    sub = parser.add_subparsers(dest="command", required=True)

    # Sub-comando zero-shot
    zs = sub.add_parser("zero-shot", help="Evaluar un modelo en instancias nuevas.")
    zs.add_argument("--train-instance", type=str, default="p01")
    zs.add_argument("--test-instances", type=str, nargs="+", default=["p02", "p03"])
    zs.add_argument("--model-path", type=str, default=None)

    # Sub-comando multi
    mi = sub.add_parser("multi", help="Entrenar agente multi-instancia.")
    mi.add_argument("--instances", type=str, nargs="+", default=["p01", "p02", "p03"])
    mi.add_argument("--timesteps", type=int, default=300_000)
    mi.add_argument("--selection", type=str, default="cyclic",
                    choices=["cyclic", "random"])
    mi.add_argument("--ent-coef", type=float, default=0.05)
    mi.add_argument("--net-arch", type=int, nargs="+", default=[256, 256])
    mi.add_argument("--seed", type=int, default=42)
    mi.add_argument("--tensorboard", action="store_true")
    mi.add_argument("--verbose", type=int, default=1)

    args = parser.parse_args()
    if args.command == "zero-shot":
        return run_zero_shot(args)
    elif args.command == "multi":
        return run_multi(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
