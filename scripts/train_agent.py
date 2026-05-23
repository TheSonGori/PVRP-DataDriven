"""
Script ejecutable para entrenar el agente PVRP-RL.

Uso:

    # Entrenamiento corto de prueba (50k pasos)
    python scripts/train_agent.py --instance p01 --timesteps 50000

    # Entrenamiento completo con TensorBoard
    python scripts/train_agent.py --instance p01 --timesteps 200000 --tensorboard

    # Para visualizar el entrenamiento en tiempo real (en otra terminal):
    tensorboard --logdir results/tensorboard
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Permite ejecutar el script directamente desde cualquier ubicación.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.policy_config import PPOConfig
from src.agent.train import build_env, train_agent
from src.data.instance_loader import load_instance
from src.data.solution_loader import load_solution


def evaluate_agent(model, env, deterministic: bool = True, max_steps: int = 5000):
    """Ejecuta un episodio con el agente entrenado y retorna la solución."""
    obs, _ = env.reset()
    terminated = False
    steps = 0
    while not terminated and steps < max_steps:
        mask = env.action_masks()
        action, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)
        obs, _, terminated, truncated, _ = env.step(int(action))
        steps += 1
    return env.unwrapped.get_solution()


def main() -> int:
    parser = argparse.ArgumentParser(description="Entrenar agente PVRP-RL.")
    parser.add_argument(
        "--instance", type=str, default="p01",
        help="Nombre de la instancia de entrenamiento (ej. p01, p23).",
    )
    parser.add_argument(
        "--timesteps", type=int, default=100_000,
        help="Número total de pasos de entrenamiento.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4,
    )
    parser.add_argument(
        "--n-steps", type=int, default=2048,
        help="Pasos por rollout antes de cada actualización.",
    )
    parser.add_argument(
        "--ent-coef", type=float, default=0.05,
        help="Coeficiente de entropía (mayor = más exploración).",
    )
    parser.add_argument(
        "--net-arch", type=int, nargs="+", default=[256, 256],
        help="Arquitectura de la red (ej. --net-arch 256 256).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--tensorboard", action="store_true",
        help="Activar logs de TensorBoard en results/tensorboard.",
    )
    parser.add_argument(
        "--verbose", type=int, default=1,
        help="0 = silencioso, 1 = info estándar, 2 = debug.",
    )
    args = parser.parse_args()

    # Cargar instancia
    data_dir = PROJECT_ROOT / "data" / "raw"
    instance = load_instance(data_dir / f"{args.instance}.txt")
    print(f"Instancia: {instance}")

    # Configurar entrenamiento
    config = PPOConfig(
        total_timesteps=args.timesteps,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        ent_coef=args.ent_coef,
        policy_kwargs={"net_arch": list(args.net_arch)},
        seed=args.seed,
        verbose=args.verbose,
    )

    results_dir = PROJECT_ROOT / "results"
    model_path = results_dir / "models" / f"ppo_{args.instance}"
    tb_path = results_dir / "tensorboard" if args.tensorboard else None

    if args.tensorboard:
        print(f"TensorBoard logs en: {tb_path}")
        print(f"Para visualizar: tensorboard --logdir {tb_path}")

    print(f"\nIniciando entrenamiento ({args.timesteps:,} pasos)...")
    start = time.time()
    model = train_agent(
        instance,
        config=config,
        save_path=model_path,
        tensorboard_log=tb_path,
    )
    elapsed = time.time() - start
    print(f"\nEntrenamiento completado en {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Modelo guardado en: {model_path}.zip")

    # Evaluación post-entrenamiento
    print("\nEvaluando agente entrenado...")
    env = build_env(instance, seed=args.seed)
    sol = evaluate_agent(model, env, deterministic=True)
    cost = sol.total_cost(instance)
    feasible, _ = sol.is_feasible(instance)

    print(f"  Costo:       {cost:.2f}")
    print(f"  Rutas:       {len(sol.routes)}")
    print(f"  Factible:    {feasible}")

    # Comparar con BKS si existe
    bks_path = data_dir / f"{args.instance}.res"
    if bks_path.exists():
        bks = load_solution(bks_path)
        gap = (cost - bks.reported_cost) / bks.reported_cost * 100
        print(f"  BKS:         {bks.reported_cost:.2f}")
        print(f"  Gap vs BKS:  {gap:+.1f}%")

    return 0 if feasible else 1


if __name__ == "__main__":
    sys.exit(main())
