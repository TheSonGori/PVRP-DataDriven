"""
Experimento comparativo: penalización fija (lambda=0) vs proporcional.

Demuestra el fenómeno de reward hacking cuando la penalización terminal
es de magnitud fija ante cualquier infactibilidad, independientemente
del número de clientes no atendidos. Bajo esa formulación, el agente
carece de gradiente para mejorar la cobertura y colapsa a políticas
triviales.

Construye el entorno manualmente con reward_config custom, ya que
train_agent() no lo acepta como parámetro. La lógica de entrenamiento
replica la de train_agent() pero permite variar el reward_config.

    python scripts/experimento_lambda_cero.py

Tiempo estimado: ~18-20 minutos (dos entrenamientos de ~9 min cada uno).
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from src.agent.evaluate import evaluate_deterministic
from src.agent.policy_config import PPOConfig
from src.agent.train import _mask_fn  # helper interno del proyecto
from src.data.instance_loader import load_instance
from src.data.solution_loader import load_solution
from src.environment.pvrp_env import PVRPEnv
from src.environment.reward import RewardConfig


DATA_DIR = PROJECT_ROOT / "data" / "raw"


def _bks(name: str):
    p = DATA_DIR / f"{name}.res"
    if p.exists():
        try:
            return load_solution(p).reported_cost
        except Exception:
            return None
    return None


def _build_env_with_reward(instance, reward_config, seed=0):
    """Construye el entorno PVRP con un reward_config custom, replicando build_env()."""
    env = PVRPEnv(instance, reward_config=reward_config, seed=seed)
    env = ActionMasker(env, _mask_fn)
    return env


def _train_with_reward(instance, ppo_config, reward_config):
    """
    Replica la lógica esencial de train_agent() pero permite variar el
    reward_config del entorno.
    """
    env = _build_env_with_reward(instance, reward_config, seed=ppo_config.seed)

    model = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=ppo_config.learning_rate,
        n_steps=ppo_config.n_steps,
        batch_size=ppo_config.batch_size,
        n_epochs=ppo_config.n_epochs,
        gamma=ppo_config.gamma,
        gae_lambda=ppo_config.gae_lambda,
        clip_range=ppo_config.clip_range,
        ent_coef=ppo_config.ent_coef,
        policy_kwargs=ppo_config.policy_kwargs,
        seed=ppo_config.seed,
        verbose=ppo_config.verbose,
    )

    model.learn(total_timesteps=ppo_config.total_timesteps)
    return model


def main():
    instance = load_instance(DATA_DIR / "p01.txt")
    bks = _bks("p01")

    # Formulación rota: penalización fija ante infactibilidad
    reward_broken = RewardConfig(
        terminal_bonus=100.0,
        infeasibility_penalty=-500.0,
        per_missing_penalty=0.0,  # <-- la clave del reward hacking
    )

    # Formulación correcta: penalización proporcional (default de la memoria)
    reward_fixed = RewardConfig(
        terminal_bonus=100.0,
        infeasibility_penalty=-500.0,
        per_missing_penalty=-50.0,
    )

    base_config = PPOConfig(
        total_timesteps=300_000,
        ent_coef=0.05,
        policy_kwargs={"net_arch": [256, 256]},
        seed=0,
        verbose=0,
    )

    print(f"\n{'='*72}")
    print("  EXPERIMENTO: reward hacking (lambda=0 vs proporcional)")
    print(f"  Instancia: p01  |  Semilla: 0  |  BKS: {bks:.2f}")
    print(f"{'='*72}")

    # --- Corrida 1: formulación rota ---
    print("\n[1/2] Entrenando con penalizacion FIJA (lambda = 0)...")
    t0 = time.time()
    model_broken = _train_with_reward(instance, base_config, reward_broken)
    t_broken = time.time() - t0

    ev_broken = evaluate_deterministic(model_broken, instance, bks_cost=bks)
    feas_broken = "sí" if ev_broken.feasible else "NO"
    gap_broken = f"{ev_broken.gap_pct:+.1f}%" if ev_broken.gap_pct is not None else "N/A"
    print(f"    costo={ev_broken.cost:.2f}  gap={gap_broken}  "
          f"factible={feas_broken}  rutas={ev_broken.num_routes}  "
          f"({t_broken/60:.1f} min)")

    # --- Corrida 2: formulación correcta ---
    print("\n[2/2] Entrenando con penalizacion PROPORCIONAL (lambda = -50)...")
    t0 = time.time()
    model_fixed = _train_with_reward(instance, base_config, reward_fixed)
    t_fixed = time.time() - t0

    ev_fixed = evaluate_deterministic(model_fixed, instance, bks_cost=bks)
    feas_fixed = "sí" if ev_fixed.feasible else "NO"
    gap_fixed = f"{ev_fixed.gap_pct:+.1f}%" if ev_fixed.gap_pct is not None else "N/A"
    print(f"    costo={ev_fixed.cost:.2f}  gap={gap_fixed}  "
          f"factible={feas_fixed}  rutas={ev_fixed.num_routes}  "
          f"({t_fixed/60:.1f} min)")

    # --- Resumen ---
    print(f"\n{'='*72}")
    print("  RESUMEN COMPARATIVO — instancia p01, semilla 0")
    print(f"{'='*72}")
    print(f"  {'Formulación':<32} {'Costo':>10} {'Gap':>10} {'Factible':>10} {'Rutas':>7}")
    print(f"  {'-'*70}")
    print(f"  {'lambda = 0 (fija)':<32} {ev_broken.cost:>10.2f} "
          f"{gap_broken:>10} {feas_broken:>10} {ev_broken.num_routes:>7}")
    print(f"  {'lambda proporcional (adoptada)':<32} {ev_fixed.cost:>10.2f} "
          f"{gap_fixed:>10} {feas_fixed:>10} {ev_fixed.num_routes:>7}")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()