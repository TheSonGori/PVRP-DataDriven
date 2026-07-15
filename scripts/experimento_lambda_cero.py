"""
Experimento de reward hacking: compara penalización terminal fija
(lambda=0, sin componente proporcional a las visitas faltantes) contra
penalización proporcional, sobre p01. Con penalización fija el agente
carece de gradiente para mejorar la cobertura y colapsa a políticas
triviales; construye el entorno manualmente con un reward_config custom
porque train_agent() no lo acepta como parámetro (uso:
`python scripts/experimento_lambda_cero.py`; ~18-20 min).

Entrada: ninguna (usa la instancia p01 fija y dos RewardConfig predefinidos).
Salida: tabla comparativa (costo, gap, factibilidad, rutas) impresa en
consola para ambas formulaciones de recompensa.
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
from src.agent.train import _mask_fn
from src.data.instance_loader import load_instance
from src.data.solution_loader import load_solution
from src.environment.pvrp_env import PVRPEnv
from src.environment.reward import RewardConfig


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


# Construye el entorno PVRP con un reward_config custom (replica build_env()).
def _build_env_with_reward(instance, reward_config, seed=0):
    env = PVRPEnv(instance, reward_config=reward_config, seed=seed)
    env = ActionMasker(env, _mask_fn)
    return env


# Replica train_agent() permitiendo variar el reward_config del entorno.
def _train_with_reward(instance, ppo_config, reward_config):
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

    reward_broken = RewardConfig(
        terminal_bonus=100.0,
        infeasibility_penalty=-500.0,
        per_missing_penalty=0.0,
    )

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
