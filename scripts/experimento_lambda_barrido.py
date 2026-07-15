"""
Experimento de barrido de infeasibility_penalty sobre p01: reduce
progresivamente la magnitud de la penalización base para identificar el
umbral en el que emerge el reward hacking (colapso a soluciones triviales),
y muestra que la penalización proporcional (lambda=-50) rescata al agente
en ese régimen (uso: `python scripts/experimento_lambda_barrido.py`; ~32 min).

Entrada: ninguna (usa la instancia p01 fija y 4 combinaciones predefinidas
de infeasibility_penalty / per_missing_penalty).
Salida: tabla comparativa (costo, gap, factibilidad, rutas) impresa en
consola para cada configuración del barrido.
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


# Construye el entorno PVRP con un reward_config custom.
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

    # (etiqueta, infeasibility_penalty, per_missing_penalty)
    configs = [
        ("baseline: -500, lambda=0",     -500.0,   0.0),
        ("penal media: -100, lambda=0",  -100.0,   0.0),
        ("penal baja: -50, lambda=0",     -50.0,   0.0),
        ("rescate: -50, lambda=-50",      -50.0, -50.0),
    ]

    base_ppo = PPOConfig(
        total_timesteps=300_000,
        ent_coef=0.05,
        policy_kwargs={"net_arch": [256, 256]},
        seed=0,
        verbose=0,
    )

    print(f"\n{'='*76}")
    print("  EXPERIMENTO: barrido de infeasibility_penalty (reward hacking)")
    print(f"  Instancia: p01  |  Semilla: 0  |  BKS: {bks:.2f}")
    print(f"{'='*76}")

    resultados = []

    for i, (label, inf_pen, per_pen) in enumerate(configs, start=1):
        reward_cfg = RewardConfig(
            terminal_bonus=100.0,
            infeasibility_penalty=inf_pen,
            per_missing_penalty=per_pen,
        )
        print(f"\n[{i}/{len(configs)}] {label}")
        print(f"       (terminal_bonus=100, infeasibility={inf_pen}, per_missing={per_pen})")

        t0 = time.time()
        model = _train_with_reward(instance, base_ppo, reward_cfg)
        elapsed = time.time() - t0

        ev = evaluate_deterministic(model, instance, bks_cost=bks)
        feas = "sí" if ev.feasible else "NO"
        gap = f"{ev.gap_pct:+.1f}%" if ev.gap_pct is not None else "N/A"

        print(f"       costo={ev.cost:.2f}  gap={gap}  factible={feas}  "
              f"rutas={ev.num_routes}  ({elapsed/60:.1f} min)")

        resultados.append({
            "label": label,
            "inf_pen": inf_pen,
            "per_pen": per_pen,
            "cost": ev.cost,
            "gap": ev.gap_pct,
            "feasible": ev.feasible,
            "num_routes": ev.num_routes,
        })

    print(f"\n{'='*76}")
    print("  RESUMEN DEL BARRIDO — instancia p01, semilla 0, 300k pasos")
    print(f"{'='*76}")
    print(f"  {'Configuración':<32} {'Costo':>10} {'Gap':>10} {'Factible':>10} {'Rutas':>7}")
    print(f"  {'-'*74}")
    for r in resultados:
        feas = "sí" if r["feasible"] else "NO"
        gap = f"{r['gap']:+.1f}%" if r["gap"] is not None else "N/A"
        print(f"  {r['label']:<32} {r['cost']:>10.2f} "
              f"{gap:>10} {feas:>10} {r['num_routes']:>7}")
    print(f"{'='*76}\n")


if __name__ == "__main__":
    main()
