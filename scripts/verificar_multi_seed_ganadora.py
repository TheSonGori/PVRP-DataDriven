"""
Verificación multi-semilla de la configuración ganadora del barrido
de calibración sobre p04.

El barrido con seed=0 identificó a [256, 256] + ent_coef=0.01 como
la configuración con mejor gap (+24.7%). Este script corre las 4
semillas restantes (1, 2, 3, 4) para determinar si esa diferencia
respecto a la configuración adoptada se sostiene bajo análisis
multi-semilla, o si es un artefacto de la seed=0.

    python scripts/verificar_multi_seed_ganadora.py

Tiempo estimado: ~60 minutos (4 entrenamientos de ~15 min cada uno).
"""

from __future__ import annotations
import sys
import time
from pathlib import Path
from statistics import mean, stdev

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


DATA_DIR = PROJECT_ROOT / "data" / "raw"


def _bks(name: str):
    p = DATA_DIR / f"{name}.res"
    if p.exists():
        try:
            return load_solution(p).reported_cost
        except Exception:
            return None
    return None


def _build_env(instance, seed=0):
    env = PVRPEnv(instance, seed=seed)
    env = ActionMasker(env, _mask_fn)
    return env


def _train_with_config(instance, ppo_config):
    env = _build_env(instance, seed=ppo_config.seed)
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
    instance = load_instance(DATA_DIR / "p04.txt")
    bks = _bks("p04")

    # Configuración a verificar: la ganadora del barrido
    net_arch = [256, 256]
    ent_coef = 0.01

    # seed=0 ya la tenemos del barrido: 1041.60, +24.7%
    seeds_a_correr = [1, 2, 3, 4]

    resultado_seed_0 = {
        "seed": 0,
        "cost": 1041.60,
        "gap": 24.7,
        "feasible": True,
        "num_routes": 10,
    }

    print(f"\n{'='*76}")
    print("  VERIFICACIÓN MULTI-SEMILLA — ganadora del barrido p04")
    print(f"  Configuración: net_arch={net_arch}, ent_coef={ent_coef}")
    print(f"  Instancia: p04  |  BKS: {bks:.2f}  |  Pasos: 500.000")
    print(f"  seeds a evaluar: {seeds_a_correr} (seed=0 ya conocida del barrido)")
    print(f"  Tiempo estimado: ~{len(seeds_a_correr) * 15} minutos")
    print(f"{'='*76}")

    resultados = [resultado_seed_0]
    tiempo_total = 0.0

    for i, seed in enumerate(seeds_a_correr, start=1):
        print(f"\n[{i}/{len(seeds_a_correr)}] Entrenando semilla {seed}...")

        ppo_config = PPOConfig(
            total_timesteps=500_000,
            ent_coef=ent_coef,
            policy_kwargs={"net_arch": list(net_arch)},
            seed=seed,
            verbose=0,
        )

        t0 = time.time()
        model = _train_with_config(instance, ppo_config)
        elapsed = time.time() - t0
        tiempo_total += elapsed

        ev = evaluate_deterministic(model, instance, bks_cost=bks)
        feas = "sí" if ev.feasible else "NO"
        gap = f"{ev.gap_pct:+.1f}%" if ev.gap_pct is not None else "N/A"

        print(f"       costo={ev.cost:.2f}  gap={gap}  factible={feas}  "
              f"rutas={ev.num_routes}  ({elapsed/60:.1f} min)")

        resultados.append({
            "seed": seed,
            "cost": ev.cost,
            "gap": ev.gap_pct,
            "feasible": ev.feasible,
            "num_routes": ev.num_routes,
        })

    # Resumen final
    print(f"\n{'='*76}")
    print(f"  RESUMEN MULTI-SEMILLA — [256,256] + ent_coef=0.01 sobre p04")
    print(f"  Tiempo total: {tiempo_total/60:.1f} min")
    print(f"{'='*76}")
    print(f"  {'Semilla':>8} {'Costo':>10} {'Gap':>9} {'Factible':>10} {'Rutas':>7}")
    print(f"  {'-'*66}")

    factibles = [r for r in resultados if r["feasible"]]

    for r in resultados:
        feas = "sí" if r["feasible"] else "NO"
        gap = f"{r['gap']:+.1f}%" if r["gap"] is not None else "N/A"
        print(f"  {r['seed']:>8} {r['cost']:>10.2f} {gap:>9} {feas:>10} {r['num_routes']:>7}")

    print(f"  {'-'*66}")

    # Estadísticas
    if len(factibles) >= 2:
        gaps = [r["gap"] for r in factibles]
        costs = [r["cost"] for r in factibles]
        gap_mean = mean(gaps)
        gap_std = stdev(gaps)
        cost_mean = mean(costs)
        cost_std = stdev(costs)
        print(f"  Tasa de factibilidad: {len(factibles)}/{len(resultados)} "
              f"({len(factibles)/len(resultados)*100:.0f}%)")
        print(f"  Gap medio:   {gap_mean:+.1f}% ± {gap_std:.1f}%")
        print(f"  Gap mínimo:  {min(gaps):+.1f}%   Gap máximo: {max(gaps):+.1f}%")
        print(f"  Costo medio: {cost_mean:.2f} ± {cost_std:.2f}")

    # Comparación con la adoptada
    print(f"\n{'-'*76}")
    print(f"  COMPARACIÓN CON LA CONFIGURACIÓN ADOPTADA")
    print(f"{'-'*76}")
    print(f"  Adoptada  ([512,512] + ent_coef=0.05):  +53.9% ± 5.4%  (5/5 factible)")
    if factibles and len(factibles) >= 2:
        gap_mean = mean([r["gap"] for r in factibles])
        gap_std = stdev([r["gap"] for r in factibles])
        print(f"  Candidata ([256,256] + ent_coef=0.01): {gap_mean:+.1f}% ± {gap_std:.1f}%  "
              f"({len(factibles)}/{len(resultados)} factible)")

        diff = 53.9 - gap_mean
        print(f"\n  Diferencia de gap medio: {diff:+.1f} puntos porcentuales.")
        if abs(diff) < 5:
            print("  Las configuraciones son ESTADÍSTICAMENTE COMPARABLES.")
            print("  La adoptada es competitiva; el barrido con seed=0 fue engañoso.")
        elif diff > 10:
            print("  La candidata es SIGNIFICATIVAMENTE MEJOR que la adoptada.")
            print("  Considerar reportar como hallazgo importante para trabajo futuro.")
        else:
            print("  Diferencia moderada. Requiere interpretación contextual.")

    print(f"\n{'='*76}\n")


if __name__ == "__main__":
    main()