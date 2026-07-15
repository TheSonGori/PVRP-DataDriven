"""
Barrido de hiperparámetros (grid 3x3 de arquitectura de red × ent_coef)
sobre p04, para verificar si la configuración adoptada ([512,512] +
ent_coef=0.05, calibrada sobre p01) sigue siendo óptima en una instancia
distinta o si es idiosincrática de p01 (uso:
`python scripts/experimento_calibracion_p04.py`; ~3 horas, 9 corridas de
500k pasos cada una).

Entrada: ninguna (usa la instancia p04 fija y el grid net_archs × ent_coefs
definido en main()).
Salida: tabla de resultados (costo, gap, factibilidad, rutas, tiempo) por
configuración, ordenada por gap, más una interpretación comparada con la
configuración adoptada; todo impreso en consola.
"""

from __future__ import annotations
import sys
import time
from itertools import product
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


# Construye el entorno con la RewardConfig por defecto.
def _build_env(instance, seed=0):
    env = PVRPEnv(instance, seed=seed)
    env = ActionMasker(env, _mask_fn)
    return env


# Replica la lógica de train_agent() sin depender de él.
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

    net_archs = [[256, 256], [512, 512], [768, 768]]
    ent_coefs = [0.01, 0.05, 0.1]

    combinaciones = list(product(net_archs, ent_coefs))
    total = len(combinaciones)

    # Minutos estimados según el tamaño de la red.
    def _estimar_min(net):
        if net == [256, 256]:
            return 15
        if net == [512, 512]:
            return 20
        return 25

    tiempo_estimado = sum(_estimar_min(n) for n, _ in combinaciones)
    print(f"\n{'='*80}")
    print("  BARRIDO DE CALIBRACIÓN SOBRE p04")
    print(f"  Instancia: p04  |  Semilla: 0  |  BKS: {bks:.2f}  |  Pasos: 500.000")
    print(f"  Grid: {len(net_archs)} redes × {len(ent_coefs)} ent_coefs = {total} corridas")
    print(f"  Tiempo estimado: ~{tiempo_estimado} minutos (~{tiempo_estimado/60:.1f} horas)")
    print(f"{'='*80}")

    resultados = []
    tiempo_total = 0.0

    for i, (net_arch, ent_coef) in enumerate(combinaciones, start=1):
        etiqueta = f"net={net_arch}, ent_coef={ent_coef}"

        ppo_config = PPOConfig(
            total_timesteps=500_000,
            ent_coef=ent_coef,
            policy_kwargs={"net_arch": list(net_arch)},
            seed=0,
            verbose=0,
        )

        print(f"\n[{i}/{total}] {etiqueta}")
        print(f"       Tiempo estimado individual: ~{_estimar_min(net_arch)} min")

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
            "net_arch": net_arch,
            "ent_coef": ent_coef,
            "cost": ev.cost,
            "gap": ev.gap_pct,
            "feasible": ev.feasible,
            "num_routes": ev.num_routes,
            "time_min": elapsed / 60,
        })

    print(f"\n{'='*80}")
    print(f"  RESUMEN DEL BARRIDO — p04, semilla 0, 500k pasos")
    print(f"  Tiempo total: {tiempo_total/60:.1f} min ({tiempo_total/3600:.2f} h)")
    print(f"{'='*80}")

    print(f"\n  Ordenado por gap (solo factibles):")
    print(f"  {'Configuración':<32} {'Costo':>10} {'Gap':>10} {'Factible':>10} {'Rutas':>7} {'Tiempo':>8}")
    print(f"  {'-'*80}")

    factibles = sorted([r for r in resultados if r["feasible"]], key=lambda r: r["gap"])
    infactibles = [r for r in resultados if not r["feasible"]]

    for r in factibles + infactibles:
        etiqueta = f"{r['net_arch']} ent_coef={r['ent_coef']}"
        feas = "sí" if r["feasible"] else "NO"
        gap = f"{r['gap']:+.1f}%" if r["gap"] is not None else "N/A"
        marker = "  <== ADOPTADA" if (
            r["net_arch"] == [512, 512] and abs(r["ent_coef"] - 0.05) < 1e-6
        ) else ""
        print(f"  {etiqueta:<32} {r['cost']:>10.2f} "
              f"{gap:>10} {feas:>10} {r['num_routes']:>7} {r['time_min']:>7.1f}m{marker}")

    print(f"\n{'-'*80}")
    print(f"  INTERPRETACIÓN")
    print(f"{'-'*80}")

    adoptada = [r for r in resultados
                if r["net_arch"] == [512, 512] and abs(r["ent_coef"] - 0.05) < 1e-6]
    if adoptada:
        adoptada = adoptada[0]
        mejor = factibles[0] if factibles else None
        if mejor:
            diff = adoptada["gap"] - mejor["gap"]
            if mejor["net_arch"] == [512, 512] and abs(mejor["ent_coef"] - 0.05) < 1e-6:
                print(f"  La configuración adoptada (512x512, ent_coef=0.05) también es")
                print(f"  la mejor en p04 (gap {adoptada['gap']:+.1f}%). Robustez confirmada.")
            else:
                print(f"  La mejor configuración en p04 es {mejor['net_arch']} "
                      f"con ent_coef={mejor['ent_coef']}")
                print(f"  con gap {mejor['gap']:+.1f}%, comparado con la adoptada "
                      f"({adoptada['gap']:+.1f}%).")
                print(f"  Diferencia: {diff:+.1f} puntos porcentuales.")
                if abs(diff) < 5:
                    print(f"  Las diferencias son menores; la configuración adoptada")
                    print(f"  es competitiva incluso si no es la óptima estricta.")
                else:
                    print(f"  Diferencia significativa. Considerar reportar como hallazgo.")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
