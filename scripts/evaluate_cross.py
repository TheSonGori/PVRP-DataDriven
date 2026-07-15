"""
Evaluación cruzada (cero-shot) y del agente multi-instancia.

Carga modelos ya entrenados y los evalúa en instancias que no fueron
las usadas para su entrenamiento (transferencia directa). También
evalúa el agente multi-instancia (ppo_multi_p01_p02_p03) sobre las
tres instancias base.

    python scripts/evaluate_cross.py
"""

from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sb3_contrib import MaskablePPO
from src.agent.train import build_env
from src.data.instance_loader import load_instance
from src.data.solution_loader import load_solution


DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "results" / "models"


def _bks(name: str):
    p = DATA_DIR / f"{name}.res"
    if p.exists():
        try:
            return load_solution(p).reported_cost
        except Exception:
            return None
    return None


def evaluate(model, instance, seed=42):
    """Ejecuta el agente sobre la instancia y devuelve costo, gap, factibilidad."""
    env = build_env(instance, seed=seed)
    obs, _ = env.reset()
    terminated = False
    steps = 0
    while not terminated and steps < 5000:
        mask = env.action_masks()
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        obs, _, terminated, _, _ = env.step(int(action))
        steps += 1
    sol = env.unwrapped.get_solution()
    cost = sol.total_cost(instance)
    feas, _ = sol.is_feasible(instance)
    bks = _bks(instance.name)
    gap = (cost - bks) / bks * 100 if bks else None
    return cost, gap, feas


def main():
    # Pares (modelo, instancia_evaluación) para cero-shot
    cross_pairs = [
        ("ppo_p01", "p01"),  # baseline: agente en su propia instancia
        ("ppo_p01", "p02"),
        ("ppo_p01", "p03"),
        ("ppo_p03_seed0", "p03"),  # baseline p03
        ("ppo_p03_seed0", "p01"),
        ("ppo_p03_seed0", "p02"),
    ]

    multi_evals = [
        ("ppo_multi_p01_p02_p03", "p01"),
        ("ppo_multi_p01_p02_p03", "p02"),
        ("ppo_multi_p01_p02_p03", "p03"),
    ]

    print(f"\n{'='*72}")
    print("  EVALUACIÓN CRUZADA (cero-shot) Y MULTI-INSTANCIA")
    print(f"{'='*72}")
    print(f"  {'Modelo':<28} {'Eval en':<8} {'Costo':>10} {'Gap':>9} {'Factible':>10}")
    print(f"  {'-'*68}")

    for model_name, inst_name in cross_pairs + multi_evals:
        model_path = MODELS_DIR / model_name
        if not (model_path.with_suffix(".zip")).exists():
            print(f"  {model_name:<28} {inst_name:<8}  MODELO NO ENCONTRADO")
            continue

        model = MaskablePPO.load(str(model_path))
        instance = load_instance(DATA_DIR / f"{inst_name}.txt")
        cost, gap, feas = evaluate(model, instance)

        gap_s = f"{gap:+.1f}%" if gap is not None else "N/A"
        feas_s = "sí" if feas else "NO"
        print(f"  {model_name:<28} {inst_name:<8} {cost:>10.2f} {gap_s:>9} {feas_s:>10}")

    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()