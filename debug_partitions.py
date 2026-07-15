# debug_partitions.py
from sb3_contrib import MaskablePPO
from src.agent.train import build_env
from src.data.instance_loader import load_instance
from pathlib import Path

def run_and_dump(model_path, instance_path, label):
    model = MaskablePPO.load(model_path)
    instance = load_instance(Path(instance_path))
    env = build_env(instance, seed=42)
    obs, _ = env.reset()
    terminated = False
    steps = 0
    actions_taken = []
    while not terminated and steps < 5000:
        mask = env.action_masks()
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        actions_taken.append(int(action))
        obs, reward, terminated, truncated, info = env.step(int(action))
        steps += 1
    sol = env.unwrapped.get_solution()
    print(f"\n=== {label} ===")
    print(f"Número de acciones: {len(actions_taken)}")
    print(f"Número de rutas: {len(sol.routes)}")

    # Diagnóstico de estructura (solo la primera vez que lo corras)
    print("Atributos de ruta de ejemplo:", vars(sol.routes[0]))

    for r in sol.routes:
        cids = getattr(r, "customer_ids", None) or list(r.customers)
        print(f"  Día {r.day}: clientes {cids}")
    print(f"Costo total: {sol.total_cost(instance):.4f}")
    return actions_taken

acts_p01 = run_and_dump("results/models/ppo_p01", "data/raw/p01.txt", "ppo_p01 sobre p01")
acts_p03 = run_and_dump("results/models/ppo_p01", "data/raw/p03.txt", "ppo_p01 sobre p03")

print("\n=== COMPARACIÓN ===")
print(f"¿Secuencias de acciones idénticas?: {acts_p01 == acts_p03}")
if acts_p01 != acts_p03:
    for i, (a, b) in enumerate(zip(acts_p01, acts_p03)):
        if a != b:
            print(f"Primera divergencia en paso {i}: p01 tomó {a}, p03 tomó {b}")
            break
    print(f"Largo p01: {len(acts_p01)}, largo p03: {len(acts_p03)}")