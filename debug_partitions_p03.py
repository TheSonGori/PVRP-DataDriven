# debug_partitions_p03.py
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
    for r in sol.routes:
        cids = getattr(r, "customer_ids", None) or list(r.customers)
        print(f"  Día {r.day}: clientes {cids}")
    print(f"Costo total: {sol.total_cost(instance):.4f}")
    return actions_taken, sol

# Baseline: ppo_p03_seed0 evaluado en su propia instancia
acts_p03_on_p03, sol_p03_on_p03 = run_and_dump(
    "results/models/ppo_p03_seed0", "data/raw/p03.txt", "ppo_p03_seed0 sobre p03 (baseline)"
)

# Transferencia: ppo_p03_seed0 evaluado en p01
acts_p03_on_p01, sol_p03_on_p01 = run_and_dump(
    "results/models/ppo_p03_seed0", "data/raw/p01.txt", "ppo_p03_seed0 sobre p01 (transferencia)"
)

print("\n=== COMPARACIÓN ===")
print(f"¿Secuencias de acciones idénticas?: {acts_p03_on_p03 == acts_p03_on_p01}")
if acts_p03_on_p03 != acts_p03_on_p01:
    min_len = min(len(acts_p03_on_p03), len(acts_p03_on_p01))
    diverged = False
    for i in range(min_len):
        if acts_p03_on_p03[i] != acts_p03_on_p01[i]:
            print(f"Primera divergencia en paso {i}: en p03 tomó {acts_p03_on_p03[i]}, en p01 tomó {acts_p03_on_p01[i]}")
            diverged = True
            break
    if not diverged:
        print("Las secuencias coinciden en todos los pasos comunes, pero tienen largos distintos.")
    print(f"Largo en p03: {len(acts_p03_on_p03)}, largo en p01: {len(acts_p03_on_p01)}")

# Comparar también las particiones de clientes por ruta (independiente del día asignado)
def partition_as_sets(sol):
    parts = []
    for r in sol.routes:
        cids = getattr(r, "customer_ids", None) or list(r.customers)
        parts.append(frozenset(cids))
    return set(parts)

part_p03 = partition_as_sets(sol_p03_on_p03)
part_p01 = partition_as_sets(sol_p03_on_p01)
print(f"\n¿Particiones de clientes (sin importar día) idénticas?: {part_p03 == part_p01}")
print(f"Número de rutas en p03: {len(part_p03)}, en p01: {len(part_p01)}")