# debug_eval.py
from sb3_contrib import MaskablePPO
from src.agent.train import build_env
from src.data.instance_loader import load_instance
from pathlib import Path

model = MaskablePPO.load("results/models/ppo_p01")
p03 = load_instance(Path("data/raw/p03.txt"))

print(f"Instancia p03: horizon={p03.horizon}, num_vehicles={p03.num_vehicles}")

env = build_env(p03, seed=42)

# Inspeccionar el entorno crudo
underlying_env = env.unwrapped
print(f"Entorno interno: horizon={underlying_env.instance.horizon}, num_vehicles={underlying_env.instance.num_vehicles}")

obs, _ = env.reset()
steps = 0
days_seen = set()

while steps < 5000:
    mask = env.action_masks()
    action, _ = model.predict(obs, action_masks=mask, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(int(action))
    steps += 1
    
    # Intentar leer el día actual del entorno
    if hasattr(underlying_env, "current_day"):
        days_seen.add(underlying_env.current_day)
    
    if terminated:
        print(f"Terminado en el paso {steps}.")
        print(f"Días vistos durante el episodio: {sorted(days_seen)}")
        break

sol = underlying_env.get_solution()
print(f"\nSolución generada:")
print(f"  Número de rutas: {len(sol.routes)}")
print(f"  Días distintos en la solución: {sorted({r.day for r in sol.routes})}")
print(f"  Costo total: {sol.total_cost(p03):.2f}")
feas, reason = sol.is_feasible(p03)
print(f"  Factible: {feas}")