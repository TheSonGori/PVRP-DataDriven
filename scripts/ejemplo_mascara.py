"""
Extrae un ejemplo real de máscara de acciones de una corrida del agente
ppo_p01 sobre la instancia p01: avanza el episodio unos pasos hasta un
estado con clientes ya visitados y capacidad parcialmente consumida, y
detalla el estado y la máscara resultante (uso:
`python scripts/ejemplo_mascara.py`).

Entrada: ninguna (usa la instancia p01 y el modelo results/models/ppo_p01
fijos).
Salida: detalle del estado, la máscara de acciones y un resumen impresos
en consola.
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


DATA_DIR = PROJECT_ROOT / "data" / "raw"


def main():
    instance = load_instance(DATA_DIR / "p01.txt")
    model = MaskablePPO.load("results/models/ppo_p01")

    env = build_env(instance, seed=42)
    obs, _ = env.reset()
    underlying = env.unwrapped

    target_step = 6

    print(f"\n{'='*76}")
    print(f"  EJEMPLO REAL DE ENMASCARAMIENTO — instancia p01, seed=42")
    print(f"{'='*76}\n")

    for step in range(1, target_step + 1):
        mask = env.action_masks()
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        print(f"Paso {step}: acción tomada = {int(action)}"
              f"  (día={underlying._state.current_day},"
              f" vehículo={underlying._state.current_vehicle},"
              f" capacidad residual={underlying._state.remaining_capacity:.0f})")

    print(f"\n{'-'*76}")
    print(f"  ESTADO EN EL PASO {target_step + 1}")
    print(f"{'-'*76}")
    state = underlying._state
    print(f"  Día actual:                    {state.current_day}")
    print(f"  Vehículo activo:               {state.current_vehicle}")
    print(f"  Capacidad residual del vehículo: {state.remaining_capacity:.1f} / {instance.capacity:.0f}")
    print(f"  Posición actual (nodo):        {state.current_position}")
    print(f"  Clientes ya visitados hoy:     "
          f"{sorted([c for c in range(1, instance.num_customers + 1) if state.current_day in state.visits_completed.get(c, [])])[:10]}"
          f"...")

    mask = env.action_masks()

    print(f"\n{'-'*76}")
    print(f"  MÁSCARA DE ACCIONES EN ESTE PASO")
    print(f"{'-'*76}")
    print(f"  Acción | Válida | Descripción")
    print(f"  {'-'*70}")

    valid = "sí" if mask[0] else "NO"
    print(f"    0    |   {valid:<3}  | Cerrar ruta (retornar al depósito)")

    for c_id in range(1, instance.num_customers + 1):
        valid = "sí" if mask[c_id] else "NO"
        customer = instance.get_customer(c_id)
        already_visited_today = state.current_day in state.visits_completed.get(c_id, [])
        exceeds_capacity = customer.demand > state.remaining_capacity
        pattern_done = len(state.visits_completed.get(c_id, [])) >= customer.frequency

        motivo = "cliente disponible"
        if not mask[c_id]:
            reasons = []
            if already_visited_today:
                reasons.append("ya visitado hoy")
            if exceeds_capacity:
                reasons.append(f"demanda {customer.demand:.0f} > cap residual {state.remaining_capacity:.0f}")
            if pattern_done:
                reasons.append(f"cuota cumplida ({customer.frequency} visitas)")
            motivo = ", ".join(reasons) if reasons else "desconocido"

        if c_id <= 10 or not mask[c_id]:
            print(f"    {c_id:<4} |   {valid:<3}  | {motivo}")

    print(f"    ...  |  ...  | (resto de clientes con estado similar)")
    print(f"\n{'-'*76}")
    print(f"  RESUMEN")
    print(f"{'-'*76}")
    num_valid = int(sum(mask))
    num_invalid = int(len(mask) - num_valid)
    print(f"  Total de acciones válidas:   {num_valid} de {len(mask)}")
    print(f"  Total de acciones bloqueadas: {num_invalid}")

    print(f"\n{'='*76}")
    print(f"  SUGERENCIA PARA FIGURA DE LA MEMORIA")
    print(f"{'='*76}")
    print(f"  Elegir los primeros 5 clientes + acción 0 como ejemplo:")
    print(f"  Acción 0: {'válida' if mask[0] else 'BLOQUEADA'} — cerrar ruta")
    for c_id in range(1, 6):
        v = "válida" if mask[c_id] else "BLOQUEADA"
        d = instance.get_customer(c_id).demand
        print(f"  Acción {c_id}: {v:>9} — cliente {c_id} (demanda {d:.0f})")
    print()


if __name__ == "__main__":
    main()
