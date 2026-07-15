# diagnostico_p01_p03.py
from src.data.instance_loader import load_instance
from pathlib import Path

DATA_DIR = Path("data/raw")
p01 = load_instance(DATA_DIR / "p01.txt")
p03 = load_instance(DATA_DIR / "p03.txt")

print(f"p01 -> horizon={p01.horizon}, num_vehicles={p01.num_vehicles}, capacity={p01.capacity}, num_customers={p01.num_customers}")
print(f"p03 -> horizon={p03.horizon}, num_vehicles={p03.num_vehicles}, capacity={p03.capacity}, num_customers={p03.num_customers}")

print("\n--- Coordenadas de los primeros 3 clientes ---")
for i in range(3):
    c01 = p01.customers[i]
    c03 = p03.customers[i]
    print(f"  cliente {c01.id}: p01=({c01.x}, {c01.y})  vs  p03=({c03.x}, {c03.y})")

print("\n--- Demandas y frecuencias de los primeros 3 clientes ---")
for i in range(3):
    c01 = p01.customers[i]
    c03 = p03.customers[i]
    print(f"  cliente {c01.id}: p01 demanda={c01.demand} freq={c01.frequency}  vs  p03 demanda={c03.demand} freq={c03.frequency}")