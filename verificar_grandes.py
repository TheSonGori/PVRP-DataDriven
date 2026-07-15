# verificar_grandes.py
from src.data.instance_loader import load_instance

instancias = ["p01", "p02", "p03",     # las que ya conocemos (referencia)
              "p04", "p05", "p06",     # 75 clientes
              "p07", "p08", "p09"]     # 100 clientes

print(f"{'inst':>5} {'N':>4} {'T':>3} {'K':>3} {'Q':>5} "
      f"{'dem×freq':>10} {'cap_tot':>9} {'sat':>5}")
print("-" * 60)
for name in instancias:
    inst = load_instance(f"data/raw/{name}.txt")
    demanda_freq = sum(c.demand * c.frequency for c in inst.customers)
    cap_total = inst.capacity * inst.num_vehicles * inst.horizon
    sat = demanda_freq / cap_total * 100
    print(f"{name:>5} {inst.num_customers:>4} {inst.horizon:>3} "
          f"{inst.num_vehicles:>3} {inst.capacity:>5.0f} "
          f"{demanda_freq:>10.0f} {cap_total:>9.0f} {sat:>4.0f}%")