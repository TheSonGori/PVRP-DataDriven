# inspeccion2.py
from src.data.instance_loader import load_instance
inst = load_instance("data/raw/p01.txt")

print("=== DEPOT ===")
print("tipo:", type(inst.depot).__name__)
print("atributos:", [a for a in dir(inst.depot) if not a.startswith("_")])
print("repr:", repr(inst.depot)[:200])

print("\n=== UN CLIENTE ===")
c = inst.customers[0]
print("tipo:", type(c).__name__)
print("atributos:", [a for a in dir(c) if not a.startswith("_")])
print("repr:", repr(c)[:200])

print("\n=== UNA RUTA (nodes vs customers) ===")
from src.data.solution_loader import load_solution
sol = load_solution("data/raw/p01.res")
r = sol.routes[0]
print("r.nodes:", r.nodes)
print("r.customers:", r.customers)
print("r.day:", r.day)