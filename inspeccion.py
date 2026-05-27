# inspeccion.py 
from src.data.instance_loader import load_instance
from src.data.solution_loader import load_solution

inst = load_instance("data/raw/p01.txt")
sol = load_solution("data/raw/p01.res")

print("=== INSTANCE ===")
print("atributos:", [a for a in dir(inst) if not a.startswith("_")])
print()
print("=== SOLUTION ===")
print("atributos:", [a for a in dir(sol) if not a.startswith("_")])
print()
print("=== UNA RUTA ===")
r = sol.routes[0]
print("tipo:", type(r).__name__)
print("atributos:", [a for a in dir(r) if not a.startswith("_")])
print("repr:", repr(r)[:200])