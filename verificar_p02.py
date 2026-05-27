# verificar_p02.py
from src.data.instance_loader import load_instance

for name in ["p01", "p02", "p03"]:
    inst = load_instance(f"data/raw/{name}.txt")
    demanda_total = sum(c.demand for c in inst.customers)
    # capacidad total disponible = capacidad por vehiculo * vehiculos * dias
    cap_total = inst.capacity * inst.num_vehicles * inst.horizon
    # demanda considerando frecuencias (un cliente puede pedir visita varios dias)
    demanda_con_freq = sum(c.demand * c.frequency for c in inst.customers)
    print(f"{name}: clientes={inst.num_customers}, horizonte={inst.horizon}, "
          f"vehiculos={inst.num_vehicles}, capacidad={inst.capacity}")
    print(f"    demanda_total={demanda_total:.0f}, demanda×freq={demanda_con_freq:.0f}, "
          f"capacidad_total={cap_total:.0f}, "
          f"saturación={demanda_con_freq/cap_total*100:.0f}%")