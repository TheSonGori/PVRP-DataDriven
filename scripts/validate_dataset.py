"""
Valida masivamente el dataset PVRP del NEO Research Group: recorre todas
las instancias .txt en data/raw/, verifica que carguen sin errores y, si
existe el .res correspondiente, que la solución cargue, que su costo
recalculado coincida con el reportado y que sea factible. Imprime una
tabla resumen y, opcionalmente, la exporta a CSV.

Uso:
    python scripts/validate_dataset.py
    python scripts/validate_dataset.py --tolerance 1.0
    python scripts/validate_dataset.py --export-csv results/dataset_summary.csv

Entrada: instancias .txt/.res en data/raw/, argumentos --tolerance y
--export-csv.
Salida: tabla resumen y detalle de fallas impresos en consola; CSV opcional
en la ruta indicada; código de salida 0 si no hay fallas críticas, 1 si sí.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.instance import Instance
from src.data.instance_loader import load_instance
from src.data.solution_loader import load_solution
from src.utils.solution import Solution


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


# p11.res referencia al cliente 139, inexistente en p11.txt (declara IDs 1-138);
# es un bug conocido del dataset publicado, no de este código, así que se excluye del gap.
INSTANCES_WITH_KNOWN_BKS_ISSUES = {"p11"}


# Resultado de validar una instancia y, si existe, su solución de referencia.
class ValidationResult:

    def __init__(self, name: str):
        self.name = name
        self.instance_loaded: bool = False
        self.instance_error: Optional[str] = None
        self.instance: Optional[Instance] = None

        self.has_res: bool = False
        self.solution_loaded: bool = False
        self.solution_error: Optional[str] = None
        self.reported_cost: Optional[float] = None
        self.recalculated_cost: Optional[float] = None
        self.cost_diff: Optional[float] = None
        self.is_feasible: Optional[bool] = None
        self.num_violations: int = 0
        self.first_violations: List[str] = []

        self.has_known_bks_issue: bool = name in INSTANCES_WITH_KNOWN_BKS_ISSUES

    # True si la validación fue completamente exitosa.
    @property
    def passes(self) -> bool:
        if not self.instance_loaded:
            return False
        if self.has_res and not self.has_known_bks_issue:
            return (
                self.solution_loaded
                and self.is_feasible is True
                and self.cost_diff is not None
            )
        return True


# Carga y valida una instancia y, si existe, su solución .res.
def validate_one(name: str, tolerance: float) -> ValidationResult:
    result = ValidationResult(name)

    instance_path = DATA_DIR / f"{name}.txt"
    try:
        result.instance = load_instance(instance_path)
        result.instance_loaded = True
    except Exception as e:
        result.instance_error = f"{type(e).__name__}: {e}"
        return result

    res_path = DATA_DIR / f"{name}.res"
    if not res_path.exists():
        return result

    result.has_res = True

    try:
        solution = load_solution(res_path)
        result.solution_loaded = True
        result.reported_cost = solution.reported_cost
    except Exception as e:
        result.solution_error = f"al cargar .res: {type(e).__name__}: {e}"
        return result

    if result.has_known_bks_issue:
        feasible, violations = solution.is_feasible(result.instance)
        result.is_feasible = feasible
        result.num_violations = len(violations)
        result.first_violations = violations[:3]
        return result

    try:
        result.recalculated_cost = solution.total_cost(result.instance)
        if result.reported_cost is not None:
            result.cost_diff = abs(result.reported_cost - result.recalculated_cost)

        feasible, violations = solution.is_feasible(result.instance)
        result.is_feasible = feasible
        result.num_violations = len(violations)
        result.first_violations = violations[:3]
    except Exception as e:
        result.solution_error = f"al validar: {type(e).__name__}: {e}"

    return result


# Lista los nombres (sin extensión) de las instancias p##/pr## en data/raw/.
def discover_instances() -> List[str]:
    pattern = re.compile(r"^pr?\d+$", re.IGNORECASE)
    return sorted(
        p.stem for p in DATA_DIR.glob("*.txt")
        if pattern.match(p.stem)
    )


# Formatea un valor numérico opcional, mostrando "--" si es None.
def _fmt(value: Optional[float], width: int = 10, precision: int = 2) -> str:
    if value is None:
        return f"{'--':>{width}}"
    return f"{value:>{width}.{precision}f}"


# Imprime la tabla resumen de instancias y su estado de validación.
def print_summary_table(results: List[ValidationResult]) -> None:
    print()
    print("=" * 105)
    print(
        f"{'Instancia':<10} {'Clientes':>9} {'Días':>5} {'Vehíc.':>7} {'Capac.':>8} "
        f"{'BKS':>10} {'Recalc.':>10} {'Δ':>8} {'Fact.':>7} {'Estado':>8}"
    )
    print("-" * 105)

    for r in results:
        if not r.instance_loaded:
            print(f"{r.name:<10}  ERROR cargando instancia: {r.instance_error}")
            continue

        inst = r.instance
        base = (
            f"{r.name:<10} {inst.num_customers:>9d} {inst.horizon:>5d} "
            f"{inst.num_vehicles:>7d} {inst.capacity:>8.1f}"
        )

        if not r.has_res:
            print(f"{base} {'--':>10} {'--':>10} {'--':>8} {'--':>7} {'NO_RES':>8}")
            continue

        bks_str = _fmt(r.reported_cost)
        rec_str = _fmt(r.recalculated_cost)
        diff_str = _fmt(r.cost_diff, width=8, precision=4)

        if r.is_feasible is None:
            feasibility = "--"
        else:
            feasibility = "Sí" if r.is_feasible else "NO"

        if r.has_known_bks_issue:
            status = "BKS_ISSUE"
        elif not r.solution_loaded:
            status = "ERR_RES"
        elif r.recalculated_cost is None:
            status = "ERR_VAL"
        elif r.passes:
            status = "OK"
        else:
            status = "FAIL"

        print(
            f"{base} {bks_str} {rec_str} {diff_str} {feasibility:>7} {status:>9}"
        )

    print("=" * 105)


# Imprime el detalle de las instancias que fallaron alguna validación, agrupadas por tipo de falla.
def print_failures(results: List[ValidationResult], tolerance: float) -> None:
    instance_fails = []
    res_load_fails = []
    validation_fails = []
    infeasible = []
    warnings = []
    known_issues = []

    for r in results:
        if not r.instance_loaded:
            instance_fails.append(r)
            continue
        if not r.has_res:
            continue
        if r.has_known_bks_issue:
            known_issues.append(r)
            continue
        if not r.solution_loaded:
            res_load_fails.append(r)
            continue
        if r.recalculated_cost is None:
            validation_fails.append(r)
            continue
        if r.is_feasible is False:
            infeasible.append(r)
            continue
        if r.cost_diff is not None and r.cost_diff > tolerance:
            warnings.append(r)

    total_fails = (
        len(instance_fails) + len(res_load_fails)
        + len(validation_fails) + len(infeasible)
    )
    if total_fails == 0 and not warnings and not known_issues:
        print("\n✓ Todas las instancias y soluciones validan correctamente.\n")
        return

    if total_fails == 0 and not warnings:
        print(f"\n✓ {len(results) - len(known_issues)} instancia(s) validadas correctamente.")

    if known_issues:
        print(f"\nℹ {len(known_issues)} instancia(s) con problemas conocidos en el BKS publicado:")
        for r in known_issues:
            print(f"  - {r.name}: la solución de referencia es inconsistente con la instancia.")
            print(f"      Esta instancia es válida y usable para entrenar/evaluar, pero")
            print(f"      su BKS se excluye del análisis cuantitativo de gap.")
            if r.first_violations:
                print(f"      Ejemplo de inconsistencia: {r.first_violations[0]}")

    if instance_fails:
        print(f"\n✗ {len(instance_fails)} instancia(s) con error al cargar .txt:")
        for r in instance_fails:
            print(f"  - {r.name}: {r.instance_error}")

    if res_load_fails:
        print(f"\n✗ {len(res_load_fails)} solución(es) con error al cargar .res:")
        for r in res_load_fails:
            print(f"  - {r.name}: {r.solution_error}")

    if validation_fails:
        print(f"\n✗ {len(validation_fails)} solución(es) con error al validar contra la instancia:")
        for r in validation_fails:
            print(f"  - {r.name}: {r.solution_error}")

    if infeasible:
        print(f"\n✗ {len(infeasible)} solución(es) marcada(s) como INFACTIBLE:")
        for r in infeasible:
            print(f"  - {r.name}: {r.num_violations} violación(es)")
            for v in r.first_violations:
                print(f"      · {v}")

    if warnings:
        print(f"\n⚠ {len(warnings)} solución(es) con diferencia de costo > {tolerance}:")
        for r in warnings:
            print(f"  - {r.name}: Δ = {r.cost_diff:.4f}")
    print()


# Exporta la tabla resumen a CSV.
def export_csv(results: List[ValidationResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "instance", "num_customers", "horizon", "num_vehicles", "capacity",
            "bks_cost", "recalculated_cost", "cost_diff", "is_feasible",
            "has_res", "passes",
        ])
        for r in results:
            if not r.instance_loaded:
                continue
            inst = r.instance
            writer.writerow([
                r.name,
                inst.num_customers,
                inst.horizon,
                inst.num_vehicles,
                inst.capacity,
                r.reported_cost if r.reported_cost is not None else "",
                f"{r.recalculated_cost:.4f}" if r.recalculated_cost is not None else "",
                f"{r.cost_diff:.4f}" if r.cost_diff is not None else "",
                "Sí" if r.is_feasible else ("No" if r.is_feasible is False else ""),
                r.has_res,
                r.passes,
            ])

    print(f"✓ Tabla exportada a {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Valida masivamente el dataset PVRP del NEO Research Group."
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.5,
        help="Tolerancia absoluta en la diferencia de costo (default: 0.5).",
    )
    parser.add_argument(
        "--export-csv",
        type=Path,
        default=None,
        help="Ruta opcional donde exportar la tabla resumen como CSV.",
    )
    args = parser.parse_args()

    names = discover_instances()
    if not names:
        print(f"No se encontraron archivos .txt en {DATA_DIR}.", file=sys.stderr)
        return 1

    print(f"Validando {len(names)} instancia(s) en {DATA_DIR}...")
    results = [validate_one(name, args.tolerance) for name in names]

    print_summary_table(results)
    print_failures(results, args.tolerance)

    if args.export_csv:
        export_csv(results, args.export_csv)

    critical_failures = sum(
        1 for r in results
        if not r.instance_loaded
        or (r.has_res and not r.has_known_bks_issue and not r.passes)
    )
    return 0 if critical_failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
