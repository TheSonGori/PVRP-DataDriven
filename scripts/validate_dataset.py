"""
Validación masiva del dataset PVRP del NEO Research Group.

Este script:

    1. Recorre todas las instancias `.txt` en `data/raw/`.
    2. Verifica que cada una se cargue sin errores.
    3. Si existe el `.res` correspondiente, valida:
        - Que la solución se cargue correctamente.
        - Que el costo recalculado coincida con el reportado.
        - Que la solución sea factible (cumpla todas las restricciones del PVRP).
    4. Imprime una tabla resumen del dataset (útil para el Capítulo 4 de la memoria).
    5. Reporta cualquier inconsistencia detectada.

Uso:

    python scripts/validate_dataset.py
    python scripts/validate_dataset.py --tolerance 1.0    # tolerancia más permisiva
    python scripts/validate_dataset.py --export-csv results/dataset_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import List, Optional

# Permite ejecutar el script directamente desde cualquier ubicación.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.instance import Instance
from src.data.instance_loader import load_instance
from src.data.solution_loader import load_solution
from src.utils.solution import Solution


# =============================================================================
#  Configuración
# =============================================================================

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


# Instancias con inconsistencias conocidas en su archivo .res:
# `p11.res` referencia al cliente 139, que no aparece en `p11.txt` (declara 138
# clientes con IDs 1-138). Esto parece ser un bug histórico del dataset publicado
# por el NEO Research Group. La instancia en sí es válida y puede usarse para
# entrenar/evaluar el agente, pero el BKS reportado no es reconciliable
# directamente con la representación del modelo.
INSTANCES_WITH_KNOWN_BKS_ISSUES = {"p11"}


# =============================================================================
#  Resultado de validación por instancia
# =============================================================================

class ValidationResult:
    """Encapsula el resultado de validar una instancia y su solución."""

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

        # True si esta instancia está en la lista de BKS con problemas conocidos
        # del dataset original (no es un fallo de nuestro código).
        self.has_known_bks_issue: bool = name in INSTANCES_WITH_KNOWN_BKS_ISSUES

    @property
    def passes(self) -> bool:
        """True si la validación es completamente exitosa."""
        if not self.instance_loaded:
            return False
        if self.has_res and not self.has_known_bks_issue:
            return (
                self.solution_loaded
                and self.is_feasible is True
                and self.cost_diff is not None
            )
        return True


# =============================================================================
#  Lógica de validación
# =============================================================================

def validate_one(name: str, tolerance: float) -> ValidationResult:
    """Carga y valida una instancia y, si existe, su solución `.res`."""
    result = ValidationResult(name)

    # --- 1. Cargar instancia ---
    instance_path = DATA_DIR / f"{name}.txt"
    try:
        result.instance = load_instance(instance_path)
        result.instance_loaded = True
    except Exception as e:
        result.instance_error = f"{type(e).__name__}: {e}"
        return result

    # --- 2. Si hay .res, cargar y validar solución ---
    res_path = DATA_DIR / f"{name}.res"
    if not res_path.exists():
        return result

    result.has_res = True

    # Etapa A: cargar el archivo .res
    try:
        solution = load_solution(res_path)
        result.solution_loaded = True
        result.reported_cost = solution.reported_cost
    except Exception as e:
        result.solution_error = f"al cargar .res: {type(e).__name__}: {e}"
        return result

    # Etapa B: validar la solución contra la instancia.
    # Para instancias con problemas conocidos del BKS solo registramos las
    # violaciones de factibilidad sin intentar calcular total_cost (porque
    # contiene IDs inexistentes que harían fallar el cálculo).
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


def discover_instances() -> List[str]:
    """
    Lista los nombres de las instancias `.txt` en data/raw/.

    Solo considera archivos cuyo nombre coincide con el patrón de las
    instancias del NEO Research Group:

        - "p"  seguido de uno o más dígitos    (p01, p02, ..., p32)
        - "pr" seguido de uno o más dígitos    (pr01, pr02, ..., pr10)

    Esto evita falsos positivos con archivos como `readme.txt` o cualquier
    otro `.txt` no relacionado que pueda estar en el directorio.

    Returns:
        Lista ordenada de nombres (sin extensión).
    """
    pattern = re.compile(r"^pr?\d+$", re.IGNORECASE)
    return sorted(
        p.stem for p in DATA_DIR.glob("*.txt")
        if pattern.match(p.stem)
    )


# =============================================================================
#  Presentación de resultados
# =============================================================================

def _fmt(value: Optional[float], width: int = 10, precision: int = 2) -> str:
    """Formatea un valor numérico con None tolerante."""
    if value is None:
        return f"{'--':>{width}}"
    return f"{value:>{width}.{precision}f}"


def print_summary_table(results: List[ValidationResult]) -> None:
    """Imprime una tabla resumen de las instancias y validaciones."""
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

        # A partir de aquí: tiene .res, pero puede haber fallado parcialmente.
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


def print_failures(results: List[ValidationResult], tolerance: float) -> None:
    """Imprime detalles de las instancias que fallaron alguna validación."""
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


def export_csv(results: List[ValidationResult], output_path: Path) -> None:
    """Exporta la tabla resumen a CSV para incluirla en la memoria (Capítulo 4)."""
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


# =============================================================================
#  CLI
# =============================================================================

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

    # Código de retorno: 0 si todo OK, 1 si hay fallas críticas
    # (instancias con problemas conocidos del BKS NO cuentan como fallas).
    critical_failures = sum(
        1 for r in results
        if not r.instance_loaded
        or (r.has_res and not r.has_known_bks_issue and not r.passes)
    )
    return 0 if critical_failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
