"""
Tests del análisis multi-semilla: verifican la mecánica de agregación
(una corrida por semilla, estadísticas media/std/min/max, tasa de
factibilidad) con un agente mínimamente entrenado; no evalúan calidad.

Entrada: la instancia p01 (data/raw/p01.txt) y, para las pruebas de
estadísticas, MultiSeedResult sintéticos construidos a mano.
Salida: aserciones pytest; no retorna valores.
"""

from __future__ import annotations

from pathlib import Path

from src.agent.multi_seed import MultiSeedResult, SeedRun, run_multi_seed
from src.agent.policy_config import SMOKE_TEST_CONFIG
from src.data.instance_loader import load_instance


DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


class TestMultiSeedAggregation:

    # run_multi_seed ejecuta exactamente una corrida por cada semilla dada.
    def test_runs_once_per_seed(self):
        instance = load_instance(DATA_DIR / "p01.txt")
        result = run_multi_seed(
            instance, seeds=[0, 1], base_config=SMOKE_TEST_CONFIG,
            bks_cost=524.61, verbose=False,
        )
        assert result.n_seeds == 2
        assert len(result.runs) == 2
        assert {r.seed for r in result.runs} == {0, 1}

    # La tasa de factibilidad agregada está en [0, 1].
    def test_feasibility_rate_in_range(self):
        instance = load_instance(DATA_DIR / "p01.txt")
        result = run_multi_seed(
            instance, seeds=[0, 1], base_config=SMOKE_TEST_CONFIG,
            bks_cost=524.61, verbose=False,
        )
        assert 0.0 <= result.feasibility_rate <= 1.0


class TestStatisticsCoherence:

    # Construye un MultiSeedResult sintético para probar las estadísticas agregadas.
    def _fake_result(self, gaps, feasibles):
        runs = [
            SeedRun(seed=i, cost=500 + g, feasible=f, num_routes=5,
                    gap_pct=g if f else None, train_time=1.0)
            for i, (g, f) in enumerate(zip(gaps, feasibles))
        ]
        return MultiSeedResult(instance_name="p01", n_seeds=len(runs),
                               bks_cost=524.61, runs=runs)

    # gap_mean/min/max/std se calculan correctamente sobre corridas factibles.
    def test_mean_std_min_max(self):
        r = self._fake_result([10.0, 20.0, 30.0], [True, True, True])
        assert abs(r.gap_mean - 20.0) < 1e-6
        assert abs(r.gap_min - 10.0) < 1e-6
        assert abs(r.gap_max - 30.0) < 1e-6
        assert r.gap_std > 0

    # Las estadísticas de gap solo consideran las corridas factibles.
    def test_only_feasible_runs_counted(self):
        r = self._fake_result([10.0, 999.0, 30.0], [True, False, True])
        assert abs(r.gap_mean - 20.0) < 1e-6
        assert r.feasibility_rate == 2 / 3

    # Sin ninguna corrida factible, gap_mean es None y feasibility_rate es 0.
    def test_no_feasible_returns_none(self):
        r = self._fake_result([10.0, 20.0], [False, False])
        assert r.gap_mean is None
        assert r.feasibility_rate == 0.0
