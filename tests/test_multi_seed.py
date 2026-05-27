"""
Tests del análisis multi-semilla (Día 14).

Verifican la mecánica de agregación con un agente mínimamente entrenado
(SMOKE_TEST_CONFIG) y pocas semillas. No verifican calidad, solo que:
    - Se ejecuta una corrida por semilla.
    - Las estadísticas (media, std, min, max) son coherentes.
    - La tasa de factibilidad está en [0, 1].
"""

from __future__ import annotations

from pathlib import Path

from src.agent.multi_seed import MultiSeedResult, SeedRun, run_multi_seed
from src.agent.policy_config import SMOKE_TEST_CONFIG
from src.data.instance_loader import load_instance


DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


class TestMultiSeedAggregation:
    def test_runs_once_per_seed(self):
        instance = load_instance(DATA_DIR / "p01.txt")
        result = run_multi_seed(
            instance, seeds=[0, 1], base_config=SMOKE_TEST_CONFIG,
            bks_cost=524.61, verbose=False,
        )
        assert result.n_seeds == 2
        assert len(result.runs) == 2
        assert {r.seed for r in result.runs} == {0, 1}

    def test_feasibility_rate_in_range(self):
        instance = load_instance(DATA_DIR / "p01.txt")
        result = run_multi_seed(
            instance, seeds=[0, 1], base_config=SMOKE_TEST_CONFIG,
            bks_cost=524.61, verbose=False,
        )
        assert 0.0 <= result.feasibility_rate <= 1.0


class TestStatisticsCoherence:
    def _fake_result(self, gaps, feasibles):
        """Construye un MultiSeedResult sintético para probar estadísticas."""
        runs = [
            SeedRun(seed=i, cost=500 + g, feasible=f, num_routes=5,
                    gap_pct=g if f else None, train_time=1.0)
            for i, (g, f) in enumerate(zip(gaps, feasibles))
        ]
        return MultiSeedResult(instance_name="p01", n_seeds=len(runs),
                               bks_cost=524.61, runs=runs)

    def test_mean_std_min_max(self):
        r = self._fake_result([10.0, 20.0, 30.0], [True, True, True])
        assert abs(r.gap_mean - 20.0) < 1e-6
        assert abs(r.gap_min - 10.0) < 1e-6
        assert abs(r.gap_max - 30.0) < 1e-6
        assert r.gap_std > 0

    def test_only_feasible_runs_counted(self):
        # 2 factibles (gaps 10, 30) y 1 infactible -> media sobre factibles = 20
        r = self._fake_result([10.0, 999.0, 30.0], [True, False, True])
        assert abs(r.gap_mean - 20.0) < 1e-6
        assert r.feasibility_rate == 2 / 3

    def test_no_feasible_returns_none(self):
        r = self._fake_result([10.0, 20.0], [False, False])
        assert r.gap_mean is None
        assert r.feasibility_rate == 0.0