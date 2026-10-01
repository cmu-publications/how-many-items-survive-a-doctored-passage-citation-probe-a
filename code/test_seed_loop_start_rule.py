"""The seed loop never drops a reserved seed for time: a short budget only warns."""
from typing import Any, Dict, List

import main  # noqa: F401  sets the CUDA / HF-offline environment before torch is imported
import seed_loop
from main import HYPERPARAMETERS, build_config
from metrics import SELF_END, TOPICAL_END
from testing_fakes import FakeHarness

SEEDS = (0, 1, 2, 3)


class _Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def time(self) -> float:
        return self.t


def _run(monkeypatch, budget: float, seed_cost: float) -> Dict[str, Any]:
    clock = _Clock()
    recorded: List[int] = []

    def fake_run_seed(seed, design, deps, datasets, cfg, harness):
        clock.t += seed_cost
        conds = {TOPICAL_END: {"itt_citation_migration_rate": 0.1},
                 SELF_END: {"itt_citation_migration_rate": 0.2}}
        return conds, {"n_included": 1}

    def fake_record(harness, seed, conds, extra):
        recorded.append(int(seed))

    monkeypatch.setattr(seed_loop.time, "time", clock.time)
    monkeypatch.setattr(seed_loop, "run_seed", fake_run_seed)
    monkeypatch.setattr(seed_loop, "record_seed_payload", fake_record)
    run_info: Dict[str, Any] = {"t_start": clock.t, "unrun_seeds": [], "seed_failures": [],
                                "reduced_components": []}
    seed_loop.run_all_seeds(list(SEEDS), {"est_sec": 0.0}, None, {}, build_config(HYPERPARAMETERS),
                            FakeHarness(), run_info, budget)
    run_info["recorded"] = recorded
    return run_info


def test_short_budget_still_runs_every_reserved_seed(monkeypatch, capsys):
    # After seed 0, remaining 40s < 1.1 x 60s: the old rule dropped seeds 1-3 here.
    info = _run(monkeypatch, budget=100.0, seed_cost=60.0)
    assert info["recorded"] == list(SEEDS)
    assert info["unrun_seeds"] == []
    assert not any(r.get("component") == "seeds" for r in info["reduced_components"])
    warnings = info["seed_start_warnings"]
    assert [w["seed"] for w in warnings] == [1, 2, 3]
    assert all(w["started"] for w in warnings)
    assert "starting it anyway" in capsys.readouterr().out


def test_ample_budget_runs_every_seed_without_warnings(monkeypatch):
    info = _run(monkeypatch, budget=1e6, seed_cost=10.0)
    assert info["recorded"] == list(SEEDS)
    assert info["unrun_seeds"] == []
    assert info.get("seed_start_warnings", []) == []