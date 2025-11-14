"""Tests for `hypersweeper` package."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pandas as pd


def test_logfiles():
    if Path("branin_logging").exists():
        shutil.rmtree(Path("branin_logging"))
    subprocess.check_output(
        [
            "python",
            "examples/branin.py",
            "--config-name=branin_rs",
            "-m",
            "hydra.run.dir=branin_logging",
            "hydra.sweep.dir=branin_logging",
            "hydra.sweeper.n_trials=10",
        ]
    )
    assert Path("branin_logging").exists(), "Run directory not created"

    assert Path("branin_logging/final_config.yaml").exists(), "Final config file not created"
    assert Path("branin_logging/incumbent.csv").exists(), "Incumbent file not created"
    assert Path("branin_logging/runhistory.csv").exists(), "Runhistory file not created"

    runhistory = pd.read_csv("branin_logging/runhistory.csv")
    assert len(runhistory) > 0, "Runhistory file is empty."
    assert len(runhistory) == 10, "Runhistory file does not contain expected number of trials."
    assert "x0" in runhistory.columns, "Runhistory file missing expected column 'x0'."
    assert "x1" in runhistory.columns, "Runhistory file missing expected column 'x1'."
    assert "performance" in runhistory.columns, "Runhistory file missing expected column 'performance'."
    assert "budget" in runhistory.columns, "Runhistory file missing expected column 'budget'."
    assert "config_id" in runhistory.columns, "Runhistory file missing expected column 'config_id'."

    incumbent = pd.read_csv("branin_logging/incumbent.csv")
    assert len(incumbent) > 0, "Incumbent file is empty."
    assert "x0" in incumbent.columns, "Incumbent file missing expected column 'x0'."
    assert "x1" in incumbent.columns, "Incumbent file missing expected column 'x1'."
    assert "performance" in incumbent.columns, "Incumbent file missing expected column 'performance'."
    assert "budget" in incumbent.columns, "Incumbent file missing expected column 'budget'."
    assert "config_id" in incumbent.columns, "Incumbent file missing expected column 'config_id'."
    assert "total_wallclock_time" in incumbent.columns, "Incumbent file missing expected column 'wall_clock_time'."
    assert "total_optimization_time" in incumbent.columns, (
        "Incumbent file missing expected column 'total_optimization_time'."
    )

    shutil.rmtree(Path("branin_logging"))
