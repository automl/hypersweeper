from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pandas as pd


def test_grid_branin_example():
    if Path("branin_grid").exists():
        shutil.rmtree(Path("branin_grid"))
    subprocess.call(
        [
            "python",
            "examples/branin.py",
            "--config-name=branin_grid",
            "-m",
            "hydra.sweeper.n_trials=5",
            "hydra.run.dir=branin_grid",
            "hydra.sweep.dir=branin_grid",
        ]
    )
    assert Path("branin_grid").exists(), "Run directory not created"
    assert Path("branin_grid/runhistory.csv").exists(), "Run history file not created"
    runhistory = pd.read_csv("branin_grid/runhistory.csv")
    assert not runhistory.empty, "Run history is empty"
    assert len(runhistory) == 5, "Run history should contain 5 entries"
    shutil.rmtree(Path("branin_grid"))
