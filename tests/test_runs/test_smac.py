from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pandas as pd


def test_smac_branin_example():
    if Path("branin_smac").exists():
        shutil.rmtree(Path("branin_smac"))
    subprocess.call(
        [
            "python",
            "examples/branin.py",
            "--config-name=branin_smac",
            "-m",
            "hydra.sweeper.n_trials=5",
            "hydra.run.dir=branin_smac",
            "hydra.sweep.dir=branin_smac",
        ]
    )
    assert Path("branin_smac").exists(), "Run directory not created"
    assert Path("branin_smac/runhistory.csv").exists(), "Run history file not created"
    runhistory = pd.read_csv("branin_smac/runhistory.csv")
    assert not runhistory.empty, "Run history is empty"
    assert len(runhistory) == 5, "Run history should contain 5 entries"
    shutil.rmtree(Path("branin_smac"))
