from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pandas as pd


def test_hebo_mlp_example():
    if Path("mlp_hebo").exists():
        shutil.rmtree(Path("mlp_hebo"))
    subprocess.call(
        [
            "python",
            "examples/mlp.py",
            "--config-name=mlp_hebo",
            "-m",
            "hydra.sweeper.n_trials=5",
            "hydra.run.dir=mlp_hebo",
            "hydra.sweep.dir=mlp_hebo",
        ]
    )
    assert Path("mlp_hebo").exists(), "Run directory not created"
    assert Path("mlp_hebo/runhistory.csv").exists(), "Run history file not created"
    runhistory = pd.read_csv("mlp_hebo/runhistory.csv")
    assert not runhistory.empty, "Run history is empty"
    assert len(runhistory) == 5, "Run history should contain 5 entries"
    shutil.rmtree(Path("mlp_hebo"))
