import shutil
import subprocess
from pathlib import Path

import pandas as pd


def test_dehb_mlp_example():
    if Path("./tmp/mlp_dehb").exists():
        shutil.rmtree(Path("./tmp/mlp_dehb"))
    subprocess.call(["python", "examples/mlp.py", "--config-name=mlp_dehb", "-m", "hydra.sweeper.n_trials=5"])
    assert Path("./tmp/mlp_dehb").exists(), "Run directory not created"
    assert Path("./tmp/mlp_dehb/runhistory.csv").exists(), "Run history file not created"
    runhistory = pd.read_csv("./tmp/mlp_dehb/runhistory.csv")
    assert not runhistory.empty, "Run history is empty"
    assert len(runhistory) == 5, "Run history should contain 5 entries"
    shutil.rmtree(Path("./tmp/mlp_dehb"))
