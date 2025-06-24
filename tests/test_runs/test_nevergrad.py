import shutil
import subprocess
from pathlib import Path

import pandas as pd


def test_nevergrad_branin_example():
    if Path("branin_nevergrad").exists():
        shutil.rmtree(Path("branin_nevergrad"))
    subprocess.call(
        ["python", "examples/branin.py", "--config-name=branin_nevergrad", "-m", "hydra.sweeper.n_trials=5"]
    )
    assert Path("branin_nevergrad").exists(), "Run directory not created"
    assert Path("branin_nevergrad/runhistory.csv").exists(), "Run history file not created"
    runhistory = pd.read_csv("branin_nevergrad/runhistory.csv")
    assert not runhistory.empty, "Run history is empty"
    assert len(runhistory) == 5, "Run history should contain 5 entries"
    shutil.rmtree(Path("branin_nevergrad"))
