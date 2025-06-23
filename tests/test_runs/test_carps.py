import shutil
import subprocess
from pathlib import Path

import pandas as pd


def test_carps_smac_branin_example():
    if Path("./tmp/branin_carps").exists():
        shutil.rmtree(Path("./tmp/branin_carps_smac"))
    subprocess.call(
        ["python", "examples/branin.py", "--config-name=branin_carps_smac", "-m", "hydra.sweeper.n_trials=5"]
    )
    assert Path("./tmp/branin_carps_smac").exists(), "Run directory not created"
    assert Path("./tmp/branin_carps_smac/runhistory.csv").exists(), "Run history file not created"
    runhistory = pd.read_csv("./tmp/branin_carps_smac/runhistory.csv")
    assert not runhistory.empty, "Run history is empty"
    assert len(runhistory) == 5, "Run history should contain 5 entries"
    shutil.rmtree(Path("./tmp/branin_carps_smac"))


def test_carps_hebo_branin_example():
    if Path("./tmp/branin_carps_hebo").exists():
        shutil.rmtree(Path("./tmp/branin_carps_hebo"))
    subprocess.call(
        ["python", "examples/branin.py", "--config-name=branin_carps_hebo", "-m", "hydra.sweeper.n_trials=5"]
    )
    assert Path("./tmp/branin_carps_hebo").exists(), "Run directory not created"
    assert Path("./tmp/branin_carps_hebo/runhistory.csv").exists(), "Run history file not created"
    runhistory = pd.read_csv("./tmp/branin_carps_hebo/runhistory.csv")
    assert not runhistory.empty, "Run history is empty"
    assert len(runhistory) == 5, "Run history should contain 5 entries"
    shutil.rmtree(Path("./tmp/branin_carps_hebo"))
