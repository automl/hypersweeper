import shutil
import subprocess
from pathlib import Path

import pandas as pd


def test_rs_branin_example():
    if Path("branin_rs").exists():
        shutil.rmtree(Path("branin_rs"))
    subprocess.call(
        [
            "python",
            "examples/branin.py",
            "--config-name=branin_rs",
            "-m",
            "hydra.sweeper.n_trials=5",
            "hydra.run.dir=branin_rs",
            "hydra.sweep.dir=branin_rs",
        ]
    )
    assert Path("branin_rs").exists(), "Run directory not created"
    assert Path("branin_rs/runhistory.csv").exists(), "Run history file not created"
    runhistory = pd.read_csv("branin_rs/runhistory.csv")
    assert not runhistory.empty, "Run history is empty"
    assert len(runhistory) == 5, "Run history should contain 5 entries"
    shutil.rmtree(Path("branin_rs"))
