import shutil
import subprocess
from pathlib import Path

import pandas as pd


def test_lpi_branin_example():
    if Path("branin_lpi").exists():
        shutil.rmtree(Path("branin_lpi"))
    subprocess.call(
        [
            "python",
            "examples/branin.py",
            "--config-name=branin_lpi",
            "-m",
            "hydra.sweeper.n_trials=5",
            "hydra.sweeper.sweeper_kwargs.optimizer_kwargs.configs_per_hp=2",
            "hydra.run.dir=branin_lpi",
            "hydra.sweep.dir=branin_lpi",
        ]
    )
    assert Path("branin_lpi").exists(), "Run directory not created"
    assert Path("branin_lpi/runhistory.csv").exists(), "Run history file not created"
    assert Path("branin_lpi/lpi_plot.png").exists(), "Plot not created"
    runhistory = pd.read_csv("branin_lpi/runhistory.csv")
    assert not runhistory.empty, "Run history is empty"
    assert len(runhistory) == 3, f"Run history should contain 3 entries, was {len(runhistory)}"
    shutil.rmtree(Path("branin_lpi"))
