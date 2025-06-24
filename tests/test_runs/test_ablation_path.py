import shutil
import subprocess
from pathlib import Path

import pandas as pd


def test_ablation_path_branin_example():
    if Path("branin_ablation_path").exists():
        shutil.rmtree(Path("branin_ablation_path"))
    subprocess.call(
        [
            "python",
            "examples/branin.py",
            "--config-name=branin_ablation_path",
            "-m",
            "hydra.run.dir=branin_ablation_path",
            "hydra.sweep.dir=branin_ablation_path",
        ]
    )
    assert Path("branin_ablation_path").exists(), "Run directory not created"
    assert Path("branin_ablation_path/runhistory.csv").exists(), "Run history file not created"
    assert Path("branin_ablation_path/ablation_path.png").exists(), "Plot not created"
    runhistory = pd.read_csv("branin_ablation_path/runhistory.csv")
    assert not runhistory.empty, "Run history is empty"
    assert len(runhistory) == 4, f"Run history should contain 4 entries, was {len(runhistory)}"
    shutil.rmtree(Path("branin_ablation_path"))
