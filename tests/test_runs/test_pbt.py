import shutil
import subprocess
from pathlib import Path

import pandas as pd


def test_pbt_sac_example():
    if Path("sac_pbt").exists():
        shutil.rmtree(Path("sac_pbt"))
    subprocess.call(
        [
            "python",
            "examples/sb3_rl_agent.py",
            "--config-name=sac_pbt",
            "-m",
            "algorithm.total_timesteps=1000",
            "hydra.sweeper.sweeper_kwargs.optimizer_kwargs.config_interval=500",
            "hydra.run.dir=sac_pbt",
            "hydra.sweep.dir=sac_pbt",
        ]
    )
    assert Path("sac_pbt").exists(), "Run directory not created"
    assert Path("sac_pbt/runhistory.csv").exists(), "Run history file not created"
    runhistory = pd.read_csv("sac_pbt/runhistory.csv")
    assert not runhistory.empty, "Run history is empty"
    assert len(runhistory) == 4, "Run history should contain 4 entries"
    shutil.rmtree(Path("sac_pbt"))


def test_pb2_sac_example():
    if Path("sac_pb2").exists():
        shutil.rmtree(Path("sac_pb2"))
    subprocess.call(
        [
            "python",
            "examples/sb3_rl_agent.py",
            "--config-name=sac_pb2",
            "-m",
            "algorithm.total_timesteps=1000",
            "hydra.sweeper.sweeper_kwargs.optimizer_kwargs.config_interval=500",
            "hydra.run.dir=sac_pb2",
            "hydra.sweep.dir=sac_pb2",
        ]
    )
    assert Path("sac_pb2").exists(), "Run directory not created"
    assert Path("sac_pb2/runhistory.csv").exists(), "Run history file not created"
    runhistory = pd.read_csv("sac_pb2/runhistory.csv")
    assert not runhistory.empty, "Run history is empty"
    assert len(runhistory) == 4, "Run history should contain 4 entries"
    shutil.rmtree(Path("sac_pb2"))
