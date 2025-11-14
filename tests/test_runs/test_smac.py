from __future__ import annotations

import re
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


def test_smac_hyperband_termination():
    if Path("mlp_smac_hyperband").exists():
        shutil.rmtree(Path("mlp_smac_hyperband"))
    process_logs = subprocess.check_output(
        [
            "python",
            "examples/mlp.py",
            "--config-name=mlp_smac",
            "-m",
            "hydra.run.dir=mlp_smac_hyperband",
            "hydra.sweep.dir=mlp_smac_hyperband",
            "hydra.sweeper.n_trials=20",
            "+hydra.sweeper.sweeper_kwargs.max_parallelization=1",
        ]
    ).decode("utf-8")
    assert Path("mlp_smac_hyperband").exists(), "Run directory not created"

    # Goal: first launch 13 jobs in bracket 1, then 6 in bracket 2, then 3 in bracket 3, then 2 and terminate
    keyword = "Launching "
    all_keyword_indices = [m.start() for m in re.finditer(keyword, process_logs)]
    n_jobs_spawned = [int(process_logs[occ + len(keyword) : occ + len(keyword) + 2]) for occ in all_keyword_indices]
    total_jobs_spawned = sum(n_jobs_spawned)

    assert total_jobs_spawned == 20, (
        f"Total number of spawned jobs doesn't match the overall budget. Used: {total_jobs_spawned}, expected: 20."
    )
    assert n_jobs_spawned[0] == 13, (
        f"Number of spawned jobs in bracket 1 is incorrect. Used: {n_jobs_spawned[0]}, expected: 13."
    )
    assert n_jobs_spawned[1] == 6, (
        f"Number of spawned jobs in bracket 2 is incorrect. Used: {n_jobs_spawned[1]}, expected: 6."
    )
    assert n_jobs_spawned[2] == 1, (
        f"Number of spawned jobs in bracket 3 is incorrect. Used: {n_jobs_spawned[2]}, expected: 3."
    )

    shutil.rmtree(Path("mlp_smac_hyperband"))


def test_smac_hyperband_optimizer_termination():
    if Path("mlp_smac_hyperband").exists():
        shutil.rmtree(Path("mlp_smac_hyperband"))
    process_logs = subprocess.check_output(
        [
            "python",
            "examples/mlp.py",
            "--config-name=mlp_smac",
            "-m",
            "hydra.run.dir=mlp_smac_hyperband",
            "hydra.sweep.dir=mlp_smac_hyperband",
            "hydra.sweeper.n_trials=24",
            "+hydra.sweeper.sweeper_kwargs.max_parallelization=1",
        ]
    ).decode("utf-8")
    assert Path("mlp_smac_hyperband").exists(), "Run directory not created"

    # Goal: first launch 13 jobs in bracket 1, then 6 in bracket 2, then 3 in bracket 3, then 2 and terminate
    keyword = "Launching "
    all_keyword_indices = [m.start() for m in re.finditer(keyword, process_logs)]
    n_jobs_spawned = [int(process_logs[occ + len(keyword) : occ + len(keyword) + 2]) for occ in all_keyword_indices]
    total_jobs_spawned = sum(n_jobs_spawned)
    print(n_jobs_spawned)

    assert total_jobs_spawned == 23, f"Optimizer termination not happening. Used: {total_jobs_spawned}, expected: 23."
    assert n_jobs_spawned[0] == 13, (
        f"Number of spawned jobs in bracket 1 is incorrect. Used: {n_jobs_spawned[0]}, expected: 13."
    )
    assert n_jobs_spawned[1] == 6, (
        f"Number of spawned jobs in bracket 2 is incorrect. Used: {n_jobs_spawned[1]}, expected: 6."
    )
    assert n_jobs_spawned[2] == 3, (
        f"Number of spawned jobs in bracket 3 is incorrect. Used: {n_jobs_spawned[2]}, expected: 3."
    )

    shutil.rmtree(Path("mlp_smac_hyperband"))
