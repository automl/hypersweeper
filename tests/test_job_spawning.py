#!/usr/bin/env python
"""Tests for `hypersweeper` package."""

from __future__ import annotations

import re
import shutil
import subprocess
import pytest
from pathlib import Path


@pytest.mark.parametrize("max_parallel, job_limit", [(0.05, 10), (0.5, 9)])
def test_max_parallel(max_parallel, job_limit):
    if Path("branin_non_max").exists():
        shutil.rmtree(Path("branin_max_parallel"))
    process_logs = subprocess.check_output(
        [
            "python",
            "examples/branin.py",
            "--config-name=branin_rs",
            "-m",
            "hydra.run.dir=branin_max_parallel",
            "hydra.sweep.dir=branin_max_parallel",
            f"hydra.sweeper.sweeper_kwargs.max_parallelization={max_parallel}",
            f"hydra.sweeper.n_trials={job_limit}",
        ]
    ).decode("utf-8")
    assert Path("branin_max_parallel").exists(), "Run directory not created"
    keyword = "Launching "
    all_keyword_indices = [m.start() for m in re.finditer(keyword, process_logs)]
    n_jobs_spawned = [int(process_logs[occ + len(keyword)]) for occ in all_keyword_indices]
    n_times_spawned = process_logs.count(keyword) 
    total_jobs_spawned = sum(n_jobs_spawned)

    assert total_jobs_spawned == job_limit, f"Total number of spawned jobs doesn't match the overall budget. Used: {total_jobs_spawned}, expected: {job_limit}."
    for i, n_jobs in enumerate(n_jobs_spawned):
        assert n_jobs > 0, f"No jobs spawned in execution {i+1}/{n_times_spawned}."
        assert n_jobs <= max_parallel*job_limit or n_jobs == 1, f"Number of spawned jobs exceeds the maximum parallelization limit in execution {i+1}/{n_times_spawned}: {n_jobs}."
    shutil.rmtree(Path("branin_max_parallel"))

@pytest.mark.parametrize("max_parallel, job_limit, seeds", [(0.05, 10, []), (0.5, 9, [1,2])])
def test_max_parallel_with_seeds(max_parallel, job_limit, seeds):
    if Path("branin_non_max").exists():
        shutil.rmtree(Path("mlp_max_parallel_with_seeds"))
    process_logs = subprocess.check_output(
        [
            "python",
            "examples/mlp.py",
            "--config-name=mlp_smac",
            "-m",
            "hydra.run.dir=mlp_max_parallel_with_seeds",
            "hydra.sweep.dir=mlp_max_parallel_with_seeds",
            f"+hydra.sweeper.sweeper_kwargs.max_parallelization={max_parallel}",
            f"hydra.sweeper.n_trials={job_limit}",
            f"+hydra.sweeper.sweeper_kwargs.seeds={seeds}",
        ]
    ).decode("utf-8")
    assert Path("mlp_max_parallel_with_seeds").exists(), "Run directory not created"
    keyword = "Launching "
    all_keyword_indices = [m.start() for m in re.finditer(keyword, process_logs)]
    n_jobs_spawned = [int(process_logs[occ + len(keyword)]) for occ in all_keyword_indices]
    n_times_spawned = process_logs.count(keyword) 
    total_jobs_spawned = sum(n_jobs_spawned)

    assert total_jobs_spawned == job_limit*max(1, len(seeds)), f"Total number of spawned jobs doesn't match the overall budget. Used: {total_jobs_spawned}, expected: {job_limit*max(1, len(seeds))}."
    for i, n_jobs in enumerate(n_jobs_spawned):
        assert n_jobs > 0, f"No jobs spawned in execution {i+1}/{n_times_spawned}."
        assert n_jobs <= max_parallel*job_limit or n_jobs == 1, f"Number of spawned jobs exceeds the maximum parallelization limit in execution {i+1}/{n_times_spawned}: {n_jobs}."
    shutil.rmtree(Path("mlp_max_parallel_with_seeds"))