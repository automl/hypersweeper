#!/usr/bin/env python
"""Tests for `hypersweeper` package."""

from __future__ import annotations

import re
import shutil
import subprocess
import pytest
from pathlib import Path


@pytest.mark.parametrize("n_trials", [1, 7, 10])
def test_terminate_n_trials(n_trials):
    if Path("branin_trial_termination").exists():
        shutil.rmtree(Path("branin_trial_termination"))
    process_logs = subprocess.check_output(
        [
            "python",
            "examples/branin.py",
            "--config-name=branin_rs",
            "-m",
            "hydra.run.dir=branin_trial_termination",
            "hydra.sweep.dir=branin_trial_termination",
            f"hydra.sweeper.n_trials={n_trials}",
        ]
    ).decode("utf-8")
    assert Path("branin_trial_termination").exists(), "Run directory not created"
    keyword = "Launching "
    all_keyword_indices = [m.start() for m in re.finditer(keyword, process_logs)]
    n_jobs_spawned = [int(process_logs[occ + len(keyword)]) for occ in all_keyword_indices] 
    total_jobs_spawned = sum(n_jobs_spawned)

    assert total_jobs_spawned == n_trials, f"Total number of spawned jobs doesn't match the n_trials. Used: {total_jobs_spawned}, expected: {n_trials}."
    shutil.rmtree(Path("branin_trial_termination"))

@pytest.mark.parametrize("budget", [5, 10, 25])
def test_terminate_budget(budget):
    if Path("mlp_budget_termination").exists():
        shutil.rmtree(Path("mlp_budget_termination"))
    process_logs = subprocess.check_output(
        [
            "python",
            "examples/branin.py",
            "--config-name=branin_rs",
            "-m",
            "hydra.run.dir=mlp_budget_termination",
            "hydra.sweep.dir=mlp_budget_termination",
            f"hydra.sweeper.sweeper_kwargs.max_budget={budget}",
        ]
    ).decode("utf-8")
    assert Path("mlp_budget_termination").exists(), "Run directory not created"
    keyword = "epochs="
    all_keyword_indices = [m.start() for m in re.finditer(keyword, process_logs)]
    all_budgets = [int(process_logs[occ + len(keyword)]) for occ in all_keyword_indices] 
    assert sum(all_budgets) <= budget, f"Total budget used exceeds the maximum budget. Used: {sum(all_budgets)}, budget: {budget}."
    shutil.rmtree(Path("mlp_budget_termination"))