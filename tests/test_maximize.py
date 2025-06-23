#!/usr/bin/env python
"""Tests for `hypersweeper` package."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from hydra_plugins.hypersweeper.utils import read_warmstart_data
from ConfigSpace import ConfigurationSpace, Float

def test_non_max_incumbent():
    if Path("./tmp/branin_rs").exists():
        Path("./tmp/branin_rs").rmdir()
    subprocess.call(["python", "examples/branin.py", "--config-name=branin_rs", "-m"])
    assert Path("./tmp/branin_rs").exists(), "Run directory not created"
    runhistory = pd.read_csv("./tmp/branin_rs/runhistory.csv")
    incumbent = pd.read_csv("./tmp/branin_rs/incumbent.csv")
    incumbent = incumbent.iloc[-1]

    assert np.round(incumbent["performance"], decimals=3) == np.round(
        runhistory["performance"].min(), decimals=3
    ), "Incumbent is not the minimum performance in the runhistory"
    Path("./tmp/branin_rs").rmdir()


def test_max_incumbent():
    if Path("./tmp/branin_rs").exists():
        Path("./tmp/branin_rs").rmdir()
    subprocess.call(
        ["python", "examples/branin.py", "--config-name=branin_rs", "-m", "+hydra.sweeper.sweeper_kwargs.maximize=True"]
    )
    assert Path("./tmp/branin_rs").exists(), "Run directory not created"
    runhistory = pd.read_csv("./tmp/branin_rs/runhistory.csv")
    incumbent = pd.read_csv("./tmp/branin_rs/incumbent.csv")
    incumbent = incumbent.iloc[-1]

    print(incumbent["performance"], runhistory["performance"].max(), runhistory.performance.min())
    print(runhistory.values)
    assert np.round(incumbent["performance"], decimals=3) == np.round(
        runhistory["performance"].max(), decimals=3
    ), "Incumbent is not the maximum score in the runhistory even though maximize is enabled"
    Path("./tmp/branin_rs").rmdir()

def test_max_warmstarting():
    example_path = Path("./examples/example_grids/branin/runhistory.csv")
    example_configspace = ConfigurationSpace({"x0": Float("x0", bounds=(-5.0, 10.0)), "x1": Float("x1", bounds=(0.0, 15.0))})
    true_data = pd.read_csv(example_path)
    performances = true_data["performance"].to_list()
    inverted_performances = [-p for p in performances]
    warmstart_data = read_warmstart_data(example_path, example_configspace, maximize=False)
    warmstart_performances = [res.performance for _, res in warmstart_data]
    inverted_warmstart_data = read_warmstart_data(example_path, example_configspace, maximize=True)
    inverted_warmstart_performances = [res.performance for _, res in inverted_warmstart_data]
    assert warmstart_data is not None, "Warmstart data is None"
    assert inverted_warmstart_data is not None, "Inverted warmstart data is None"
    assert performances == warmstart_performances, "Warmstart performances do not match performances"
    assert inverted_performances == inverted_warmstart_performances, "Inverted performances do not match inverted warmstart performances"