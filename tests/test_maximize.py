#!/usr/bin/env python
"""Tests for `hypersweeper` package."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def test_non_max_incumbent():
    subprocess.call(["python", "examples/branin.py", "--config-name=branin_rs", "-m"])
    assert Path("./tmp/branin_rs").exists(), "Run directory not created"
    runhistory = pd.read_csv("./tmp/branin_rs/runhistory.csv")
    incumbent = pd.read_csv("./tmp/branin_rs/incumbent.csv")
    incumbent = incumbent.iloc[-1]

    assert np.round(incumbent["performance"], decimals=3) == np.round(
        runhistory["performance"].min(), decimals=3
    ), "Incumbent is not the minimum performance in the runhistory"
    shutil.rmtree("./tmp")


def test_max_incumbent():
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
    shutil.rmtree("./tmp")
