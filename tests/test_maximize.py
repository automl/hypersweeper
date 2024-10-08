#!/usr/bin/env python
"""Tests for `hypersweeper` package."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def test_non_max_incumbent():
    subprocess.call(["python", "examples/branin.py", "--config-name=branin_rs", "-m"])
    assert Path("./tmp/branin_rs").exists(), "Run directory not created"
    runhistory = pd.read_csv("./tmp/branin_rs/runhistory.csv")
    with open(Path("./tmp/branin_rs/incumbent.json")) as f:
        last_line = f.readlines()[-1]
        incumbent = json.loads(last_line)
    assert np.round(incumbent["score"], decimals=3) == np.round(
        runhistory["performance"].min(), decimals=3
    ), "Incumbent is not the minimum score in the runhistory"
    shutil.rmtree("./tmp")


def test_max_incumbent():
    subprocess.call(
        ["python", "examples/branin.py", "--config-name=branin_rs", "-m", "+hydra.sweeper.sweeper_kwargs.maximize=True"]
    )
    assert Path("./tmp/branin_rs").exists(), "Run directory not created"
    runhistory = pd.read_csv("./tmp/branin_rs/runhistory.csv")
    with open(Path("./tmp/branin_rs/incumbent.json")) as f:
        last_line = f.readlines()[-1]
        incumbent = json.loads(last_line)
    print(incumbent["score"], runhistory["performance"].max(), runhistory.performance.min())
    print(runhistory.values)
    assert np.round(incumbent["score"], decimals=3) == np.round(
        runhistory["performance"].max(), decimals=3
    ), "Incumbent is not the maximum score in the runhistory even though maximize is enabled"
    shutil.rmtree("./tmp")
