# HyperSweeper

[![PyPI Version](https://img.shields.io/pypi/v/hypersweeper.svg)](https://pypi.python.org/pypi/hypersweeper)
[![Test](https://github.com/automl-private/hypersweeper/actions/workflows/pytest.yml/badge.svg)](https://github.com/automl-private/hypersweeper/actions/workflows/pytest.yml)
[![Doc Status](https://github.com/automl-private/hypersweeper/actions/workflows/docs.yml/badge.svg)](https://github.com/automl-private/hypersweeper/actions/workflows/docs.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)


Hydra sweeper integration of our favorite optimization packages, utilizing ask-and-tell interfaces.

- Free software: BSD license
- Documentation: https://automl.github.io/hypersweeper

## Installation 
We recommend installing hypersweeper in a fresh conda environment:

```bash
conda create -n hypersweeper python=3.10
make install
```

## Basic Usage

To use the sweeper, you need to specify a target function with a hydra interface (see our examples). 
Then you can add one of the Hypersweeper variations as a sweeper and run with the '-m' flag to start the optimization.
This will start a sequential run of your selected optimizer.
If you want to use Hypersweeper on a cluster, you should additionally add a launcher, e.g. the submitit launcher for slurm.

As an example, take black-box optimization for Branin using SMAC. Simply run:
```bash
python examples/branin.py -m
```
You should see the launched configurations in the terminal. 
The results are located in 'tmp', including a record of each run, the final config and a full runhistory.

## Current Sweeper Integrations
- Random Search
- SMAC
- HEBO
- PBT
- CARP-S
