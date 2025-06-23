# HyperSweeper

[![PyPI Version](https://img.shields.io/pypi/v/hypersweeper.svg)](https://pypi.python.org/pypi/hypersweeper)
[![Test](https://github.com/automl-private/hypersweeper/actions/workflows/pytest.yml/badge.svg)](https://github.com/automl-private/hypersweeper/actions/workflows/pytest.yml)
[![Doc Status](https://github.com/automl-private/hypersweeper/actions/workflows/docs.yml/badge.svg)](https://github.com/automl-private/hypersweeper/actions/workflows/docs.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)


Hydra sweeper integration of our favorite optimization packages, utilizing ask-and-tell interfaces.

- Free software: BSD license
- Documentation: https://automl.github.io/hypersweeper

## Installation 
We recommend installing hypersweeper via a uv virtual environment:

```bash
pip install uv
uv venv --python 3.10
source .venv/bin/activate
make install
```

For extra dependencies, add them like this:
```bash
uv sync --extra dev --extra carps
```

Note that CARP-S requires you to install benchmarks and optimizers on your own, e.g. by running:
```bash
python -m carps.build.make optimizer_smac
```
The full optimizer options are:
```bash
optimizer_smac optimizer_dehb optimizer_nevergrad optimizer_optuna optimizer_ax optimizer_skopt optimizer_synetune
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
For more information, see our example ReadMe.

## Current Sweeper Integrations
- Random Search
- SMAC
- HEBO
- PBT
- CARP-S (which contains many different optimizers in itself)

## Cite Us

If you use Hypersweeper in your project, please cite us:

```bibtex
@misc{eimer24,
  author    = {T. Eimer},
  title     = {Hypersweeper},
  year      = {2024},
  url = {https://github.com/automl/hypersweeper},
```
