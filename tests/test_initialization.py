from __future__ import annotations

import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace, Float
from hydra.utils import to_absolute_path
from hydra_plugins.hypersweeper import HypersweeperSweeper
from omegaconf import DictConfig
from pytest import mark  # noqa: PT013

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


class DummyOptimizer:
    """A dummy optimizer class for testing purposes."""

    checkpoint_dir: str | None = None
    checkpoint_path_typing: str | None = None
    seeds: list = []  # noqa: RUF012


def dummy_func(cs, kwargs):  # noqa: ARG001
    """Dummy function to simulate a task function."""
    return DummyOptimizer()


GLOBAL_CONFIG = DictConfig({})
CS = ConfigurationSpace({"x0": Float("x0", bounds=(-5.0, 10.0)), "x1": Float("x1", bounds=(0.0, 15.0))})

TEST_CONFIG1 = {
    "global_config": GLOBAL_CONFIG,
    "global_overrides": ["seed=42"],
    "launcher": "this should be a launcher class",
    "make_optimizer": dummy_func,
    "budget_arg_name": "budget_alt",
    "load_arg_name": "load_alt",
    "save_arg_name": "save_alt",
    "cs": CS,
    "budget": 20,
    "optimizer_kwargs": {},
    "seeds": [],
    "seed_keyword": "seed",
    "slurm": True,
    "slurm_timeout": 10,
    "max_parallelization": 0.2,
    "job_array_size_limit": 12,
    "base_dir": None,
    "maximize": True,
    "deterministic": True,
    "checkpoint_tf": True,
    "load_tf": True,
    "checkpoint_path_typing": ".npz",
    "warmstart_file": None,
}

TEST_CONFIG2 = {
    "global_config": GLOBAL_CONFIG,
    "global_overrides": [],
    "launcher": "this should be a launcher class",
    "make_optimizer": dummy_func,
    "budget_arg_name": "budget",
    "load_arg_name": "load",
    "save_arg_name": "save",
    "cs": CS,
    "n_trials": 20,
    "optimizer_kwargs": None,
    "max_budget": 10,
    "base_dir": "some_dir",
    "min_budget": 2,
    "maximize": False,
    "deterministic": True,
    "checkpoint_tf": False,
    "load_tf": False,
    "checkpoint_path_typing": ".pt",
    "warmstart_file": None,
}


@mark.parametrize("config", [TEST_CONFIG1, TEST_CONFIG2])
def test_init(config):
    sweeper = HypersweeperSweeper(**config)
    assert sweeper is not None, "Sweeper initialization failed"
    assert isinstance(sweeper, HypersweeperSweeper), "Initialized object is not of type HypersweeperSweeper"
    assert sweeper.global_overrides == config["global_overrides"], "Global overrides do not match"
    assert sweeper.launcher == config["launcher"], "Launcher does not match"
    assert sweeper.budget_arg_name == config["budget_arg_name"], "Budget argument name does not match"
    assert sweeper.save_arg_name == config["save_arg_name"], "Save argument name does not match"
    assert sweeper.load_arg_name == config["load_arg_name"], "Load argument name does not match"
    assert sweeper.checkpoint_tf == config["checkpoint_tf"], "Checkpoint target function flag does not match"
    assert sweeper.load_tf == config["load_tf"], "Load target function flag does not match"
    assert sweeper.configspace == config["cs"], "Configuration space does not match"
    output_dir = Path(to_absolute_path(config["base_dir"]) if config["base_dir"] else to_absolute_path("./"))
    assert sweeper.output_dir == output_dir, "Output directory does not match"
    assert Path(config["base_dir"] or "./").exists(), "Base directory does not exist"
    assert sweeper.checkpoint_dir == Path(sweeper.output_dir) / "checkpoints", "Checkpoint directory does not match"
    assert sweeper.checkpoint_dir.exists(), "Checkpoint directory does not exist"
    assert sweeper.job_idx == 0, "Job index should be initialized to 0"
    assert sweeper.seeds == config.get("seeds", None), "Seeds do not match"
    assert sweeper.maximize == config["maximize"], "Maximize flag does not match"
    assert sweeper.deterministic == config["deterministic"], "Deterministic flag does not match"
    assert sweeper.slurm == config.get("slurm", False), "Slurm flag does not match"
    assert sweeper.slurm_timeout == config.get("slurm_timeout", 10), "Slurm timeout does not match"
    assert sweeper.max_parallel == min(
        config.get("job_array_size_limit", 100),
        max(1, int(config.get("max_parallelization", 0.1) * config.get("n_trials", 1_000_000))),
    ), "Max parallelization does not match"
    assert sweeper.budget == config.get("budget", 1_000_000), "Budget does not match"
    assert sweeper.min_budget == config.get("min_budget", None), "Minimum budget does not match"
    assert sweeper.max_budget == config.get("max_budget", None), "Maximum budget does not match"
    assert sweeper.n_trials == config.get("n_trials", 1_000_000), "Number of trials does not match"
    assert sweeper.trials_run == 0, "Number of trials run should be initialized to 0"
    assert sweeper.iteration == 0, "Iteration should be initialized to 0"
    assert sweeper.opt_time == 0, "Optimization time should be initialized to 0"
    assert isinstance(sweeper.history, defaultdict), "History should be a defaultdict"
    assert not sweeper.history, "Incumbents should be empty at initialization"
    assert isinstance(sweeper.incumbents, defaultdict), "Incumbents should be a defaultdict"
    assert not sweeper.incumbents, "Incumbents should be empty at initialization"
    assert sweeper.checkpoint_path_typing == config["checkpoint_path_typing"], "Checkpoint path typing does not match"
    assert isinstance(sweeper.warmstart_data, list), "Warmstart data should be a list"
    assert not sweeper.warmstart_data, "Warmstart data should be empty at initialization"
    assert sweeper.wandb_project == config.get("wandb_project", None), "WandB project does not match"
    assert isinstance(sweeper.optimizer, DummyOptimizer), "Optimizer should be an instance of DummyOptimizer"
    assert sweeper.optimizer.checkpoint_dir == sweeper.checkpoint_dir, (
        "Optimizer checkpoint directory does not match sweeper checkpoint directory"
    )
    assert sweeper.optimizer.checkpoint_path_typing == sweeper.checkpoint_path_typing, (
        "Optimizer checkpoint path type does not match sweeper checkpoint pyth type"
    )
    assert sweeper.optimizer.seeds == sweeper.seeds, "Optimizer seeds do not match sweeper seeds"
    shutil.rmtree("some_dir", ignore_errors=True)  # Clean up the base directory if it was created
    shutil.rmtree("checkpoints", ignore_errors=True)


@mark.parametrize("config", [TEST_CONFIG1, TEST_CONFIG2])
def test_init_with_warmstart(config):
    config["warmstart_file"] = Path("./examples/example_grids/branin/runhistory.csv")
    actual_warmstart_values = pd.read_csv(config["warmstart_file"])
    sweeper = HypersweeperSweeper(**config)
    assert sweeper.warmstart_data is not None, "Warmstart data should not be None"
    assert isinstance(sweeper.warmstart_data, list), "Warmstart data should be a list"
    actual_performances = actual_warmstart_values["performance"].to_list()
    if config["maximize"]:
        actual_performances = [-x for x in actual_performances]
    warmstart_performances = [res.performance for _, res in sweeper.warmstart_data]
    assert np.allclose(warmstart_performances, actual_performances), (
        "Warmstart performances do not match the expected values"
    )
    shutil.rmtree("some_dir", ignore_errors=True)  # Clean up the base directory if it was created
    shutil.rmtree("checkpoints", ignore_errors=True)


@mark.parametrize("config", [TEST_CONFIG1, TEST_CONFIG2])
def test_init_with_seeds(config):
    config["seeds"] = [42, 43, 44]
    sweeper = HypersweeperSweeper(**config)
    assert sweeper.seeds == config["seeds"], "Seeds do not match the expected values"
    assert sweeper.optimizer.seeds == config["seeds"], "Optimizer seeds do not match the expected values"
    if any("seed" in o for o in config["global_overrides"]):
        assert len(sweeper.global_overrides) == len(config["global_overrides"]) - 1, (
            f"Global overrides length does not match config-1. 'seed' was not removed: {sweeper.global_overrides}"
        )
        assert not any("seed" in o for o in sweeper.global_overrides), (
            "Global overrides should not contain 'seed' after initialization"
        )
    shutil.rmtree("some_dir", ignore_errors=True)  # Clean up the base directory if it was created
    shutil.rmtree("checkpoints", ignore_errors=True)


@mark.parametrize("config", [TEST_CONFIG1, TEST_CONFIG2])
def test_init_non_deterministic(config):
    config["deterministic"] = False
    sweeper = HypersweeperSweeper(**config)
    assert not sweeper.deterministic, "Deterministic flag should be set to False"
    if any("seed" in o for o in config["global_overrides"]):
        assert len(sweeper.global_overrides) == len(config["global_overrides"]) - 1, (
            f"Global overrides length does not match config-1. 'seed' was not removed: {sweeper.global_overrides}"
        )
        assert not any("seed" in o for o in sweeper.global_overrides), (
            "Global overrides should not contain 'seed' after initialization"
        )
    shutil.rmtree("some_dir", ignore_errors=True)  # Clean up the base directory if it was created
    shutil.rmtree("checkpoints", ignore_errors=True)


@mark.parametrize("config", [TEST_CONFIG1, TEST_CONFIG2])
def test_init_without_n_trials(config):
    config["n_trials"] = None
    sweeper = HypersweeperSweeper(**config)
    assert sweeper.n_trials is None, "Number of trials should be set to None"
    job_array_limit = config.get("job_array_size_limit", 100)
    if "seeds" in config and config["seeds"] is not None:
        job_array_limit = max(job_array_limit // len(config["seeds"]), 1)
    assert sweeper.max_parallel == job_array_limit, (
        f"Max parallelization should match job array size limit without n_trials (is {sweeper.max_parallel})"
    )
    shutil.rmtree("some_dir", ignore_errors=True)  # Clean up the base directory if it was created
    shutil.rmtree("checkpoints", ignore_errors=True)


@mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions due to wandb logging.")
@mark.parametrize("config", [TEST_CONFIG1, TEST_CONFIG2])
def test_init_with_wandb(config):
    config["wandb_project"] = "test"
    config["wandb_entity"] = "hypersweeper"
    config["wandb_tags"] = ["test", "hypersweeper"]
    config["wandb_mode"] = "offline"
    HypersweeperSweeper(**config)
    shutil.rmtree("some_dir", ignore_errors=True)  # Clean up the base directory if it was created
    shutil.rmtree("wandb", ignore_errors=True)
    shutil.rmtree("checkpoints", ignore_errors=True)
