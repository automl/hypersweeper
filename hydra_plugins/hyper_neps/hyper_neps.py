"""Hypersweeper Interface for NEPS."""

from __future__ import annotations

import importlib
import math
import random
import time
from pathlib import Path

import numpy as np
from neps.runtime import Trial

if (spec := importlib.util.find_spec("neps")) is not None:
    import neps
    import neps.search_spaces

from typing import TYPE_CHECKING

from ConfigSpace.hyperparameters import (CategoricalHyperparameter,
                                         NormalFloatHyperparameter,
                                         NormalIntegerHyperparameter,
                                         UniformFloatHyperparameter,
                                         UniformIntegerHyperparameter)

from hydra_plugins.hypersweeper import Info

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace
    from neps.optimizers.base_optimizer import BaseOptimizer


class HyperNEPS:
    """NEPS."""

    def __init__(self, configspace: ConfigurationSpace, optimizer: BaseOptimizer, fidelity_variable: str) -> None:
        """Initialize the optimizer."""
        self.configspace = configspace
        self.optimizer = optimizer
        self.previous_results = {}
        self.pending_evaluations = {}
        self.fidelity_variable = fidelity_variable

    def ask(self) -> tuple[Info, bool]:
        """Randomly sample a configuration."""
        self.optimizer.load_results(
            previous_results={
                config_id: report.to_config_result(self.optimizer.load_config)
                for config_id, report in self.previous_results.items()
            },
            pending_evaluations={
                config_id: self.optimizer.load_config(trial.config)
                for config_id, trial in self.pending_evaluations.items()
            },
        )

        config, config_id, prev_config_id = self.optimizer.get_config_and_ids()
        previous = None
        if prev_config_id is not None:
            previous = self.previous_results[prev_config_id]

        time_sampled = time.time()

        trial = Trial(
            id=config_id,
            config=config,
            report=None,
            time_sampled=time_sampled,
            pipeline_dir=Path(),  # TODO
            previous=previous,
            metadata={"time_sampled": time_sampled},
        )

        self.pending_evaluations[config_id] = trial

        config = dict(config)
        budget = config.pop(self.fidelity_variable)

        info = Info(dict(config), budget, config_id, None, None)
        return info, False

    def tell(self, info: Info, value):
        """Return the performance."""
        trial = self.pending_evaluations.pop(info.config_id)

        trial.report = trial.create_success_report(value.performance)

        self.previous_results[info.config_id] = trial


def make_neps(configspace, hyper_neps_args):
    """Make a NEPS instance for optimization."""
    # important for NePS optimizers
    random.seed(hyper_neps_args["seed"])

    np.random.seed(hyper_neps_args["seed"])  # noqa: NPY002

    dict_search_space = get_dict_from_configspace(configspace)

    dict_search_space[hyper_neps_args["fidelity_variable"]] = neps.FloatParameter(
        lower=hyper_neps_args["min_budget"], upper=hyper_neps_args["max_budget"], is_fidelity=True
    )

    neps_search_space = neps.search_spaces.SearchSpace(**dict_search_space)

    optimizer = hyper_neps_args["optimizer"](
        pipeline_space=neps_search_space,
    )

    print("Budget levels for NEPS:")
    check_budget_levels(hyper_neps_args["min_budget"], hyper_neps_args["max_budget"], optimizer.eta)

    return HyperNEPS(
        configspace=configspace, optimizer=optimizer, fidelity_variable=hyper_neps_args["fidelity_variable"]
    )


def get_dict_from_configspace(configspace: ConfigurationSpace) -> dict:
    """Get a dictionary containing NEPS hyperparameters from a ConfigSpace object."""
    search_space = {}
    for k in configspace:
        param = configspace[k]
        if isinstance(param, NormalFloatHyperparameter | UniformFloatHyperparameter):
            search_space[k] = neps.FloatParameter(
                lower=param.lower,
                upper=param.upper,
                log=param.log,
                default=param.default_value,
                default_confidence="medium",
            )
        elif isinstance(param, NormalIntegerHyperparameter | UniformIntegerHyperparameter):
            search_space[k] = neps.IntegerParameter(
                lower=param.lower,
                upper=param.upper,
                log=param.log,
                default=param.default_value,
                default_confidence="medium",
            )
        elif isinstance(param, CategoricalHyperparameter):
            search_space[k] = neps.CategoricalParameter(
                choices=param.choices, default=param.default_value, default_confidence="medium"
            )
    return search_space


def check_budget_levels(min_epoch, max_epoch, eta):
    """Check the Hyperband budget levels for NEPS."""
    total_budget = 0
    _min = max_epoch
    counter = 0
    fid_level = math.ceil(math.log(max_epoch / min_epoch) / math.log(eta))
    while _min >= min_epoch:
        print(f"Level: {fid_level} -> {_min}")
        total_budget += _min * eta
        _min = _min // eta
        counter += 1
        fid_level -= 1
