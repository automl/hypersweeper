"""HyperPBT Sweeper implementation."""

from __future__ import annotations

import numpy as np
from ConfigSpace.hyperparameters import (NormalIntegerHyperparameter,
                                         UniformIntegerHyperparameter)

from hydra_plugins.hypersweeper import Info


class PBT:
    """Population Based Training optimizer."""

    def __init__(self):
        """Initialize the optimizer."""
        self.rng = np.random.default_rng(seed=None)  # TODO
        self.config_history = []
        self.performance_history = []
        self.init = True
        self.configspace = None  # TODO

        self.population_size = None  # TODO
        self.budget_per_run = None  # TODO
        self.population_id = 0
        self.population_evaluated = 0
        self.iteration = 0

    def ask(self):
        """Ask for the next configuration."""
        iteration_end = self.population_id <= self.population_size
        if self.init:
            config = self.configspace.sample_configuration()
            self.population_id += 1
            if iteration_end:
                self.iteration += 1
            return Info(
                config=config, budget=self.budget_per_run, load_path=None, seed=None
            ), iteration_end
        config, load_path = self.perturb_config(config, self.population_id)
        self.population_id += 1
        if iteration_end:
            self.iteration += 1
        return Info(
            config=config,
            budget=self.budget_per_run,
            load_path=load_path,
            seed=None,
        ), iteration_end

    def perturb_config(self, population_id):
        """Perturb existing configuration."""
        last_config = self.config_history[
            -(self.population_size + self.population_id) : -self.population_id
        ][population_id]
        last_performances = self.performance_history[
            -(self.population_size + self.population_id) : -self.population_id
        ]
        performance_quantiles = np.quantile(last_performances, [self.quantiles])[0]
        worst_config_ids = [
            i
            for i in range(len(last_performances))
            if last_performances[i] > performance_quantiles[1]
        ]
        best_config_ids = [
            i
            for i in range(len(last_performances))
            if last_performances[i] < performance_quantiles[0]
        ]
        if len(best_config_ids) == 0:
            best_config_ids = [np.argmax(last_performances)]
        if population_id in worst_config_ids:
            load_agent = self.rng.choice(best_config_ids)
            load_path = f"iteration_{self.iteration-1}_id_{load_agent}.pt"
        new_config = self.perturb_hps(last_config)
        return new_config, load_path

    def perturb_hps(self, config):
        """Perturb the hyperparameters."""
        for name in self.continuous_hps:
            hp = self.configspace.get_hyperparameter(name)
            if self.rng.random() < self.resample_probability:
                # Resample hyperparamter
                config[name] = hp.rvs()
            else:
                # Perturb
                perturbation_factor = self.rng.choice(self.perturbation_factors)
                perturbed_value = config[name] * perturbation_factor
                if isinstance(
                    hp, NormalIntegerHyperparameter | UniformIntegerHyperparameter
                ):
                    perturbed_value = int(perturbed_value)
                config[name] = max(min(perturbed_value, hp.upper), hp.lower)

        if not self.categorical_fixed:
            for name in self.categorical_hps:
                if self.rng.random() < self.categorical_prob:
                    hp = self.configspace.get_hyperparameter(name)
                    config[name] = hp.rvs()
        return config

    def tell(self, info, result):
        """Report the result."""
        self.config_history.append(info.config)
        self.performance_history.append(result.cost)
        self.population_evaluated += 1
        if self.population_evaluated == self.population_size:
            self.population_evaluated = 0
            self.population_id = 0
            self.init = False


def make_pbt(configspace, pbt_args):
    """Make a PBT instance for optimization."""
    return PBT(configspace, **pbt_args)
