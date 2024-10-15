"""HyperPBT Sweeper implementation."""

from __future__ import annotations

import os
import shutil

import numpy as np
from ConfigSpace.hyperparameters import (CategoricalHyperparameter,
                                         NormalIntegerHyperparameter,
                                         OrdinalHyperparameter,
                                         UniformIntegerHyperparameter)

from hydra_plugins.hypersweeper import Info


class PBT:
    """Population Based Training optimizer."""

    def __init__(
        self,
        configspace,
        population_size,
        config_interval,
        seed=42,
        quantiles=None,
        resample_probability=0.25,
        perturbation_factors=None,
        categorical_prob=0.1,
        categorical_fixed=False,
        self_destruct=False,
    ):
        """Initialize the optimizer."""
        self.model_based = False
        if perturbation_factors is None:
            perturbation_factors = [0.8, 1.2]
        if quantiles is None:
            quantiles = [0.2, 0.8]
        self.rng = np.random.default_rng(seed=seed)
        self.config_history = []
        self.performance_history = []
        self.init = True
        self.configspace = configspace
        self.configspace.seed(seed)

        self.population_size = population_size
        self.budget_per_run = config_interval
        self.population_id = 0
        self.population_evaluated = 0
        self.iteration = 0

        self.quantiles = quantiles
        self.resample_probability = resample_probability
        self.perturbation_factors = perturbation_factors
        self.categorical_prob = categorical_prob
        self.categorical_fixed = categorical_fixed

        self.categorical_hps = [
            n
            for n in list(self.configspace.keys())
            if isinstance(self.configspace.get_hyperparameter(n), CategoricalHyperparameter)
        ]
        self.categorical_hps += [
            n
            for n in list(self.configspace.keys())
            if isinstance(self.configspace.get_hyperparameter(n), OrdinalHyperparameter)
        ]
        self.continuous_hps = [n for n in list(self.configspace.keys()) if n not in self.categorical_hps]
        self.hp_bounds = np.array(
            [
                [
                    self.configspace.get_hyperparameter(n).lower,
                    self.configspace.get_hyperparameter(n).upper,
                ]
                for n in list(self.configspace.keys())
                if n not in self.categorical_hps
            ]
        )
        self.self_destruct = self_destruct

    def ask(self):
        """Ask for the next configuration."""
        iteration_end = self.population_id == self.population_size - 1
        if self.init:
            config = self.configspace.sample_configuration()
            self.population_id += 1
            if iteration_end:
                self.iteration += 1
            return Info(config=config, budget=self.budget_per_run, load_path=None, seed=None), iteration_end
        config, load_path = self.perturb_config(self.population_id)
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
        last_iteration_configs = self.config_history[-(self.population_size + self.population_evaluated) :]
        last_iteration_performances = self.performance_history[-(self.population_size + self.population_evaluated) :]
        if self.population_evaluated > 0:
            last_iteration_configs = last_iteration_configs[: -self.population_evaluated]
            last_iteration_performances = last_iteration_performances[: -self.population_evaluated]
        last_config = last_iteration_configs[population_id]
        last_performance = last_iteration_performances[population_id]
        performance_quantiles = np.quantile(last_iteration_performances, self.quantiles)
        worst_config_ids = [
            i
            for i in range(len(last_iteration_performances))
            if last_iteration_performances[i] > performance_quantiles[1]
        ]
        best_config_ids = [
            i
            for i in range(len(last_iteration_performances))
            if last_iteration_performances[i] < performance_quantiles[0]
        ]
        if len(best_config_ids) == 0:
            best_config_ids = [np.argmax(last_iteration_performances)]
        load_agent = population_id
        if population_id in worst_config_ids:
            load_agent = self.rng.choice(best_config_ids)
        load_path = f"iteration_{self.iteration-1}_id_{load_agent}"
        new_config = self.perturb_hps(
            last_config, performance=last_performance, is_good=population_id in best_config_ids
        )
        return new_config, load_path

    def perturb_hps(self, config, performance=None, is_good=None):  # noqa: ARG002
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
                if isinstance(hp, NormalIntegerHyperparameter | UniformIntegerHyperparameter):
                    perturbed_value = int(perturbed_value)
                config[name] = max(min(perturbed_value, hp.upper), hp.lower)

        if not self.categorical_fixed:
            for name in self.categorical_hps:
                if self.rng.random() < self.categorical_prob:
                    hp = self.configspace.get_hyperparameter(name)
                    config[name] = hp.rvs()
        return config

    def tell(self, info, value):
        """Report the result."""
        self.config_history.append(info.config)
        self.performance_history.append(value.performance)
        self.population_evaluated += 1
        if self.population_evaluated == self.population_size:
            self.population_evaluated = 0
            self.population_id = 0
            self.init = False
            if self.model_based:
                self.fit_model(self.performance_history, self.config_history)

            # Now that we have finished the iteration,
            # we can safely remove all checkpoints from the previous iteration
            print(f"Finished iteration {self.iteration}")
            print("Remove checkpoints")
            if self.self_destruct and self.iteration > 1:
                self.remove_checkpoints(self.iteration - 2)

    def remove_checkpoints(self, iteration: int) -> None:
        """Remove checkpoints."""
        # Delete all files in checkpoints dir starting with iteration_{iteration}
        for file in os.listdir(self.checkpoint_dir):
            if file.startswith(f"iteration_{iteration}"):
                file_path = os.path.join(self.checkpoint_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                else:
                    shutil.rmtree(file_path)


def make_pbt(configspace, pbt_args):
    """Make a PBT instance for optimization."""
    return PBT(configspace, **pbt_args)
