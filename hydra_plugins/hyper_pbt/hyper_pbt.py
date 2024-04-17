"""HyperPBT Sweeper implementation."""

from __future__ import annotations

import numpy as np
from ConfigSpace.hyperparameters import (NormalIntegerHyperparameter,
                                         UniformIntegerHyperparameter)

from hydra_plugins.hypersweeper import HypersweeperSweeper, Info


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
    return PBT(**pbt_args)


class HyperPBTSweeper(HypersweeperSweeper):
    """Hydra Sweeper for PBT."""

    def __init__(
        self,
        global_config,
        global_overrides,
        launcher,
        optimizer_kwargs,
        budget_arg_name,
        save_arg_name,
        load_arg_name,
        n_trials,
        cs,
        seeds=False,
        slurm=False,
        slurm_timeout=10,
        max_parallelization=0.1,
        job_array_size_limit=100,
        max_budget=None,
        deterministic=True,
        base_dir=False,
        min_budget=None,
        wandb_project=False,
        wandb_entity=False,
        wandb_tags=None,
        maximize=False,
    ):
        """Initialize the Hypersweeper with PBT as the optimizer."""
        if wandb_tags is None:
            wandb_tags = ["pbt"]
        super().__init__(
            global_config=global_config,
            global_overrides=global_overrides,
            launcher=launcher,
            make_optimizer=make_pbt,
            optimizer_kwargs=optimizer_kwargs,
            budget_arg_name=budget_arg_name,
            save_arg_name=save_arg_name,
            load_arg_name=load_arg_name,
            n_trials=n_trials,
            cs=cs,
            seeds=seeds,
            slurm=slurm,
            slurm_timeout=slurm_timeout,
            max_parallelization=max_parallelization,
            job_array_size_limit=job_array_size_limit,
            max_budget=max_budget,
            deterministic=deterministic,
            base_dir=base_dir,
            min_budget=min_budget,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_tags=wandb_tags,
            maximize=maximize,
        )
        # Save and load target functions
        self.checkpoint_tf = True
        self.load_tf = True

        # Provide HP info to PBT
        self.optimizer.categorical_hps = self.categorical_hps
        self.optimizer.continuous_hps = self.continuous_hps
        self.optimizer.continuous_hps = self.continuous_hps
        self.optimizer.hp_bounds = self.hp_bounds
