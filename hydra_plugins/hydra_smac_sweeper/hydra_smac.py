# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import os
import math
import time
import json
import pickle
import wandb
import numpy as np
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    NormalIntegerHyperparameter,
    OrdinalHyperparameter,
    UniformIntegerHyperparameter,
)
from hydra.utils import to_absolute_path
#from deepcave import Recorder, Objective
from omegaconf import OmegaConf
from smac import HyperparameterOptimizationFacade, Scenario
from smac.intensifier.hyperband import Hyperband
from smac.runhistory.dataclasses import TrialValue, TrialInfo

log = logging.getLogger(__name__)


class HydraSMAC:
    def __init__(
        self,
        global_config,
        global_overrides,
        launcher,
        budget_arg_name,
        save_arg_name,
        n_trials,
        cs,
        seeds=False,
        slurm=False,
        slurm_timeout=10,
        max_parallelization=0.1,
        job_array_size_limit=100,
        intensifier="HB",
        max_budget=None,
        deterministic=True,
        base_dir=False,
        min_budget=None,
        wandb_project=False,
        wandb_entity=False,
        wandb_tags=["smac"],
        maximize=False,
    ):
        """
        Classic PBT Implementation.

        Parameters
        ----------
        launcher: HydraLauncher
            A hydra launcher (usually either for local runs or slurm)
        budget_arg_name: str
            Name of the argument controlling the budget, e.g. num_steps.
        loading_arg_name: str
            Name of the argument controlling the loading of agent parameters.
        saving_arg_name: str
            Name of the argument controlling the checkpointing.
        total_budget: int
            Total budget for a single population member.
            This could be e.g. the total number of steps to train a single agent.
        cs: ConfigSpace
            Configspace object containing the hyperparameter search space.
        seeds: List[int] | False
            If not False, optimization will be run and averaged across the given seeds.
        model_based: bool
            Whether a model-based backend (such as BO) is used. Should always be false if using default PBT.
        base_dir: str | None
            Directory for logs.
        population_size: int
            Number of agents in the population.
        config_interval: int | None
            Number of steps before new configuration is chosen. Either this or num_config_changes must be given.
        num_config_changes: int | None
            Total number of times the configuration is changed. Either this or config_interval must be given.
        quantiles: float
            Upper/lower performance percentages beyond which agents are replaced.
            Lower numbers correspond to more exploration, higher ones to more exploitation.
        resample_probability: float
            Probability of a hyperparameter being resampled.
        perturbation_factors: List[int]
            Hyperparamters are multiplied with the first factor when their value is increased
            and with the second if their value is decreased.
        categorical_fixed: bool
            Whether categorical hyperparameters are ignored or optimized jointly.
        categorical_prob: float
            Probability of categorical values being resampled.
        Returns
        -------
        None
        """
        self.global_overrides = global_overrides
        self.launcher = launcher
        self.budget_arg_name = budget_arg_name
        self.save_arg_name = save_arg_name
        self.configspace = cs
        self.output_dir = to_absolute_path(base_dir) if base_dir else to_absolute_path("./")
        os.makedirs(self.output_dir, exist_ok=True)
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.job_idx = 0
        self.seeds = seeds
        if (seeds or not deterministic) and len(self.global_overrides) > 0:
            for i in range(len(self.global_overrides)):
                if self.global_overrides[i].split("=")[0] == "seed":
                    self.global_overrides = self.global_overrides[:i] + self.global_overrides[i + 1 :]
                    break

        self.maximize = maximize
        self.slurm = slurm
        self.slurm_timeout = slurm_timeout
        self.max_parallel = min(job_array_size_limit, max(1, int(max_parallelization * n_trials)))

        self.min_budget = min_budget
        self.iteration = 0
        self.n_trials = n_trials
        self.opt_time = 0
        self.incumbent = []
        self.history = {}
        self.history["configs"] = []
        self.history["performances"] = []
        self.history["budgets"] = []
        self.deterministic = deterministic
        self.max_budget = max_budget

        self.scenario = Scenario(self.configspace, deterministic=deterministic, n_trials=n_trials, min_budget=min_budget, max_budget=max_budget)
        max_config_calls = len(self.seeds) if seeds and not deterministic else 1
        if intensifier == "HB":
            self.intensifier = Hyperband(self.scenario, incumbent_selection="highest_budget", n_seeds=max_config_calls)
        else:
            self.intensifier = HyperparameterOptimizationFacade.get_intensifier(
                self.scenario,
                max_config_calls=max_config_calls,
            )

        def dummy(arg, seed, budget):
            pass

        self.smac = HyperparameterOptimizationFacade(
            self.scenario,
            dummy,
            intensifier=self.intensifier,
            overwrite=True,
        )

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

        self.wandb_project = wandb_project
        if self.wandb_project:
            wandb_config = OmegaConf.to_container(global_config, resolve=False, throw_on_missing=False)
            assert wandb_entity, "Please provide an entity to log to W&B."
            wandb.init(
                project=self.wandb_project,
                entity=wandb_entity,
                tags=wandb_tags,
                config=wandb_config,
            )

    def run_configs(self, configs, budgets, seeds):
        """
        Run a set of overrides

        Parameters
        ----------
        overrides: List[Tuple]
            A list of overrides to launch

        Returns
        -------
        List[float]
            The resulting performances.
        List[float]
            The incurred costs.
        """
        # Generate overrides
        #TODO: handle budget correctly
        overrides = []
        for i in range(len(configs)):
            names = (list(configs[0].keys()) + [self.budget_arg_name] + [self.save_arg_name])
            if self.slurm:
               names += ["hydra.launcher.timeout_min"]
               optimized_timeout = (
                       self.slurm_timeout * 1 / (self.total_budget // budgets[i]) + 0.1 * self.slurm_timeout
                   )

            if self.seeds and self.deterministic:
                for s in self.seeds:
                    save_path = os.path.join(
                            self.checkpoint_dir, f"iteration_{self.iteration}_id_{i}_s{s}.pt"
                        )
                    values = list(configs[i].values()) + [budgets[i]] + [save_path]
                    if self.slurm:
                            values += [int(optimized_timeout)]
                    job_overrides = tuple(self.global_overrides) + tuple(
                            f"{name}={val}" for name, val in zip(names + ["seed"], values + [s])
                        )
                    overrides.append(job_overrides)
            elif not self.deterministic:
                save_path = os.path.join(
                            self.checkpoint_dir, f"iteration_{self.iteration}_id_{i}_s{s}.pt"
                        )
                values = list(configs[i].values()) + [budgets[i]] + [save_path]
                if self.slurm:
                            values += [int(optimized_timeout)]
                job_overrides = tuple(self.global_overrides) + tuple(
                            f"{name}={val}" for name, val in zip(names + ["seed"], values + [seeds[i]])
                        )
                overrides.append(job_overrides)
            else:
                save_path = os.path.join(self.checkpoint_dir, f"iteration_{self.iteration}_id_{i}.pt")
                values = list(configs[i].values()) + [budgets[i]] + [save_path]
                if self.slurm:
                    values += [int(optimized_timeout)]
                job_overrides = tuple(self.global_overrides) + tuple(
                        f"{name}={val}" for name, val in zip(names, values)
                    )
                overrides.append(job_overrides)

        # Run overrides
        res = self.launcher.launch(overrides, initial_job_idx=self.job_idx)
        self.job_idx += len(overrides)
        costs = [budgets[i] for i in range(len(res))]
        done = False
        while not done:
            for j in range(len(overrides)):
                try:
                    res[j].return_value
                    done = True
                except:
                    done = False

        performances = []
        if self.seeds and self.deterministic:
            for j in range(0, self.population_size):
                performances.append(np.mean([res[j * k + k].return_value for k in range(len(self.seeds))]))
        else:
            for j in range(len(overrides)):
                performances.append(res[j].return_value)
        if self.maximize:
            performances = [-p for p in performances]
        return performances, costs

    def get_incumbent(self):
        """
        Get the best sequence of configurations so far.

        Returns
        -------
        List[Configuration]
            Sequence of best hyperparameter configs
        Float
            Best performance value
        """
        best_current_id = np.argmin(self.history["performances"])
        inc_performance = self.history["performances"][best_current_id]
        inc_config = self.history["configs"][best_current_id]
        return inc_config, inc_performance

    def record_iteration(self, performances, configs, budgets):
        """
        Add current iteration to history.

        Parameters
        ----------
        performances: List[float]
            A list of the latest agent performances
        configs: List[Configuration]
            A list of the recent configs
        """
        for i in range(len(configs)):
            self.history["configs"].append(configs[i])
            self.history["performances"].append(performances[i])
            self.history["budgets"].append(budgets[i])
        self.iteration += 1

        if self.wandb_project:
            stats = {}
            stats["iteration"] = self.iteration
            stats["optimization_time"] = time.time() - self.start
            stats["incumbent_performance"] = -min(performances)
            best_config = configs[np.argmin(performances)]
            for n in best_config.keys():
                stats[f"incumbent_{n}"] = best_config.get(n)
            wandb.log(stats)

    def _save_incumbent(self, name=None):
        """
        Log current incumbent to file (as well as some additional info).

        Parameters
        ----------
        name: str | None
            Optional filename
        """
        if name is None:
            name = "incumbent.json"
        res = dict()
        incumbent, inc_performance = self.get_incumbent()
        res["config"] = incumbent.get_dictionary()
        res["score"] = float(inc_performance)
        res["total_training_steps"] = sum(self.history["budgets"])
        res["total_wallclock_time"] = self.start - time.time()
        res["total_optimization_time"] = self.opt_time
        with open(os.path.join(self.output_dir, name), "a+") as f:
            json.dump(res, f)
            f.write("\n")

    def run(self, verbose=False):
        """
        Actual optimization loop.
        In each iteration:
        - get configs (either randomly upon init or through perturbation)
        - run current configs
        - record performances

        Parameters
        ----------
        verbose: bool
            More logging info

        Returns
        -------
        List[Configuration]
            The incumbent configurations.
        """
        if verbose:
            log.info("Starting SMAC Sweep")
        self.start = time.time()
        while self.iteration <= self.n_trials:
            opt_time_start = time.time()
            configs = []
            budgets = []
            seeds = []
            for _ in range(self.max_parallel):
                info = self.smac.ask()
                configs.append(info.config)
                if info.budget is not None:
                    budgets.append(info.budget)
                else:
                    budgets.append(self.max_budget)
                seeds.append(info.seed)
            self.opt_time += time.time() - opt_time_start
            performances, costs = self.run_configs(configs, budgets, seeds)
            opt_time_start = time.time()
            if self.seeds and self.deterministic:
                seeds = np.zeros(len(performances))
            for config, performance, budget, seed, cost in zip(configs, performances, budgets, seeds, costs):
                info = TrialInfo(budget=budget, seed=seed, config=config)
                value = TrialValue(cost=-performance if self.maximize else performance, time=cost)
                self.smac.tell(info=info, value=value)
            self.record_iteration(performances, configs, budgets)
            if verbose:
                log.info(f"Finished Iteration {self.iteration}!")
                _, inc_performance = self.get_incumbent()
                log.info(f"Current incumbent currently has a performance of {np.round(inc_performance, decimals=2)}.")
            self._save_incumbent()
            self.opt_time += time.time() - opt_time_start
        total_time = time.time() - self.start
        inc_config, inc_performance = self.get_incumbent()
        if verbose:
            log.info(
                f"Finished SMAC Sweep! Total duration was {np.round(total_time, decimals=2)}s, \
                    best agent had a performance of {np.round(inc_performance, decimals=2)}"
            )
            log.info(f"The incumbent configuration is {inc_config}")
        return self.incumbent
