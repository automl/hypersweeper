"""Base class for ask-tell sweepers."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wandb
from ConfigSpace.hyperparameters import (CategoricalHyperparameter,
                                         OrdinalHyperparameter)
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from smac.runhistory.dataclasses import TrialInfo, TrialValue

log = logging.getLogger(__name__)


@dataclass
class Info:
    """Information for the sweeper."""

    config: dict
    budget: float
    load_path: str
    seed: int = None


class Hypersweeper:
    """Base class for ask-tell sweepers."""

    def __init__(
        self,
        global_config,
        global_overrides,
        launcher,
        make_optimizer,
        optimizer_kwargs,
        budget_arg_name,
        load_arg_name,
        save_arg_name,
        n_trials,
        cs,
        seeds=False,
        slurm=False,
        slurm_timeout=10,
        max_parallelization=0.1,
        job_array_size_limit=100,
        max_budget=None,
        base_dir=False,
        min_budget=None,
        wandb_project=False,
        wandb_entity=False,
        wandb_tags=None,
        maximize=False,
        deterministic=False,
    ):
        """Ask-Tell sweeper for hyperparameter optimization.

        Parameters
        ----------
        global_config: DictConfig
            The global configuration
        global_overrides:
            Global overrides for all jobs
        launcher: HydraLauncher
            A hydra launcher (usually either for local runs or slurm)
        make_optimizer: Function
            Function to create the optimizer object
        optimizer_kwargs: Dict
            Optimizer arguments
        budget_arg_name: str
            Name of the argument controlling the budget, e.g. num_steps.
        load_arg_name: str
            Name of the argument controlling the loading of agent parameters.
        save_arg_name: str
            Name of the argument controlling the checkpointing.
        n_trials: int
            Number of trials to run
        cs: ConfigSpace
            Configspace object containing the hyperparameter search space.
        seeds: List[int] | False
            If not False, optimization will be run and averaged across the given seeds.
        slurm: bool
            Whether to use slurm for parallelization
        slurm_timeout: int
            Timeout for slurm jobs, used for scaling the timeout based on budget
        max_parallelization: float
            Maximum parallelization factor.
            1 will run all jobs in parallel, 0 will run completely sequentially.
        job_array_size_limit:
            Maximum number of jobs to submit in parallel
        max_budget: int
            Maximum budget for a single trial
        base_dir:
            Base directory for saving checkpoints
        min_budget: int
            Minimum budget for a single trial
        wandb_project: str
            W&B project to log to. If False, W&B logging is disabled.
        wandb_entity: str
            W&B entity to log to
        wandb_tags:
            Tags to log to W&B
        maximize: bool
            Whether to maximize the objective function
        deterministic: bool
            Whether the target function is deterministic

        Returns:
        -------
        None
        """
        if wandb_tags is None:
            wandb_tags = ["hypersweeper"]
        self.global_overrides = global_overrides
        self.launcher = launcher
        self.budget_arg_name = budget_arg_name
        self.save_arg_name = save_arg_name
        self.load_arg_name = load_arg_name
        self.checkpoint_tf = False
        self.load_tf = False

        self.configspace = cs
        self.output_dir = Path(
            to_absolute_path(base_dir) if base_dir else to_absolute_path("./")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = Path(self.output_dir) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.job_idx = 0
        self.seeds = seeds
        if (seeds or not deterministic) and len(self.global_overrides) > 0:
            for i in range(len(self.global_overrides)):
                if self.global_overrides[i].split("=")[0] == "seed":
                    self.global_overrides = (
                        self.global_overrides[:i] + self.global_overrides[i + 1 :]
                    )
                    break

        self.maximize = maximize
        self.slurm = slurm
        self.slurm_timeout = slurm_timeout
        self.max_parallel = min(
            job_array_size_limit, max(1, int(max_parallelization * n_trials))
        )

        self.min_budget = min_budget
        self.trials_run = 0
        self.n_trials = n_trials
        self.iteration = 0
        self.opt_time = 0
        self.incumbent = []
        self.history = {}
        self.history["configs"] = []
        self.history["performances"] = []
        self.history["budgets"] = []
        self.deterministic = deterministic
        self.max_budget = max_budget

        self.optimizer = make_optimizer(self.configspace, optimizer_kwargs)

        self.categorical_hps = [
            n
            for n in list(self.configspace.keys())
            if isinstance(
                self.configspace.get_hyperparameter(n), CategoricalHyperparameter
            )
        ]
        self.categorical_hps += [
            n
            for n in list(self.configspace.keys())
            if isinstance(self.configspace.get_hyperparameter(n), OrdinalHyperparameter)
        ]
        self.continuous_hps = [
            n for n in list(self.configspace.keys()) if n not in self.categorical_hps
        ]
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
            wandb_config = OmegaConf.to_container(
                global_config, resolve=False, throw_on_missing=False
            )
            assert wandb_entity, "Please provide an entity to log to W&B."
            wandb.init(
                project=self.wandb_project,
                entity=wandb_entity,
                tags=wandb_tags,
                config=wandb_config,
            )

    def run_configs(self, configs, budgets, seeds, load_paths):  # noqa: PLR0912
        """Run a set of overrides.

        Parameters
        ----------
        overrides: List[Tuple]
            A list of overrides to launch

        Returns:
        -------
        List[float]
            The resulting performances.
        List[float]
            The incurred costs.
        """
        # Generate overrides
        overrides = []
        for i in range(len(configs)):
            names = [*list(configs[0].keys())]
            if self.budget_arg_name is not None:
                names += [self.budget_arg_name]
            if self.checkpoint_tf:
                names += [self.save_arg_name]
            if self.load_tf and self.iteration > 0:
                names += [self.load_arg_name]

            if self.slurm:
                names += ["hydra.launcher.timeout_min"]
                optimized_timeout = (
                    self.slurm_timeout * 1 / (self.total_budget // budgets[i])
                    + 0.1 * self.slurm_timeout
                )

            if self.seeds and self.deterministic:
                for s in self.seeds:
                    save_path = (
                        Path(self.checkpoint_dir)
                        / f"iteration_{self.iteration}_id_{i}_s{s}.pt"
                    )
                    values = [*list(configs[i].values())]
                    if self.budget_arg_name is not None:
                        values += [budgets[i]]
                    if self.checkpoint_tf:
                        values += [save_path]
                    if self.load_tf and self.iteration > 0:
                        values += [load_paths[i]]

                    if self.slurm:
                        values += [int(optimized_timeout)]
                    job_overrides = tuple(self.global_overrides) + tuple(
                        f"{name}={val}"
                        for name, val in zip(
                            [*names, "seed"], [*values, s], strict=True
                        )
                    )
                    overrides.append(job_overrides)
            elif not self.deterministic:
                save_path = (
                    Path(self.checkpoint_dir)
                    / f"iteration_{self.iteration}_id_{i}_s{s}.pt"
                )
                values = [*list(configs[i].values())]
                if self.budget_arg_name is not None:
                        values += [budgets[i]]
                if self.checkpoint_tf:
                    values += [save_path]
                if self.load_tf and self.iteration > 0:
                    values += [load_paths[i]]

                if self.slurm:
                    values += [int(optimized_timeout)]
                job_overrides = tuple(self.global_overrides) + tuple(
                    f"{name}={val}"
                    for name, val in zip(
                        [*names, "seed"], [*values, seeds[i]], strict=True
                    )
                )
                overrides.append(job_overrides)
            else:
                save_path = (
                    Path(self.checkpoint_dir) / f"iteration_{self.iteration}_id_{i}.pt"
                )
                values = [*list(configs[i].values())]
                if self.budget_arg_name is not None:
                    values += [budgets[i]]
                if self.checkpoint_tf:
                    values += [save_path]
                if self.load_tf and self.iteration > 0:
                    values += [Path(self.checkpoint_dir) / load_paths[i]]

                if self.slurm:
                    values += [int(optimized_timeout)]
                job_overrides = tuple(self.global_overrides) + tuple(
                    f"{name}={val}" for name, val in zip(names, values, strict=True)
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
                    res[j].return_value  # noqa:B018
                    done = True
                    self.trials_run += 1
                except:  # noqa:E722
                    done = False

        performances = []
        if self.seeds and self.deterministic:
            for j in range(self.population_size):
                performances.append(
                    np.mean(
                        [res[j * k + k].return_value for k in range(len(self.seeds))]
                    )
                )
        else:
            for j in range(len(overrides)):
                performances.append(res[j].return_value)
        if self.maximize:
            performances = [-p for p in performances]
        return performances, costs

    def get_incumbent(self):
        """Get the best sequence of configurations so far.

        Returns:
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
        """Add current iteration to history.

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
            for n in best_config:
                stats[f"incumbent_{n}"] = best_config.get(n)
            wandb.log(stats)

    def _save_incumbent(self, name=None):
        """Log current incumbent to file (as well as some additional info).

        Parameters
        ----------
        name: str | None
            Optional filename
        """
        if name is None:
            name = "incumbent.json"
        res = {}
        incumbent, inc_performance = self.get_incumbent()
        res["config"] = incumbent.get_dictionary()
        res["score"] = float(inc_performance)
        if self.budget_arg_name is not None:
            res["total_training_steps"] = sum(self.history["budgets"])
        else:
            res["total_training_steps"] = self.trials_run
        res["total_wallclock_time"] = self.start - time.time()
        res["total_optimization_time"] = self.opt_time
        with open(Path(self.output_dir) / name, "a+") as f:
            json.dump(res, f)
            f.write("\n")

    def run(self, verbose=False):
        """Actual optimization loop.
        In each iteration:
        - get configs (either randomly upon init or through perturbation)
        - run current configs
        - record performances.

        Parameters
        ----------
        verbose: bool
            More logging info

        Returns:
        -------
        List[Configuration]
            The incumbent configurations.
        """
        if verbose:
            log.info("Starting Sweep")
        self.start = time.time()
        while self.trials_run <= self.n_trials:
            opt_time_start = time.time()
            configs = []
            budgets = []
            seeds = []
            loading_paths = []
            t = 0
            terminate = False
            while t < self.max_parallel and not terminate:
                info, terminate = self.optimizer.ask()
                configs.append(info.config)
                t += 1
                if info.budget is not None:
                    budgets.append(info.budget)
                else:
                    budgets.append(self.max_budget)
                seeds.append(info.seed)
                if info.load_path is not None:
                    loading_paths.append(info.load_path)
            self.opt_time += time.time() - opt_time_start
            performances, costs = self.run_configs(
                configs, budgets, seeds, loading_paths
            )
            opt_time_start = time.time()
            if self.seeds and self.deterministic:
                seeds = np.zeros(len(performances))
            for config, performance, budget, seed, cost in zip(
                configs, performances, budgets, seeds, costs, strict=True
            ):
                logged_performance = -performance if self.maximize else performance
                info = TrialInfo(budget=budget, seed=seed, config=config)
                value = TrialValue(cost=logged_performance, time=cost)
                self.optimizer.tell(info=info, value=value)
            self.record_iteration(performances, configs, budgets)
            if verbose:
                log.info(f"Finished Iteration {self.iteration}!")
                _, inc_performance = self.get_incumbent()
                log.info(
                    f"Current incumbent currently has a performance of {np.round(inc_performance, decimals=2)}."  # noqa:E501
                )
            self._save_incumbent()
            self.opt_time += time.time() - opt_time_start
        total_time = time.time() - self.start
        inc_config, inc_performance = self.get_incumbent()
        if verbose:
            log.info(
                f"Finished Sweep! Total duration was {np.round(total_time, decimals=2)}s, \
                    incumbent had a performance of {np.round(inc_performance, decimals=2)}"  # noqa:E501
            )
            log.info(f"The incumbent configuration is {inc_config}")
        return self.incumbent
