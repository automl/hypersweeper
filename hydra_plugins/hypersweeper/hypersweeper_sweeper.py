"""Base class for ask-tell sweepers."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from hydra_plugins.hypersweeper.utils import Info, Result, read_warmstart_data

if TYPE_CHECKING:
    from ConfigSpace import Configuration, ConfigurationSpace
    from hydra.plugins.launcher import Launcher

log = logging.getLogger(__name__)


class HypersweeperSweeper:
    """Base class for ask-tell sweepers."""

    def __init__(
        self,
        global_config: DictConfig,
        global_overrides: list[str],
        launcher: Launcher,
        make_optimizer: Callable,
        budget_arg_name: str,
        load_arg_name: str,
        save_arg_name: str,
        cs: ConfigurationSpace,
        budget: int = 1_000_000,
        n_trials: int = 1_000_000,
        optimizer_kwargs: dict[str, str] | None = None,
        seeds: list[int] | None = None,
        seed_keyword: str = "seed",
        slurm: bool = False,
        slurm_timeout: int = 10,
        max_parallelization: float = 0.1,
        job_array_size_limit: int = 100,
        max_budget: int | None = None,
        base_dir: str | None = None,
        min_budget: str | None = None,
        wandb_project: str | None = None,
        wandb_entity: str | None = None,
        wandb_tags: list[str] | None = None,
        maximize: bool = False,
        deterministic: bool = True,
        checkpoint_tf: bool = False,
        load_tf: bool = False,
        checkpoint_path_typing: str = ".pt",
        warmstart_file: str | None = None,
    ):
        """Ask-Tell sweeper for hyperparameter optimization.

        Parameters
        ----------
        global_config: DictConfig
            The global configuration
        global_overrides: List[str]
            Global overrides for all jobs
        launcher: Launcher
            A hydra launcher (usually either for local runs or slurm)
        make_optimizer: Callable
            Function to create the optimizer object
        optimizer_kwargs: dict[str, str]
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
        seeds: List[int]
            If not False, optimization will be run and averaged across the given seeds.
        seed_keyword: str = "seed"
            Keyword for the seed argument
        slurm: bool
            Whether to use slurm for parallelization
        slurm_timeout: int
            Timeout for slurm jobs, used for scaling the timeout based on budget
        max_parallelization: float
            Maximum parallelization factor.
            1 will run all jobs in parallel, 0 will run completely sequentially.
        job_array_size_limit: int
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
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        if wandb_tags is None:
            wandb_tags = ["hypersweeper"]
        self.global_overrides = global_overrides
        self.launcher = launcher
        self.budget_arg_name = budget_arg_name
        self.save_arg_name = save_arg_name
        self.load_arg_name = load_arg_name
        self.checkpoint_tf = checkpoint_tf
        self.load_tf = load_tf

        self.configspace = cs
        self.output_dir = Path(to_absolute_path(base_dir) if base_dir else to_absolute_path("./"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = Path(self.output_dir) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.job_idx = 0
        self.seeds = seeds
        self.seed_keyword = seed_keyword
        if (seeds or not deterministic) and len(self.global_overrides) > 0:
            for i in range(len(self.global_overrides)):
                if self.global_overrides[i].split("=")[0] == self.seed_keyword:
                    self.global_overrides = self.global_overrides[:i] + self.global_overrides[i + 1 :]
                    break

        self.maximize = maximize
        self.slurm = slurm
        self.slurm_timeout = slurm_timeout
        if n_trials is not None:
            self.max_parallel = min(job_array_size_limit, max(1, int(max_parallelization * n_trials)))
        else:
            self.max_parallel = job_array_size_limit

        self.budget = budget
        self.min_budget = min_budget
        self.trials_run = 0
        self.n_trials = n_trials
        self.iteration = 0
        self.opt_time = 0
        self.history = defaultdict(list)
        self.incumbents = defaultdict(list)
        self.deterministic = deterministic
        self.max_budget = max_budget
        self.checkpoint_path_typing = checkpoint_path_typing

        self.optimizer = make_optimizer(self.configspace, optimizer_kwargs)
        self.optimizer.checkpoint_dir = self.checkpoint_dir
        self.optimizer.checkpoint_path_typing = self.checkpoint_path_typing
        self.optimizer.seeds = seeds

        self.warmstart_data: list[tuple[Info, Result]] = []

        if warmstart_file:
            self.warmstart_data = read_warmstart_data(warmstart_filename=warmstart_file, search_space=self.configspace)

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

    def run_configs(self, infos):
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
        if self.load_tf and self.iteration > 0:
            assert not any(p.load_path is None for p in infos), """
            Load paths must be provided for all configurations
            when working with checkpoints. If your optimizer does not support this,
            set the 'load_tf' parameter of the sweeper to False."""

        # Generate overrides
        overrides = []
        for i in range(len(infos)):
            names = [*list(infos[i].config.keys())]
            if self.budget_arg_name is not None:
                names += [self.budget_arg_name]
            if self.load_tf and self.iteration > 0:
                names += [self.load_arg_name]
            if self.checkpoint_tf:
                names += [self.save_arg_name]

            values = [*list(infos[i].config.values())]
            if self.budget_arg_name is not None:
                values += [infos[i].budget]

            if self.slurm and self.max_budget is not None:
                names += ["hydra.launcher.timeout_min"]
                optimized_timeout = (
                    self.slurm_timeout * 1 / (self.max_budget // infos[i].budget) + 0.1 * self.slurm_timeout
                )
                values += [int(optimized_timeout)]

            if self.seeds:
                for s in self.seeds:
                    local_values = values.copy()
                    load_path = Path(self.checkpoint_dir) / f"{infos[i].load_path!s}_s{s}{self.checkpoint_path_typing}"
                    save_path = self.get_save_path(i, s)

                    if self.load_tf and self.iteration > 0:
                        local_values += [load_path]
                    if self.checkpoint_tf:
                        local_values += [save_path]

                    job_overrides = tuple(self.global_overrides) + tuple(
                        f"{name}={val}"
                        for name, val in zip([*names, self.seed_keyword], [*local_values, s], strict=True)
                    )
                    overrides.append(job_overrides)
            elif not self.deterministic:
                assert not any(s.seed is None for s in infos), """
                For non-deterministic target functions, seeds must be provided.
                If the optimizer you chose does not support this,
                manually set the 'seeds' parameter of the sweeper to a list of seeds."""
                load_path = Path(self.checkpoint_dir) / f"{infos[i].load_path!s}_s{s}{self.checkpoint_path_typing}"
                save_path = self.get_save_path(i)

                job_overrides = tuple(self.global_overrides) + tuple(
                    f"{name}={val}"
                    for name, val in zip([*names, self.seed_keyword], [*values, infos[i].seed], strict=True)
                )
                overrides.append(job_overrides)
            else:
                load_path = Path(self.checkpoint_dir) / f"{infos[i].load_path!s}{self.checkpoint_path_typing}"
                save_path = self.get_save_path(i)

                if self.load_tf and self.iteration > 0:
                    values += [load_path]
                if self.checkpoint_tf:
                    values += [save_path]

                job_overrides = tuple(self.global_overrides) + tuple(
                    f"{name}={val}" for name, val in zip(names, values, strict=True)
                )
                overrides.append(job_overrides)

        # Run overrides
        res = self.launcher.launch(overrides, initial_job_idx=self.job_idx)
        self.job_idx += len(overrides)
        if self.seeds:
            costs = [infos[i].budget for i in range(len(res) // len(self.seeds))]
        else:
            costs = [infos[i].budget for i in range(len(res))]

        performances = []
        if self.seeds and self.deterministic:
            # When we have seeds, we want to have a list of performances for each config
            n_seeds = len(self.seeds)
            for config_idx in range(len(overrides) // n_seeds):
                performances.append([res[config_idx * n_seeds + seed_idx].return_value for seed_idx in range(n_seeds)])
                self.trials_run += 1
        else:
            for j in range(len(overrides)):
                performances.append(res[j].return_value)
                self.trials_run += 1
        return performances, costs

    def get_save_path(self, config_id, seed=None):
        """Get the save path for checkpoints.

        Returns:
        -------
        Path
            The save path
        """
        if self.seeds:
            save_path = (
                Path(self.checkpoint_dir)
                / f"iteration_{self.iteration}_id_{config_id}_s{seed}{self.checkpoint_path_typing}"
            )
        elif not self.deterministic:
            save_path = (
                Path(self.checkpoint_dir) / f"iteration_{self.iteration}_id_{config_id}{self.checkpoint_path_typing}"
            )
        else:
            save_path = (
                Path(self.checkpoint_dir) / f"iteration_{self.iteration}_id_{config_id}{self.checkpoint_path_typing}"
            )
        return save_path

    def get_incumbent(self) -> tuple[Configuration | dict, float]:
        """Get the best sequence of configurations so far.

        Returns:
        -------
        List[Configuration]
            Sequence of best hyperparameter configs
        Float
            Best performance value
        """
        if self.maximize:
            best_current_id = np.argmax(self.history["performance"])
        else:
            best_current_id = np.argmin(self.history["performance"])
        inc_performance = self.history["performance"][best_current_id]
        inc_config = self.history["config"][best_current_id]
        return inc_config, inc_performance

    def _write_csv(self, data: dict, filename: str) -> None:
        """Write a dictionary to a csv file.

        Parameters
        ----------
        data: dict
            The data to write to the csv file
        filename: str
            The name of the csv file
        """
        dataframe = pd.DataFrame(data)

        dataframes_to_concat = []
        if "config_id" not in dataframe.columns:
            dataframes_to_concat += [pd.DataFrame(np.arange(len(dataframe)), columns=["config_id"])]

        # Since some configs might not include values for all hyperparameters
        # (e.g. when using conditions), we need to make sure that the dataframe
        # has all hyperparameters as columns
        hyperparameters = [str(hp) for hp in list(self.configspace.keys())]
        configs_df = pd.DataFrame(list(dataframe["config"]), columns=hyperparameters)

        # Now we merge the basic dataframe with the configs
        dataframes_to_concat += [dataframe.drop(columns="config"), configs_df]
        full_dataframe = pd.concat(dataframes_to_concat, axis=1)
        full_dataframe.to_csv(Path(self.output_dir) / f"{filename}.csv", index=False)

    def write_history(
        self, performances: list[list[float]] | list[float], configs: list[Configuration], budgets: list[float]
    ) -> None:
        """Write the history of the optimization to a csv file.

        Parameters
        ----------
        performances: Union[list[list[float]], list[float]]
            A list of the latest agent performances, either one value for each config or a list of values for each seed
        configs: list[Configuration],
            A list of the recent configs
        budgets: list[float]
            A list of the recent budgets
        """
        for i in range(len(configs)):
            self.history["config"].append(configs[i])
            if self.seeds:
                # In this case we have a list of performances for each config,
                # one for each seed
                assert isinstance(performances[i], list)
                self.history["performance"].append(np.mean(performances[i]))
                for seed_idx, seed in enumerate(self.seeds):
                    self.history[f"performance_{self.seed_keyword}_{seed}"].append(performances[i][seed_idx])
            else:
                self.history["performance"].append(performances[i])
            if budgets[i] is not None:
                self.history["budget"].append(budgets[i])
            else:
                self.history["budget"].append(self.max_budget)
        self._write_csv(self.history, "runhistory")

    def write_incumbents(self) -> None:
        """Write the incumbent configurations to a csv file."""
        if self.maximize:
            best_config_id = np.argmax(self.history["performance"])
        else:
            best_config_id = np.argmin(self.history["performance"])
        self.incumbents["config_id"].append(best_config_id)
        self.incumbents["config"].append(self.history["config"][best_config_id])
        self.incumbents["performance"].append(self.history["performance"][best_config_id])
        self.incumbents["budget"].append(self.history["budget"][best_config_id])
        try:
            self.incumbents["budget_used"].append(sum(self.history["budget"]))
        except:  # noqa:E722
            self.incumbents["budget_used"].append(self.trials_run)
        self.incumbents["total_wallclock_time"].append(time.time() - self.start)
        self.incumbents["total_optimization_time"].append(self.opt_time)

        self._write_csv(self.incumbents, "incumbent")

        if self.wandb_project:
            stats = {}
            stats["iteration"] = self.iteration
            stats["total_optimization_time"] = self.incumbents["total_optimization_time"][-1]
            stats["incumbent_performance"] = self.incumbents["performance"][-1]
            best_config = self.incumbents["config"][-1]
            for n in best_config:
                stats[f"incumbent_{n}"] = best_config.get(n)
            wandb.log(stats)

    def run(self, verbose=False):
        """Actual optimization loop.
        In each iteration:
        - get configs (either randomly upon init or through perturbation)
        - run current configs
        - write history and incbument to a csv file.

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
        trial_termination = False
        budget_termination = False
        optimizer_termination = False
        done = False
        if len(self.warmstart_data) > 0:
            for info, value in self.warmstart_data:
                self.optimizer.tell(info=info, value=value)
        while not done:
            opt_time_start = time.time()
            configs = []
            budgets = []
            seeds = []
            loading_paths = []
            infos = []
            t = 0
            terminate = False
            while (t < self.max_parallel
                   and not terminate
                   and not trial_termination
                   and not budget_termination
                   and not optimizer_termination
            ):
                try:
                    info, terminate, optimizer_termination = self.optimizer.ask()
                except Exception as e:  # noqa: BLE001
                    if len(infos) > 0:
                        print("Optimizer failed on ask - running remaining configs.")
                        performances, costs = self.run_configs(infos)
                        self.write_history(performances, configs, budgets)
                    print("Optimizer failed on ask - terminating optimization.")
                    print(f"Error was: {e}")
                    return self.incumbents[-1]

                configs.append(info.config)
                t += 1
                if info.budget is not None:
                    budgets.append(info.budget)
                else:
                    budgets.append(self.max_budget)
                seeds.append(info.seed)
                if info.load_path is not None:
                    loading_paths.append(info.load_path)
                infos.append(info)
                if not any(b is None for b in self.history["budget"]) and self.budget is not None:
                    budget_termination = sum(self.history["budget"]) >= self.budget
                if self.n_trials is not None:
                    trial_termination = self.trials_run + len(configs) >= self.n_trials

            self.opt_time += time.time() - opt_time_start
            performances, costs = self.run_configs(infos)
            opt_time_start = time.time()
            if self.seeds and self.deterministic:
                seeds = np.zeros(len(performances))

            for info, performance, cost in zip(infos, performances, costs, strict=True):
                run_performance = float(np.mean(performance)) if self.seeds else performance

                logged_performance = -run_performance if self.maximize else run_performance
                value = Result(performance=logged_performance, cost=cost)
                self.optimizer.tell(info=info, value=value)

            self.write_history(performances, configs, budgets)
            self.write_incumbents()

            if verbose:
                log.info(f"Finished Iteration {self.iteration}!")
                _, inc_performance = self.get_incumbent()
                log.info(f"Current incumbent has a performance of {np.round(inc_performance, decimals=2)}.")

            self.opt_time += time.time() - opt_time_start
            done = trial_termination or budget_termination or optimizer_termination
            self.iteration += 1

        total_time = time.time() - self.start
        inc_config, inc_performance = self.get_incumbent()

        self.optimizer.finish_run(Path(self.output_dir))

        if verbose:
            log.info(
                f"Finished Sweep! Total duration was {np.round(total_time, decimals=2)}s, \
                    incumbent had a performance of {np.round(inc_performance, decimals=2)}"
            )
            log.info(f"The incumbent configuration is {inc_config}")
        return self.incumbents[-1]
