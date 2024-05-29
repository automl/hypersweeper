"""Hypersweeper backend."""

from __future__ import annotations

import logging
import operator
from collections.abc import Callable
from functools import reduce
from pathlib import Path
from typing import TYPE_CHECKING

from hydra.core.plugins import Plugins
from hydra.plugins.sweeper import Sweeper
from hydra.utils import get_class, get_method
from omegaconf import DictConfig, OmegaConf, open_dict
from rich import print as printr

from hydra_plugins.hypersweeper.search_space_encoding import \
    search_space_to_config_space

from .hypersweeper_sweeper import HypersweeperSweeper

if TYPE_CHECKING:
    from hydra.types import HydraContext, TaskFunction

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("get_class", get_class, replace=True)


class HypersweeperBackend(Sweeper):
    """Backend for the Hypersweeper."""

    def __init__(
        self,
        opt_constructor: Callable,
        search_space: DictConfig,
        resume: str | None = None,
        budget: int | None = None,
        n_trials: int | None = None,
        budget_variable: str | None = None,
        loading_variable: str | None = None,
        saving_variable: str | None = None,
        sweeper_kwargs: DictConfig | dict = None,
    ) -> None:
        """Backend for the Hypersweeper.
        Instantiate the sweeper with hydra and launch optimization.

        Parameters
        ----------
        opt_class: Class
            The hypersweeper subclass to use.
        search_space: DictConfig
            The search space, either a DictConfig from a hydra yaml config file,
            or a path to a json configuration space file
            in the format required of ConfigSpace,
            or already a ConfigurationSpace config space.
        budget: int | None
            Total budget for a single population member.
            This could be e.g. the total number of steps to train a single agent.
        budget_variable: str | None
            Name of the argument controlling the budget, e.g. num_steps.
        loading_variable: str | None
            Name of the argument controlling the loading of agent parameters.
        saving_variable: str | None
            Name of the argument controlling the checkpointing.
        sweeper_kwargs: DictConfig | None
            Arguments for sweeper.

        Returns:
        -------
        None

        """
        if sweeper_kwargs is None:
            sweeper_kwargs = {}
        self.opt_constructor = get_method(opt_constructor)
        self.search_space = search_space
        self.budget_variable = budget_variable
        self.loading_variable = loading_variable
        self.saving_variable = saving_variable
        self.sweeper_kwargs = sweeper_kwargs
        self.budget = int(budget) if budget is not None else None
        self.n_trials = int(n_trials) if n_trials is not None else None
        assert self.budget is not None or self.n_trials is not None, "Either budget or n_trials must be given."
        self.resume = resume

        self.task_function: TaskFunction | None = None
        self.sweep_dir: str | None = None

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        """Setup launcher.

        Parameters
        ----------
        hydra_context: HydraContext
        task_function: TaskFunction
        config: DictConfig

        Returns:
        -------
        None

        """
        self.config = config
        self.hydra_context = hydra_context
        self.launcher = Plugins.instance().instantiate_launcher(
            config=config, hydra_context=hydra_context, task_function=task_function
        )
        self.task_function = task_function
        self.sweep_dir = config.hydra.sweep.dir

    def sweep(self, arguments: list[str]) -> list | None:
        """Run PBT optimization and returns the incumbent configurations.

        Parameters
        ----------
        arguments: List[str]
            Hydra overrides for the sweep.

        Returns:
        -------
        List[Configuration] | None
            Incumbent (best) configuration.

        """
        assert self.config is not None
        assert self.launcher is not None
        assert self.hydra_context is not None

        printr("Config", self.config)
        printr("Hydra context", self.hydra_context)

        self.launcher.global_overrides = arguments
        if len(arguments) == 0:
            log.info("Sweep doesn't override default config.")
        else:
            log.info(f"Sweep overrides: {' '.join(arguments)}")

        configspace = search_space_to_config_space(search_space=self.search_space)

        optimizer = HypersweeperSweeper(
            make_optimizer=self.opt_constructor,
            global_config=self.config,
            global_overrides=arguments,
            launcher=self.launcher,
            budget_arg_name=self.budget_variable,
            save_arg_name=self.saving_variable,
            load_arg_name=self.loading_variable,
            budget=self.budget,
            n_trials=self.n_trials,
            base_dir=self.sweep_dir,
            cs=configspace,
            **self.sweeper_kwargs,
        )

        incumbent = optimizer.run(verbose=True)

        final_config = self.config
        with open_dict(final_config):
            del final_config["hydra"]
        for a in arguments:
            try:
                n, v = a.split("=")
                key_parts = n.split(".")
                reduce(operator.getitem, key_parts[:-1], final_config)[key_parts[-1]] = v
            except:  # noqa: E722
                print(f"Could not parse argument {a}, skipping.")
        schedules = {}
        for i in range(len(incumbent)):
            for k, v in incumbent[i].items():
                if k not in schedules:
                    schedules[k] = []
                schedules[k].append(v)
        for k in schedules:
            key_parts = k.split(".")
            reduce(operator.getitem, key_parts[:-1], final_config)[key_parts[-1]] = schedules[k]
        with open(Path(optimizer.output_dir) / "final_config.yaml", "w+") as fp:
            OmegaConf.save(config=final_config, f=fp)

        return incumbent
