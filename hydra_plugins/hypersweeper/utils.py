"""Utility functions for the hypersweeper plugin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import ConfigSpace as CS  # noqa: N817
import numpy as np
import pandas as pd
from ConfigSpace import Configuration, ConfigurationSpace
from omegaconf import DictConfig, OmegaConf


@dataclass
class Info:
    """Information for the sweeper."""

    config: dict
    budget: float
    load_path: str = None
    seed: int = None


@dataclass
class Result:
    """Evaluation result for the optimizer."""

    info: Info = None
    performance: float = None
    cost: float = None


def maybe_convert_types(k: str, v: Any, configspace: ConfigurationSpace) -> Any | None:
    """Maybe convert types to match conditions and HP types.

    Types come from pandas Dataframe.

    Parameters
    ----------
    k : str
        Hyperparameter name
    v : Any
        Hyperparameter value as read in by pandas from csv
    configspace : ConfigurationSpace
        Configuration space with information about HP type

    Returns.
    --------
    Any | None
        The hyperparameter value with correct type
    """
    if np.isnan(v):
        v = None
    elif isinstance(v, float) and isinstance(
        configspace[k], CS.UniformIntegerHyperparameter | CS.NormalIntegerHyperparameter | CS.BetaIntegerHyperparameter
    ):
        v = int(v)
    return v


def convert_to_configuration(x: pd.Series, configspace: ConfigurationSpace) -> Configuration:
    """Convert configurations from run logs (csv) to ConfigSpace.Configuration.

    Parameters
    ----------
    x : pd.Series
        Row from runlogs.
    configspace : ConfigurationSpace
        Configuration space, necessary for conversion.

    Returns.
    --------
    Configuration
        The converted configuration.
    """
    x = dict(x)
    hp_config = {
        k: maybe_convert_types(k, v, configspace=configspace) for k, v in x.items() if k.startswith("hp_config")
    }
    return Configuration(configuration_space=configspace, values=hp_config)


def read_warmstart_data(warmstart_filename: str, search_space: DictConfig) -> list[tuple[Info, Result]]:
    """Read initial design / warmstart data from csv-logfile.

    Parameters
    ----------
    warmstart_filename : str
        The path to the log file.
    search_space : DictConfig
        The search space which will be converted to a ConfigSpace.ConfigurationSpace, by default None.
        The search space can be loaded via `search_space = OmegaConf.load(search_space_fn)`.

    Returns.
    --------
    list[tuple[Info, Result]]
        The run data in hypersweeper format from the logs.
    """
    # configspace = search_space_to_config_space(search_space=search_space)
    configspace = search_space
    initial_design = pd.read_csv(warmstart_filename)

    # The seed in the log is the seed the config was run on
    configs = initial_design.apply(convert_to_configuration, args=(configspace,), axis=1).to_list()
    budgets = initial_design["budget"]
    seeds = initial_design["seed"]
    logged_performances = initial_design["performance"].to_list()
    infos = [Info(config=c, budget=b, seed=s) for c, b, s in zip(configs, budgets, seeds, strict=False)]
    # the cost in hypersweeper is the runtime of the configuration
    # cost is not tracked for initial design
    results = [Result(info=i, performance=p, cost=0.0) for i, p in zip(infos, logged_performances, strict=False)]
    return list(zip(infos, results, strict=False))


if __name__ == "__main__":
    search_space = OmegaConf.load("/home/numina/Documents/repos/ARLBench/runscripts/configs/search_space/dqn_cc.yaml")
    read_warmstart_data(
        warmstart_filename="/home/numina/Documents/repos/ARLBench/runscripts/configs/initial_design/cc_cartpole_dqn.csv",
        search_space=search_space,
    )
