from __future__ import annotations

from typing import Any

import ConfigSpace as CS  # noqa: N817
import numpy as np
import pandas as pd
from ConfigSpace import Configuration, ConfigurationSpace
from hydra_plugins.hypersweeper import Info, Result
from hydra_plugins.hypersweeper.search_space_encoding import (
    search_space_to_config_space,
)
from omegaconf import DictConfig, OmegaConf


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
    elif isinstance(v, float) and \
        isinstance(
            configspace[k],
            CS.UniformIntegerHyperparameter | CS.NormalIntegerHyperparameter | CS.BetaIntegerHyperparameter
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
        k: maybe_convert_types(k, v, configspace=configspace) for k,v in x.items() if k.startswith("hp_config")
    }
    return Configuration(configuration_space=configspace, values=hp_config)


def read_warmstart_data(initial_design_fn: str, search_space: DictConfig) -> list[tuple[Info, Result]]:
    """Read initial design / warmstart data from csv-logfile.

    Parameters
    ----------
    initial_design_fn : str
        The path to the log file.
    search_space : DictConfig
        The search space which will be converted to a ConfigSpace.ConfigurationSpace, by default None.
        The search space can be loaded via `search_space = OmegaConf.load(search_space_fn)`.

    Returns.
    --------
    list[tuple[Info, Result]]
        The run data in hypersweeper format from the logs.
    """
    configspace = search_space_to_config_space(search_space=search_space)
    initial_design = pd.read_csv(initial_design_fn)
    # Assuming the seed in the logs is the global seed, not for the the config
    configs = initial_design.apply(convert_to_configuration, args=(configspace,), axis=1).to_list()
    budgets = initial_design["budget"]
    logged_performances = initial_design["performance"].to_list()
    infos = [Info(config=c, budget=b) for c, b in zip(configs, budgets)]
    results = [Result(info=i, performance=p) for i,p in zip(infos, logged_performances)]
    return list(zip(infos, results))


if __name__ == "__main__":
    from omegaconf import OmegaConf
    search_space = OmegaConf.load("/home/numina/Documents/repos/ARLBench/runscripts/configs/search_space/dqn_cc.yaml")
    read_warmstart_data(
        initial_design_fn="/home/numina/Documents/repos/ARLBench/runscripts/configs/initial_design/cc_cartpole_dqn.csv",
        search_space=search_space
    )