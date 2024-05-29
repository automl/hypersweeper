"""HEBO in Hypersweeper."""

from __future__ import annotations

import importlib
from collections import abc

import numpy as np
import pandas as pd
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import (CategoricalHyperparameter, Constant,
                                         FloatHyperparameter, Hyperparameter,
                                         IntegerHyperparameter,
                                         OrdinalHyperparameter)

if (spec := importlib.util.find_spec("hebo")) is not None:
    from hebo.design_space.design_space import DesignSpace
    from hebo.optimizers.hebo import HEBO

from hydra_plugins.hypersweeper import Info


class HyperHEBOAdapter:
    """HEBO optimizer."""

    def __init__(self, hebo, configspace, design_space):
        """Initialize the optimizer."""
        self.hebo = hebo
        self.configspace = configspace
        self.design_space = design_space

    def ask(self):
        """Ask for the next configuration."""
        hebo_info = self.hebo.suggest(1)
        config = HEBOcfg2ConfigSpacecfg(
            hebo_suggestion=hebo_info,
            design_space=self.design_space,
            config_space=self.configspace,
        )
        info = Info(config, None, None, None)
        return info, False

    def tell(self, info, value):
        """Tell the result of the configuration."""
        cost = value.performance
        suggestion = ConfigSpacecfg2HEBOcfg(info.config)

        cost = np.asarray([cost]) if not isinstance(cost, abc.Sequence) else np.asarray(cost)

        self.hebo.observe(suggestion, cost)


def make_hebo(configspace, hebo_args):
    """Make a HEBO instance for optimization."""
    hps_hebo = []
    for _, v in configspace.items():
        hps_hebo.append(configspaceHP2HEBOHP(v))
    design_space = DesignSpace().parse(hps_hebo)
    hebo = HEBO(space=design_space, **hebo_args)
    return HyperHEBOAdapter(hebo, configspace, design_space)


# These functions were taken from the CARP-S project here: https://github.com/automl/CARP-S/blob/main/carps/optimizers/hebo.py#L23
def configspaceHP2HEBOHP(hp: Hyperparameter) -> dict:  # noqa: PLR0911, N802
    """Convert ConfigSpace hyperparameter to HEBO hyperparameter.

    Parameters
    ----------
    hp : Hyperparameter
        ConfigSpace hyperparameter

    Returns:
    -------
    dict
        HEBO hyperparameter

    Raises:
    ------
    NotImplementedError
        If ConfigSpace hyperparameter is anything else than
        IntegerHyperparameter, FloatHyperparameter, CategoricalHyperparameter,
        OrdinalHyperparameter or Constant
    """
    if isinstance(hp, IntegerHyperparameter):
        if hp.log:
            return {"name": hp.name, "type": "pow_int", "lb": hp.lower, "ub": hp.upper}
        return {"name": hp.name, "type": "int", "lb": hp.lower, "ub": hp.upper}
    elif isinstance(hp, FloatHyperparameter):  # noqa: RET505
        if hp.log:
            return {"name": hp.name, "type": "pow", "lb": hp.lower, "ub": hp.upper}
        return {"name": hp.name, "type": "num", "lb": hp.lower, "ub": hp.upper}
    elif isinstance(hp, CategoricalHyperparameter):
        return {"name": hp.name, "type": "cat", "categories": hp.choices}
    elif isinstance(hp, OrdinalHyperparameter):
        return {
            "name": hp.name,
            "type": "step_int",
            "lb": 0,
            "ub": len(hp.sequence),
            "step": 1,
        }
    elif isinstance(hp, Constant):
        return {"name": hp.name, "type": "cat", "categories": [hp.value]}
    else:
        raise NotImplementedError(f"Unknown hyperparameter type: {hp.__class__.__name__}")


def HEBOcfg2ConfigSpacecfg(  # noqa: N802
    hebo_suggestion: pd.DataFrame,
    design_space: DesignSpace,
    config_space: ConfigurationSpace,
) -> Configuration:
    """Convert HEBO config to ConfigSpace config.

    Parameters
    ----------
    hebo_suggestion : pd.DataFrame
        Configuration in HEBO format
    design_space : DesignSpace
        HEBO design space
    config_space : ConfigurationSpace
        ConfigSpace configuration space

    Returns:
    -------
    Configuration
        Config in ConfigSpace format

    Raises:
    ------
    ValueError
        If HEBO config is more than 1
    """
    if len(hebo_suggestion) > 1:
        raise ValueError(f"Only one suggestion is ok, got {len(hebo_suggestion)}.")
    hyp = hebo_suggestion.iloc[0].to_dict()
    for k in hyp:
        hp_type = design_space.paras[k]
        if hp_type.is_numeric and hp_type.is_discrete:
            hyp[k] = int(hyp[k])
            # Now we need to check if it is an ordinal hp
            hp_k = config_space.get_hyperparameter(k)
            if isinstance(hp_k, OrdinalHyperparameter):
                hyp[k] = hp_k.sequence[hyp[k]]
    return Configuration(configuration_space=config_space, values=hyp)


def ConfigSpacecfg2HEBOcfg(config: Configuration) -> pd.DataFrame:  # noqa: N802
    """Convert ConfigSpace config to HEBO suggestion.

    Parameters
    ----------
    config : Configuration
        Configuration

    Returns:
    -------
    pd.DataFrame
        Configuration in HEBO format, e.g.
            x1        x2
        0  2.817594  0.336420
    """
    config_dict = dict(config)
    return pd.DataFrame(config_dict, index=[0])
