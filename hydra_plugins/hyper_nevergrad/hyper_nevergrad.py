"""Hypersweeper Interface for Nevergrad."""

from __future__ import annotations

import importlib

if (spec := importlib.util.find_spec("nevergrad")) is not None:
    import nevergrad as ng

from ConfigSpace.hyperparameters import (CategoricalHyperparameter,
                                         NormalFloatHyperparameter,
                                         NormalIntegerHyperparameter,
                                         UniformFloatHyperparameter,
                                         UniformIntegerHyperparameter)

from hydra_plugins.hypersweeper import Info


class HyperNevergrad:
    """Nevergrad."""

    def __init__(self, configspace, optimizer) -> None:
        """Initialize the optimizer."""
        self.configspace = configspace
        self.optimizer = optimizer
        self.storage = {}

    def ask(self):
        """Randomly sample a configuration."""
        x = self.optimizer.ask()
        config = get_config_from_x(self.configspace, x)
        info = Info(config, None, None, None)
        self.storage[config] = x
        return info, False

    def tell(self, info, value):
        """Return the performance."""
        x = self.storage[info.config]
        self.optimizer.tell(x, value.performance)


def make_nevergrad(configspace, hyper_nevergrad_args):
    """Make a Nevergrad instance for optimization."""
    instrumentation = get_instrum_from_configspace(configspace)
    optimizer = hyper_nevergrad_args["optimizer"](parametrization=instrumentation)
    return HyperNevergrad(configspace, optimizer)


def get_instrum_from_configspace(configspace):
    """Get a nevergrad instrumentation from a ConfigSpace object."""
    params = []
    for k in configspace:
        param = configspace.get(k)
        if isinstance(param, NormalFloatHyperparameter | UniformFloatHyperparameter):
            params.append(ng.p.Scalar(upper=param.upper, lower=param.lower))
        elif isinstance(param, NormalIntegerHyperparameter | UniformIntegerHyperparameter):
            params.append(ng.p.Scalar(upper=param.upper, lower=param.lower).set_integer_casting())
        elif isinstance(param, CategoricalHyperparameter):
            params.append(ng.p.Choice(param.choices))
    return ng.p.Instrumentation(*params)


def get_config_from_x(configspace, x):
    """Get a configuration from nevergrad output."""
    config = configspace.sample_configuration()
    for (
        i,
        k,
    ) in enumerate(config.keys()):
        config[k] = max(
            min(configspace.get_hyperparameter(k).upper, x.args[i]),
            configspace.get_hyperparameter(k).lower,
        )
    return config
