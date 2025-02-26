from __future__ import annotations

from typing import Any

import warnings

import numpy as np
import omegaconf
from ConfigSpace.configuration_space import Configuration
from ConfigSpace.hyperparameters import Constant
from scipy.stats.qmc import Sobol

from smac.utils.configspace import transform_continuous_designs

from ConfigSpace import ConfigurationSpace, Float

from hydra_plugins.hypersweeper import Info


class HyperSobol:
    def __init__(self, configspace: ConfigurationSpace, n_configs: int, seed: int, budgets):
        self._configspace = configspace
        self.n_configs = n_configs
        self.budgets = budgets if isinstance(budgets, list) or isinstance(budgets, omegaconf.ListConfig) \
            else [1.0]
        self._rng = np.random.RandomState(seed)

        self.configurations = self.select_configurations()
        self.ask_calls = 0

    def select_configurations(self) -> list[Configuration]:
        configs: list[Configuration] = []
        configs += self._select_configurations()
        return configs

    def _select_configurations(self) -> list[Configuration]:
        params = list(self._configspace.values())

        constants = 0
        for p in params:
            if isinstance(p, Constant):
                constants += 1

        dim = len(params) - constants
        sobol_gen = Sobol(d=dim, scramble=True, seed=self._rng.randint(low=0, high=10000000))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sobol = sobol_gen.random(self.n_configs)

        return transform_continuous_designs(
            design=sobol, origin="Initial Design: Sobol",
            configspace=self._configspace
        )

    def ask(self):
        round, config_idx = divmod(self.ask_calls, self.n_configs)

        self.ask_calls += 1
        info = Info(
            config=self.configurations[config_idx], budget=self.budgets[round],
                    load_path=None, seed=None)

        return info, False

    def tell(self, info, value):
        pass


def make_sobol(configspace, hyper_sobol_args):
    """Make a Sobol instance for optimization."""
    return HyperSobol(configspace, **hyper_sobol_args)


if __name__ == '__main__':
    from ConfigSpace import Float

    cs = ConfigurationSpace(seed=0)
    cs.add([
        Float("alpha", bounds=(0.0001, 100.), default=0.5, log=True),
    ])
    sobol_design = HyperSobol(configspace=cs, n_configs=10, seed=0)
    configs = sobol_design.select_configurations()

    # parse alpha values
    print(','.join([str(config["alpha"]) for config in configs]))
