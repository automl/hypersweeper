"""Get a regular Grid across a search space, e.g. to do an ANOVA."""

from __future__ import annotations

import numpy as np
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from hydra_plugins.hypersweeper import Info


class Grid:
    """Get regular grids across a search space."""

    def __init__(
        self,
        configspace,
        max_grid_size=None,
        configs_per_hp=None,
    ) -> None:
        """Initialize the optimizer."""
        assert (
            max_grid_size is not None or configs_per_hp is not None
        ), "Either total_grid_size or configs_per_hp must be provided."

        self.configspace = configspace
        if max_grid_size is None:
            total_grid_size = configs_per_hp ** len(configspace.keys())
        if configs_per_hp is None:
            configs_per_hp = int(max_grid_size ** (1 / len(configspace.keys())))
            total_grid_size = configs_per_hp ** len(configspace.keys())

        print(f"Total grid size is {total_grid_size}, meaning {configs_per_hp} values per hyperparameter.")
        self.hp_values = {}
        for hp in configspace:
            if isinstance(configspace[hp], CategoricalHyperparameter):
                self.hp_values[hp] = np.linspace(0, len(configspace[hp].choices), configs_per_hp)
                self.hp_values[hp] = [
                    configspace[hp].choices[min(int(v), len(configspace[hp].choices) - 1)] for v in self.hp_values[hp]
                ]
            else:
                self.hp_values[hp] = np.linspace(configspace[hp].lower, configspace[hp].upper, configs_per_hp)
        print(f"HP values in grid: {self.hp_values}")
        self.config_indices = {hp: 0 for hp in configspace}

    def reset_indices(self, i):
        """Increment last index and pass on overflow to previous one."""
        self.config_indices[list(self.config_indices.keys())[i]] += 1
        if self.config_indices[list(self.config_indices.keys())[i]] >= len(
            self.hp_values[list(self.config_indices.keys())[i]]
        ):
            self.config_indices[list(self.config_indices.keys())[i]] = 0
            try:
                self.reset_indices(i - 1)
            except KeyError:
                print("Evaluated full grid.")
                raise StopIteration

    def ask(self):
        """Move one config further in the path."""
        config = self.configspace.sample_configuration()
        for hp in config:
            config[hp] = self.hp_values[hp][self.config_indices[hp]]
        self.reset_indices(len(list(self.config_indices.keys())) - 1)
        info = Info(config=config, budget=None, load_path=None, seed=None)
        return info, False

    def tell(self, info, value):
        """Do nothing for Grid."""


def make_grid(configspace, kwargs):
    """Make grid sweeper."""
    return Grid(configspace, **kwargs)
