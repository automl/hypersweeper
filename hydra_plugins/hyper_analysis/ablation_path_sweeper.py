"""Computer Ablation Paths from Source to Target Configuration."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from hydra_plugins.hyper_analysis.utils import (df_to_config,
                                                get_best_config_per_variation,
                                                get_overall_best_config,
                                                load_data, to_json_types)
from hydra_plugins.hypersweeper import Info


class AblationPath:
    """Ablation Paths: Testing the best order of hyperparameter changes from a source to a target configuration."""

    def __init__(
        self,
        configspace,
        source_config=None,
        target_config=None,
        data_path=None,
        variation=None,
        performance_key="performance",
        config_key="config_id",
        variation_key="env",
        run_source=True,
    ) -> None:
        """Initialize the optimizer."""
        assert (data_path is not None and variation is not None) or (
            source_config is not None and target_config is not None
        ), "Either data_path and variation or source_config and target_config must be provided."
        if source_config is not None and target_config is not None:
            self.source_config = configspace.get_default_configuration()
            for k in source_config:
                self.source_config[k] = source_config[k]
            self.target_config = configspace.get_default_configuration()
            for k in target_config:
                self.target_config[k] = target_config[k]
        else:
            df = load_data(data_path, performance_key, config_key, variation_key)  # noqa: PD901
            self.source_config = df_to_config(configspace, get_overall_best_config(df))
            self.target_config = df_to_config(configspace, get_best_config_per_variation(df, variation))

        self.configspace = configspace
        self.different_hps = []

        for key in self.source_config:
            if self.source_config[key] != self.target_config[key]:
                self.different_hps.append(key)

        print(
            f"""Found {len(self.different_hps)} different hyperparameters
            between source and target config: {self.different_hps}"""
        )
        self.hps_left = self.different_hps.copy()
        self.ablation_path = []
        self.returns = []
        self.recompute_diffs = False
        self.configs = self.get_configs()
        self.configs = self.configs
        if run_source:
            self.configs += [self.source_config]

    def get_configs(self):
        """Get all possible configs for the next path segment."""
        if self.recompute_diffs:
            returns = [r[1] for r in self.returns]
            chosen_hp = self.hps_left[np.argmax(returns)]
            best_return = max(returns)
            self.ablation_path.append((chosen_hp, best_return))
            self.hps_left.remove(chosen_hp)
            self.returns = []
            self.source_config[chosen_hp] = self.target_config[chosen_hp]
            self.recompute_diffs = False

        configs = []
        for hp in self.hps_left:
            config = deepcopy(self.source_config)
            config[hp] = self.target_config[hp]
            configs.append(config)
        return configs

    def ask(self):
        """Move one config further in the path."""
        if self.recompute_diffs:
            self.configs = self.get_configs()

        if len(self.configs) == 0:
            raise ValueError("No more configs to evaluate. Likely the path is finished.")
        config = self.configs.pop()
        config = deepcopy(to_json_types(config))
        info = Info(config=config, budget=None, load_path=None, seed=None)
        done = len(self.configs) == 0
        if done:
            self.recompute_diffs = True
        return info, done

    def tell(self, info, value):
        """Record result."""
        self.returns.append((info.config, value.performance))


def make_ablation_path(configspace, kwargs):
    """Make ablation path sweeper."""
    return AblationPath(configspace, **kwargs)
