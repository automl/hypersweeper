"""Perform a LPI analysis of a target configuration."""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ConfigSpace import Configuration, ForbiddenValueError
from ConfigSpace.hyperparameters import CategoricalHyperparameter, NumericalHyperparameter
from ConfigSpace.util import change_hp_value
from hydra_plugins.hyper_analysis.utils import df_to_config, dict_to_config, get_overall_best_config, load_data
from hydra_plugins.hypersweeper import Info


class LPI:
    """Perform Local Parameter Importance analysis of target configuration."""

    def __init__(
        self,
        configspace,
        config=None,
        run_source=True,
        data_path=None,
        config_key="config_id",
        performance_key="performance",
        variation_key="env",
        configs_per_hp=20,
        seed=42,
    ) -> None:
        """Initialize the optimizer."""
        self.run_source = run_source
        self.configspace = configspace
        self.configs_per_hp = configs_per_hp
        self.rs = np.random.RandomState(seed)
        assert config or data_path, "Either config or data_path must be provided."
        if config:
            self.config = dict_to_config(configspace, config)
        else:
            df = load_data(data_path, performance_key, config_key, variation_key)
            self.config = df_to_config(configspace, get_overall_best_config(df))

        neighbors = self._get_neighborhood()
        print(f"HP values in grid: {neighbors}")
        print(f"Number of HP values: {sum([len(x[0]) for x in neighbors.values()])}")

        self.hp_list = []
        for hp in self.config:
            for value in neighbors[hp][1]:
                cur_cfg = Configuration(self.configspace, self.config)
                cur_cfg[hp] = value
                if cur_cfg == self.config:
                    continue
                self.hp_list.append(cur_cfg)

        if run_source:
            self.hp_list.append(self.config)

    def ask(self):
        """Move one config further in the path."""
        config = self.hp_list.pop()
        info = Info(config=config, budget=None, load_path=None, seed=None)
        return info, False, len(self.hp_list) == 0

    def tell(self, info, value):
        """Tell not applicable for LPI sweeper."""

    def finish_run(self, output_path):
        """Finish the run and calculate LPI scores."""
        df = pd.read_csv(pathlib.PurePath(output_path, "runhistory.csv"))
        if "seed" not in df.columns:
            df["seed"] = 0
        seeds = df["seed"].unique()
        lpi_scores_by_seed = {}

        hp_keys = self.configspace.unconditional_hyperparameters + self.configspace.conditional_hyperparameters
        for seed in seeds:
            seed_df = df[df["seed"] == seed]
            stats_df = (
                seed_df.groupby(["config_id", *hp_keys], as_index=False)["performance"]
                .agg(["mean"])
                .reset_index(drop=True)
            )

            if self.run_source:
                mask = seed_df.apply(
                    lambda row: all(row[k] == v for k, v in self.config.items() if k in seed_df.columns), axis=1
                )
                if not mask.any():
                    raise ValueError(f"Base configuration not found in run history for seed {seed}!")
                main_config_id = seed_df.loc[mask, "config_id"].to_numpy()[0]
                main_config_row = stats_df[stats_df["config_id"] == main_config_id]

            seed_lpi = {}
            for hp in self.config:
                if isinstance(self.config[hp], NumericalHyperparameter):
                    mask = np.any(np.isclose(stats_df[hp].to_numpy()[:, None], self.config[hp], rtol=1e-3), axis=1)
                else:
                    mask = stats_df[hp] == self.config[hp]
                sub_df = stats_df[~mask]
                if self.run_source:
                    sub_df = pd.concat([sub_df, main_config_row], ignore_index=True)
                seed_lpi[hp] = sub_df["mean"].var()

            total_importance = sum(seed_lpi.values())
            if total_importance > 0:
                seed_lpi = {hp: score / total_importance for hp, score in seed_lpi.items()}
            else:
                seed_lpi = dict.fromkeys(seed_lpi.keys(), 0)

            lpi_scores_by_seed[seed] = seed_lpi

        final_lpi = {}
        uncertainty = {}
        for hp in self.config:
            hp_values = [lpi_scores_by_seed[s][hp] for s in seeds]
            final_lpi[hp] = np.mean(hp_values)
            uncertainty[hp] = np.var(hp_values)

        lpi_df = pd.DataFrame(
            {
                "hyperparameter": list(final_lpi.keys()),
                "lpi": list(final_lpi.values()),
                "uncertainty": list(uncertainty.values()),
            }
        )
        lpi_df = lpi_df.sort_values(by="lpi", ascending=False)
        lpi_df.to_csv(pathlib.PurePath(output_path, "lpi_scores.csv"), index=False)

        plt.figure(figsize=(8, 5))
        plt.bar(lpi_df["hyperparameter"], lpi_df["lpi"], yerr=lpi_df["uncertainty"], capsize=5)
        plt.ylim(0, 1)
        plt.xlabel("Hyperparameters")
        plt.ylabel("Local Parameter Importance (LPI)")
        plt.title("Hyperparameter Importance (LPI)")
        plt.xticks(rotation=45)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(pathlib.PurePath(output_path, "lpi_plot.png"))

    def _get_neighborhood(self):
        # Taken and adjusted from DeepCave
        hp_names = list(self.configspace.keys())
        incumbent_array = self.config.get_array()

        neighborhood = {}
        for hp_idx, hp_name in enumerate(hp_names):
            # Check if hyperparameter is active
            if not np.isfinite(incumbent_array[hp_idx]):
                continue

            hp_neighborhood = []
            checked_neighbors = []  # On unit cube
            checked_neighbors_non_unit_cube = []  # Not on unit cube
            hp = self.configspace[hp_name]
            num_neighbors = hp.get_num_neighbors(self.config[hp_name])

            if num_neighbors == 0:
                continue
            if np.isinf(num_neighbors):
                assert isinstance(hp, NumericalHyperparameter)
                if hp.log:
                    base = np.e
                    log_lower = np.log(hp.lower) / np.log(base)
                    log_upper = np.log(hp.upper) / np.log(base)
                    neighbors_range = np.logspace(
                        start=log_lower,
                        stop=log_upper,
                        num=self.configs_per_hp,
                        endpoint=True,
                        base=base,
                    )
                else:
                    neighbors_range = np.linspace(hp.lower, hp.upper, self.configs_per_hp)
                neighbors = [hp.to_vector(x) for x in neighbors_range]
            else:
                neighbors = hp.neighbors_vectorized(incumbent_array[hp_idx], n=self.configs_per_hp, seed=self.rs)

            for neighbor in neighbors:
                if neighbor in checked_neighbors:
                    continue

                new_array = incumbent_array.copy()
                new_array = change_hp_value(self.configspace, new_array, hp_name, neighbor, hp_idx)

                try:
                    new_config = Configuration(self.configspace, vector=new_array)
                    hp_neighborhood.append(new_config)
                    new_config.check_valid_configuration()
                    self.configspace.check_configuration_vector_representation(new_array)

                    checked_neighbors.append(neighbor)
                    checked_neighbors_non_unit_cube.append(new_config[hp_name])
                except (ForbiddenValueError, ValueError):
                    pass

            sort_idx = [x[0] for x in sorted(enumerate(checked_neighbors), key=lambda y: y[1])]
            if isinstance(self.configspace[hp_name], CategoricalHyperparameter):
                checked_neighbors_non_unit_cube_categorical = list(np.array(checked_neighbors_non_unit_cube)[sort_idx])
                neighborhood[hp_name] = [
                    np.array(checked_neighbors)[sort_idx],
                    checked_neighbors_non_unit_cube_categorical,
                ]
            else:
                checked_neighbors_non_unit_cube_non_categorical = np.array(checked_neighbors_non_unit_cube)[sort_idx]
                neighborhood[hp_name] = [
                    np.array(checked_neighbors)[sort_idx],
                    checked_neighbors_non_unit_cube_non_categorical,
                ]

        return neighborhood


def make_lpi(configspace, kwargs):
    """Make LPI sweeper."""
    return LPI(configspace, **kwargs)
