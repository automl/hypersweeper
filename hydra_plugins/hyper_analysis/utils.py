"""Utils for handling source configs and dataframes."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def load_data(data_path, performance_key, config_key, variation_key):
    """Load data and add mean performance columns."""
    data = pd.read_csv(data_path)
    overall_mean_per_config = data.groupby(config_key)[performance_key].mean().reset_index()
    data["overall_mean_performance"] = data[config_key].map(
        overall_mean_per_config.set_index(config_key)[performance_key]
    )
    mean_per_variation = (
        data.groupby([variation_key, config_key])[performance_key].mean().reset_index()[performance_key].to_numpy()
    )
    mean_per_variation = np.repeat(mean_per_variation, len(data) // len(mean_per_variation), axis=0)
    data["mean_performance"] = mean_per_variation
    return data


def get_overall_best_config(df):
    """Get the best config overall."""
    return df.loc[df["overall_mean_performance"].idxmin()].take([0])


def get_best_config_per_variation(df, var, variation_key="env"):
    """Get the best config for a specific variation."""
    df = df[df[variation_key] == var]  # noqa: PD901
    return df.loc[df["mean_performance"].idxmin()].take([0])


def df_to_config(configspace, row):
    """Convert a dataframe row to a configspace configuration."""
    config = configspace.sample_configuration()
    for c in row:
        if c in config:
            if math.isnan(row[c]):
                row[c] = 0
            if not isinstance(row[c], str):
                try:
                    value = float(row[c])
                except:  # noqa: E722
                    value = int(row[c])
            config[c] = value
    return config


def to_json_types(config):
    """Make sure all values are JSON serializable."""
    for key in config:
        if isinstance(config[key], np.int64):
            config[key] = int(config[key])
        elif isinstance(config[key], np.float64):
            config[key] = float(config[key])
    return config
