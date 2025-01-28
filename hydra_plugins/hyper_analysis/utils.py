"""Utils for handling source configs and dataframes."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ConfigSpace import Configuration


def load_data(data_path, performance_key, config_key, variation_key):
    """Load data and add mean performance columns."""
    data = pd.read_csv(data_path)
    overall_mean_per_config = data.groupby(config_key)[performance_key].mean().reset_index()
    data["overall_mean_performance"] = data[config_key].map(
        overall_mean_per_config.set_index(config_key)[performance_key]
    )
    mean_per_variation = (
        data.groupby([variation_key, config_key])[performance_key].mean().reset_index()[[variation_key, config_key, performance_key]]
    )
    data["mean_performance"] = data.apply(
        lambda row: mean_per_variation.set_index([variation_key, config_key])[performance_key].get(
            (row[variation_key], row[config_key]), None),
        axis=1
    )
    #mean_per_variation = np.repeat(mean_per_variation, len(data) // len(mean_per_variation), axis=0)
    #data["mean_performance"] = mean_per_variation
    return data


def get_overall_best_config(df):
    """Get the best config overall."""
    return df.loc[df["overall_mean_performance"].idxmax()]


def get_best_config_per_variation(df, var, variation_key="env"):
    """Get the best config for a specific variation."""
    df = df[df[variation_key] == var]  # noqa: PD901
    return df.loc[df["mean_performance"].idxmax()]


def df_to_config(configspace, row):
    """Convert a dataframe row to a configspace configuration."""
    config = configspace.get_default_configuration()
    unconditional_hps = configspace.unconditional_hyperparameters
    conditional_hps = configspace.conditional_hyperparameters
    row_dict = row.loc[unconditional_hps + conditional_hps].dropna().to_dict()
    config = Configuration(configspace, row_dict)
    return config


def to_json_types(config):
    """Make sure all values are JSON serializable."""
    for key in config:
        if isinstance(config[key], np.int64):
            config[key] = int(config[key])
        elif isinstance(config[key], np.float64):
            config[key] = float(config[key])
    return config
