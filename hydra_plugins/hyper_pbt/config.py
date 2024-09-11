"""Config for HyperPBT sweeper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore


@dataclass
class HyperPBTConfig:
    """Config for HyperPBT sweeper."""

    _target_: str = "hydra_plugins.hypersweeper.hypersweeper.Hypersweeper"
    opt_constructor: str = "hydra_plugins.hyper_pbt.hyper_pbt.make_pbt"
    search_space: dict | None = field(default_factory=dict)
    resume: str | bool = False
    budget: Any | None = None
    n_trials: int | None = None
    budget_variable: str | None = None
    loading_variable: str | None = None
    saving_variable: str | None = None
    sweeper_kwargs: dict | None = field(default_factory=dict)


@dataclass
class HyperPB2Config:
    """Config for HyperPB2 sweeper."""

    _target_: str = "hydra_plugins.hypersweeper.hypersweeper.Hypersweeper"
    opt_constructor: str = "hydra_plugins.hyper_pbt.hyper_pb2.make_pb2"
    search_space: dict | None = field(default_factory=dict)
    resume: str | bool = False
    budget: Any | None = None
    n_trials: int | None = None
    budget_variable: str | None = None
    loading_variable: str | None = None
    saving_variable: str | None = None
    sweeper_kwargs: dict | None = field(default_factory=dict)


ConfigStore.instance().store(
    group="hydra/sweeper",
    name="HyperPBT",
    node=HyperPBTConfig,
    provider="hypersweeper",
)

ConfigStore.instance().store(
    group="hydra/sweeper",
    name="HyperPB2",
    node=HyperPB2Config,
    provider="hypersweeper",
)
