"""Config for HyperRS sweeper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore


@dataclass
class HyperSOBOLConfig:
    """Config for HyperRS sweeper."""

    _target_: str = "hydra_plugins.hypersweeper.hypersweeper.Hypersweeper"
    opt_constructor: str = "hydra_plugins.hyper_sobol.hyper_sobol.make_sobol"
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
    name="HyperSOBOL",
    node=HyperSOBOLConfig,
    provider="hypersweeper",
)
