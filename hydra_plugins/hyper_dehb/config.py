"""Config for HyperRS sweeper."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore

if (spec := importlib.util.find_spec("dehb")) is None:
    print("Couldn't import DEHB, the DEHB Hypersweeper will not be available.")
else:

    @dataclass
    class HyperDEHBConfig:
        """Config for HyperRS sweeper."""

        _target_: str = "hydra_plugins.hypersweeper.hypersweeper.Hypersweeper"
        opt_constructor: str = "hydra_plugins.hyper_dehb.hyper_dehb.make_dehb"
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
        name="HyperDEHB",
        node=HyperDEHBConfig,
        provider="hypersweeper",
    )
