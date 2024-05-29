"""Config for HyperRS sweeper."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore

if (spec := importlib.util.find_spec("nevergrad")) is not None:

    @dataclass
    class HyperNevergradConfig:
        """Config for HyperRS sweeper."""

        _target_: str = "hydra_plugins.hypersweeper.hypersweeper.Hypersweeper"
        opt_constructor: str = "hydra_plugins.hyper_nevergrad.hyper_nevergrad.make_nevergrad"
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
        name="HyperNevergrad",
        node=HyperNevergradConfig,
        provider="hypersweeper",
    )
else:
    print("Couldn't import Nevergrad, the Nevergrad Hypersweeper will not be available.")
