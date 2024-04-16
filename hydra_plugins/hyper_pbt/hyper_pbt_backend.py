"""Backend for HyperPBT."""

from __future__ import annotations

import logging

from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf

from hydra_plugins.hyper_pbt.hyper_pbt_sweeper import HyperPBTSweeper
from hydra_plugins.hypersweeper import HypersweeperBackend

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("get_class", get_class, replace=True)


class HyperPBTBackend(HypersweeperBackend):
    """Backend for HyperPBT."""

    def __init__(
        self,
        search_space: DictConfig,
        resume: str | None = None,
        budget: int | None = None,
        budget_variable: str | None = None,
        loading_variable: str | None = None,
        saving_variable: str | None = None,
        sweeper_kwargs: DictConfig | dict = None,
    ) -> None:
        """Initialize the HyperPBT backend."""
        if sweeper_kwargs is None:
            sweeper_kwargs = {}
        super().__init__(
            opt_class=HyperPBTSweeper,
            search_space=search_space,
            resume=resume,
            budget=budget,
            budget_variable=budget_variable,
            loading_variable=loading_variable,
            saving_variable=saving_variable,
            sweeper_kwargs=sweeper_kwargs,
        )
