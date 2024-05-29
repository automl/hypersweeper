"""Wrapper class for the Hypersweeper backend functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hydra.plugins.sweeper import Sweeper

if TYPE_CHECKING:
    from hydra.types import HydraContext, TaskFunction
    from omegaconf import DictConfig


class Hypersweeper(Sweeper):
    """This is basically just a wrapper class for the backend functionality."""

    def __init__(self, *args: Any, **kwargs: dict[Any, Any]) -> None:
        """Initialize the sweeper."""
        from .hypersweeper_backend import HypersweeperBackend

        self.sweeper = HypersweeperBackend(*args, **kwargs)

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        """Setup the sweeper."""
        self.sweeper.setup(hydra_context=hydra_context, task_function=task_function, config=config)

    def sweep(self, arguments: list[str]) -> None:
        """Perform the sweep."""
        return self.sweeper.sweep(arguments)
