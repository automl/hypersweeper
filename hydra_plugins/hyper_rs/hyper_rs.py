"""HyperRS: Random Search in Hypersweeper."""

from __future__ import annotations

from hydra_plugins.hypersweeper import Info


class HyperRS:
    """Random Search."""

    def __init__(self, configspace) -> None:
        """Initialize the optimizer."""
        self.configspace = configspace

    def ask(self):
        """Randomly sample a configuration."""
        config = self.configspace.sample_configuration()
        info = Info(config, None, None, None)
        return info, False

    def tell(self, info, value):
        """Do nothing for RS."""


def make_rs(configspace, hyper_rs_args):
    """Make a RS instance for optimization."""
    return HyperRS(configspace, **hyper_rs_args)
