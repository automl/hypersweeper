"""CARP-S optimizer adapter for HyperSweeper."""

from __future__ import annotations

from carps.benchmarks.dummy_problem import DummyProblem

from hydra_plugins.hypersweeper import Info


class HyperCARPSAdapter:
    """CARP-S optimizer."""

    def __init__(self, carps) -> None:
        """Initialize the optimizer."""
        self.carps = carps

    def ask(self):
        """Ask for the next configuration."""
        carps_info = self.carps.ask()
        info = Info(carps_info.config, carps_info.budget, None, carps_info.seed)
        return info, False

    def tell(self, info, value):
        """Tell the result of the configuration."""
        self.smac.tell(info, value)


def make_carp_s(configspace, carps_args):
    """Make a CARP-S instance for optimization."""
    problem = DummyProblem()
    problem._configspace = configspace
    optimizer = carps_args["optimizer"](problem)
    return HyperCARPSAdapter(optimizer)
