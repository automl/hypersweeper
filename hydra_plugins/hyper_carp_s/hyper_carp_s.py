"""CARP-S optimizer adapter for HyperSweeper."""

from __future__ import annotations

import importlib

if (spec := importlib.util.find_spec("carps")) is not None:
    from carps.benchmarks.dummy_problem import DummyProblem

from smac.runhistory.dataclasses import TrialInfo, TrialValue

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
        smac_info = TrialInfo(info.config, seed=info.seed, budget=info.budget)
        smac_value = TrialValue(time=value.cost, cost=value.performance)
        self.carps.tell(smac_info, smac_value)


def make_carp_s(configspace, carps_args):
    """Make a CARP-S instance for optimization."""
    problem = DummyProblem()
    problem._configspace = configspace
    optimizer = carps_args(problem)
    optimizer.solver = optimizer._setup_optimizer()
    return HyperCARPSAdapter(optimizer)
