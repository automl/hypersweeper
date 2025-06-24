"""CARP-S optimizer adapter for HyperSweeper."""

from __future__ import annotations

from hydra_plugins.hypersweeper import Info
from smac.runhistory.dataclasses import TrialInfo, TrialValue


class HyperCARPSAdapter:
    """CARP-S optimizer."""

    def __init__(self, carps) -> None:
        """Initialize the optimizer."""
        self.carps = carps

    def ask(self):
        """Ask for the next configuration."""
        carps_info = self.carps.ask()
        info = Info(carps_info.config, carps_info.budget, None, carps_info.seed)
        return info, False, False

    def tell(self, info, value):
        """Tell the result of the configuration."""
        smac_info = TrialInfo(info.config, seed=info.seed, budget=info.budget)
        smac_value = TrialValue(time=value.cost, cost=value.performance)
        self.carps.tell(smac_info, smac_value)

    def finish_run(self, output_path):
        """Do nothing for CARPS."""


def make_carp_s(configspace, optimizer_kwargs):
    """Make a CARP-S instance for optimization."""
    task = optimizer_kwargs["task_config"]
    task.objective_function.configuration_space = configspace
    task.input_space.configuration_space = configspace
    optimizer = optimizer_kwargs["optimizer"](task, optimizer_kwargs["optimizer_config"])
    optimizer.solver = optimizer._setup_optimizer()
    return HyperCARPSAdapter(optimizer)
