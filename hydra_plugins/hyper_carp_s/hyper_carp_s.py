"""CARP-S optimizer adapter for HyperSweeper."""

from __future__ import annotations

from carps.objective_functions.dummy_problem import DummyObjectiveFunction
from carps.utils.task import FidelitySpace, InputSpace, OptimizationResources, OutputSpace, Task, TaskMetadata
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
    objective_function = DummyObjectiveFunction(configuration_space=configspace, return_value=42)
    fidelity_space = FidelitySpace(**optimizer_kwargs["task"]["fidelity_kwargs"])
    input_space = InputSpace(configuration_space=configspace, fidelity_space=fidelity_space)
    output_space = OutputSpace(**optimizer_kwargs["task"]["output_kwargs"])
    optimization_resources = OptimizationResources(**optimizer_kwargs["task"]["resource_kwargs"])
    metadata = TaskMetadata(**optimizer_kwargs["task"]["metadata_kwargs"])
    task = Task(
        objective_function=objective_function,
        input_space=input_space,
        output_space=output_space,
        optimization_resources=optimization_resources,
        metadata=metadata,
        seed=optimizer_kwargs.get("seed", 42),
        name=optimizer_kwargs.get("task_name", "default_task"),
    )
    optimizer = optimizer_kwargs["optimizer"](task, optimizer_kwargs["optimizer_cfg"])
    optimizer.solver = optimizer._setup_optimizer()
    return HyperCARPSAdapter(optimizer)
