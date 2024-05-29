"""HyperDEHB sweeper."""

from __future__ import annotations

from hydra_plugins.hypersweeper import Info


class HyperDEHB:
    """DEHB."""

    def __init__(self, configspace, dehb) -> None:
        """Initialize the optimizer."""
        self.configspace = configspace
        self.dehb = dehb
        self.storage = {}

    def ask(self):
        """Randomly sample a configuration."""
        job_info = self.dehb.ask()
        info = Info(job_info["config"], job_info["fidelity"], None, None)
        self.storage[job_info["config_id"]] = job_info
        return info, False

    def tell(self, info, value):
        """Return the performance."""
        config_id = next(k for k, v in self.storage.items() if v["config"] == info.config)
        job_info = self.storage[config_id]
        job_return = {"fitness": value.performance, "cost": value.cost}
        self.dehb.tell(job_info, job_return)


def make_dehb(configspace, hyper_dehb_args):
    """Make a DEHB instance for optimization."""
    dimensions = len(configspace.get_hyperparameters())
    dehb = hyper_dehb_args(dimensions=dimensions, cs=configspace)
    return HyperDEHB(configspace, dehb)
