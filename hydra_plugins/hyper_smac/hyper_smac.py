"""HyperSMAC implementation."""

from __future__ import annotations

from hydra.utils import get_class
from omegaconf import OmegaConf
from smac import Scenario

from hydra_plugins.hypersweeper import Info

OmegaConf.register_new_resolver("get_class", get_class, replace=True)


class HyperSMACAdapter:
    """Adapt SMAC ask/tell interface to HyperSweeper ask/tell interface."""

    def __init__(self, smac):
        """Initialize the adapter."""
        self.smac = smac

    def ask(self):
        """Ask for the next configuration."""
        smac_info = self.smac.ask()
        info = Info(smac_info.config, smac_info.budget, None, smac_info.seed)
        return info, False

    def tell(self, info, value):
        """Tell the result of the configuration."""
        self.smac.tell(info, value)


def make_smac(configspace, smac_args):
    """Make a SMAC instance for optimization."""

    def dummy_func(arg, seed, budget):  # noqa:ARG001
        return 0.0

    scenario = Scenario(configspace, **smac_args.pop("scenario"))
    if "intensifier" in smac_args:
        intensifier = smac_args["intensifier"](scenario)
        smac = smac_args["smac_facade"](scenario, dummy_func, intensifier=intensifier)
    else:
        smac = smac_args["smac_facade"](scenario, dummy_func)
    return HyperSMACAdapter(smac)