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
    smac_kwargs = {}

    if "callbacks" not in smac_args:
        smac_kwargs["callbacks"] = []
    elif "callbacks" in smac_args and isinstance(smac_args["callbacks"], dict):
        smac_kwargs["callbacks"] = list(smac_args["callbacks"].values())
    elif "callbacks" in smac_args and isinstance(smac_args["callbacks"], list):
        smac_kwargs["callbacks"] = smac_args["callbacks"]

    if "acquisition_function" in smac_args and "acquisition_maximizer" in smac_args:
        smac_kwargs["acquisition_maximizer"] = smac_args["acquisition_maximizer"](
            configspace=configspace,
            acquisition_function=smac_args["acquisition_function"],
        )
        if hasattr(smac_args["acquisition_maximizer"], "selector") and hasattr(
            smac_args["acquisition_maximizer"].selector, "expl2callback"
        ):
            smac_kwargs["callbacks"].append(
                smac_args["acquisition_maximizer"].selector.expl2callback
            )

    if "config_selector" in smac_args:
        smac_kwargs["config_selector"] = smac_args["config_selector"](scenario=scenario)

    if "initial_design" in smac_args:
        smac_kwargs["initial_design"] = smac_args["initial_design"](scenario=scenario)

    if "intensifier" in smac_args:
        smac_kwargs["intensifier"] = smac_args["intensifier"](scenario)

    smac = smac_args["smac_facade"](scenario, dummy_func, **smac_kwargs)
    return HyperSMACAdapter(smac)
