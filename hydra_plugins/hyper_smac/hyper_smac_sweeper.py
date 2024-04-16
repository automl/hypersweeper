"""HyperSMAC implementation."""

from __future__ import annotations

from hydra.utils import get_class
from omegaconf import OmegaConf
from smac import Scenario

from hydra_plugins.hypersweeper import Hypersweeper, Info

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
    smac = smac_args["smac_facade"](scenario, dummy_func)
    return HyperSMACAdapter(smac)


class HyperSMACSweeper(Hypersweeper):
    """Hydra Sweeper for SMAC."""

    def __init__(
        self,
        global_config,
        global_overrides,
        launcher,
        optimizer_kwargs,
        budget_arg_name,
        save_arg_name,
        load_arg_name,
        n_trials,
        cs,
        seeds=False,
        slurm=False,
        slurm_timeout=10,
        max_parallelization=0.1,
        job_array_size_limit=100,
        max_budget=None,
        deterministic=True,
        base_dir=False,
        min_budget=None,
        wandb_project=False,
        wandb_entity=False,
        wandb_tags=None,
        maximize=False,
    ):
        """Initialize the Hypersweeper with SMAC as the optimizer."""
        if wandb_tags is None:
            wandb_tags = ["smac"]
        super().__init__(
            global_config=global_config,
            global_overrides=global_overrides,
            launcher=launcher,
            make_optimizer=make_smac,
            optimizer_kwargs=optimizer_kwargs,
            budget_arg_name=budget_arg_name,
            save_arg_name=save_arg_name,
            load_arg_name=load_arg_name,
            n_trials=n_trials,
            cs=cs,
            seeds=seeds,
            slurm=slurm,
            slurm_timeout=slurm_timeout,
            max_parallelization=max_parallelization,
            job_array_size_limit=job_array_size_limit,
            max_budget=max_budget,
            deterministic=deterministic,
            base_dir=base_dir,
            min_budget=min_budget,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_tags=wandb_tags,
            maximize=maximize,
        )
        self.checkpoint_tf = False
        self.load_tf = False
