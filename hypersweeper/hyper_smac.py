from __future__ import annotations

from hypersweeper import HyperSweeper


class HyperSMACAdapter:
    def __init__(self, smac):
        self.smac = smac

    def ask(self):
        info = self.smac.ask()
        return info, False

def make_smac(smac_args):
    raise NotImplementedError("Please implement this function")

class HyperSMAC(HyperSweeper):
    def __init__(
        self,
        global_config,
        global_overrides,
        launcher,
        optimizer_kwargs,
        budget_arg_name,
        save_arg_name,
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
        if wandb_tags is None:
            wandb_tags = ["smac"]
        super().__init__(
            global_config,
            global_overrides,
            launcher,
            make_smac,
            optimizer_kwargs,
            budget_arg_name,
            save_arg_name,
            n_trials,
            cs,
            seeds,
            slurm,
            slurm_timeout,
            max_parallelization,
            job_array_size_limit,
            max_budget,
            deterministic,
            base_dir,
            min_budget,
            wandb_project,
            wandb_entity,
            wandb_tags,
            maximize,
        )