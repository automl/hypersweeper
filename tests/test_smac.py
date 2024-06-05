from __future__ import annotations

from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer
from omegaconf import OmegaConf

BBO_CONFIG = {
    "smac_facade": OmegaConf.create(
        {
            "_target_": "smac.facade.blackbox_facade.BlackBoxFacade",
            "_partial_": True,
            "logging_level": 20,
        }
    ),
    "scenario": {
        "seed": 42,
        "n_trials": 10,
        "deterministic": True,
        "n_workers": 4,
        "name": "bb_test",
    },
}

MF_CONFIG = {
    "smac_facade": OmegaConf.create(
        {
            "_target_": "smac.facade.multi_fidelity_facade.MultiFidelityFacade",
            "_partial_": True,
            "logging_level": 20,
        }
    ),
    "intensifier": OmegaConf.create(
        {
            "_target_": "smac.facade.multi_fidelity_facade.MultiFidelityFacade.get_intensifier",
            "_partial_": True,
            "eta": 3,
        }
    ),
    "scenario": {
        "seed": 42,
        "n_trials": 10,
        "deterministic": True,
        "n_workers": 1,
        "min_budget": 5,
        "max_budget": 50,
        "name": "mf_test",
    },
}

DEFAULT_CONFIG_SPACE = ConfigurationSpace(
    {
        Float("a", bounds=[0.0, 1.0]),
        Integer("b", bounds=[1, 10]),
        Categorical("c", ["a", "b", "c"]),
    }
)

# ISSUE HERE: scenario saving doesn't work for some reason with this config.
# Not sure why and if this is a SMAC issue or not.


# class TestHyperSMAC:
#     def setup(self, mf=False):
#         configspace = DEFAULT_CONFIG_SPACE
#         hyper_smac_args = deepcopy(BBO_CONFIG) if not mf else deepcopy(MF_CONFIG)
#         hyper_smac_args["intensifier"] =
#                instantiate(hyper_smac_args["intensifier"]) if mf else None
#         hyper_smac_args["smac_facade"] = instantiate(hyper_smac_args["smac_facade"])
#         return make_smac(configspace, hyper_smac_args)

#     def test_init(self):
#         hyper_smac = self.setup()
#         assert hyper_smac.smac is not None, "SMAC is not initialized"

#         hyper_smac_mf = self.setup(mf=True)
#         assert hyper_smac_mf.smac is not None, "SMAC is not initialized"

#     def test_ask(self):
#         hyper_smac = self.setup(mf=True)
#         info, _ = hyper_smac.ask()
#         assert info.config is not None, "Configuration is not generated"
#         assert isinstance(info, Info), "Return value is not an Info object"
#         assert info.budget is not None, "Budget is not generated"
#         assert info.load_path is None, "Load path is set"

#     def tell(self):
#         hyper_smac = self.setup()
#         info, _ = hyper_smac.ask()
#         result = Result(0.0, 0.0)
#         hyper_smac.tell(info, result)

#         hyper_smac = self.setup(mf=True)
#         info, _ = hyper_smac.ask()
#         result = Result(0.0, 0.0)
#         hyper_smac.tell(info, result)
