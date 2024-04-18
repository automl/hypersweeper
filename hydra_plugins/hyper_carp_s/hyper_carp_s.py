from carps.benchmarks.dummy_problem import DummyProblem
from hydra_plugins.hypersweeper import Info

class HyperCARPSAdapter:
    def __init__(self, carps) -> None:
        self.carps = carps

    def ask(self):
        carps_info = self.carps.ask()
        info = Info(carps_info.config, carps_info.budget, None, carps_info.seed)
        return info, False

    def tell(self, info, value):
        self.smac.tell(info, value)

def make_carp_s(configspace, carps_args):
    problem = DummyProblem()
    problem._configspace = configspace
    optimizer = carps_args["optimizer"](problem)
    return HyperCARPSAdapter(optimizer)