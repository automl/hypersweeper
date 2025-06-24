# FIXME: this is disabled since due to low configspace version, neps is not compatible with hypersweeper
# Should be reactivated once the configspace version in neps is updated


# import shutil
# import subprocess
# from pathlib import Path

# import pandas as pd


# def test_neps_asha_mlp_example():
#     if Path("mlp_neps_asha").exists():
#         shutil.rmtree(Path("mlp_neps_asha"))
#     subprocess.call(["python", "examples/mlp.py", "--config-name=mlp_neps_asha", "-m", "hydra.sweeper.n_trials=5", "hydra.run.dir=mlp_neps_asha", "hydra.sweep.dir=mlp_neps_asha"])
#     assert Path("mlp_neps_asha").exists(), "Run directory not created"
#     assert Path("mlp_neps_asha/runhistory.csv").exists(), "Run history file not created"
#     runhistory = pd.read_csv("mlp_neps_asha/runhistory.csv")
#     assert not runhistory.empty, "Run history is empty"
#     assert len(runhistory) == 5, "Run history should contain 5 entries"
#     shutil.rmtree(Path("mlp_neps_asha"))

# def test_neps_priorband_mlp_example():
#     if Path("mlp_neps_priorband").exists():
#         shutil.rmtree(Path("mlp_neps_priorband"))
#     subprocess.call(["python", "examples/mlp.py", "--config-name=mlp_neps_priorband", "-m", "hydra.sweeper.n_trials=5", "hydra.run.dir=mlp_neps_asha", "hydra.sweep.dir=mlp_neps_asha"])
#     assert Path("mlp_neps_priorband").exists(), "Run directory not created"
#     assert Path("mlp_neps_priorband/runhistory.csv").exists(), "Run history file not created"
#     runhistory = pd.read_csv("mlp_neps_priorband/runhistory.csv")
#     assert not runhistory.empty, "Run history is empty"
#     assert len(runhistory) == 5, "Run history should contain 5 entries"
#     shutil.rmtree(Path("mlp_neps_priorband"))
