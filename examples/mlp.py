"""
MLP with Multi-Fidelity
^^^^^^^^^^^^^^^^^^^^^^^

Example for optimizing a Multi-Layer Perceptron (MLP) using multiple budgets.
Since we want to take advantage of Multi-Fidelity, the SMAC4MF facade is a good choice. By default,
SMAC4MF internally runs with `hyperband <https://arxiv.org/abs/1603.06560>`_, which is a combination of an
aggressive racing mechanism and successive halving.

MLP is a deep neural network, and therefore, we choose epochs as fidelity type. The digits dataset
is chosen to optimize the average accuracy on 5-fold cross validation.

This example is adapted from `<https://github.com/automl/SMAC3/blob/main/examples/2_multi_fidelity/1_mlp_epochs.py>`_.
"""

__copyright__ = "Copyright 2022, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


import warnings

import hydra
import numpy as np
from omegaconf import DictConfig
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier

digits = load_digits()


# Target Algorithm
@hydra.main(config_path="configs", config_name="mlp_smac", version_base="1.1")
def mlp_from_cfg(cfg: DictConfig):
    """
    Creates a MLP classifier from sklearn and fits the given data on it.

    Parameters
    ----------
    cfg: Configuration
        configuration chosen by smac

    Returns
    -------
    float
    """
    # For deactivated parameters, the configuration stores None-values.
    # This is not accepted by the MLP, so we replace them with placeholder values.
    lr = cfg.learning_rate or "constant"
    lr_init = cfg.learning_rate_init or 0.001
    batch_size = cfg.batch_size or 200

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        mlp = MLPClassifier(
            hidden_layer_sizes=[cfg.n_neurons] * cfg.n_layer,
            solver=cfg.solver,
            batch_size=batch_size,
            activation=cfg.activation,
            learning_rate=lr,
            learning_rate_init=lr_init,
            max_iter=int(np.ceil(cfg.epochs)),
            random_state=cfg.seed,
        )

        # returns the cross validation accuracy
        cv = StratifiedKFold(n_splits=5, random_state=cfg.seed, shuffle=True)  # to make CV splits consistent
        score = cross_val_score(mlp, digits.data, digits.target, cv=cv, error_score="raise")

    return 1 - np.mean(score)


if __name__ == "__main__":
    mlp_from_cfg()
