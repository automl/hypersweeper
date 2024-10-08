"""
Branin
^^^^^^
"""

import hydra
import numpy as np
from omegaconf import DictConfig

__copyright__ = "Copyright 2022, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


@hydra.main(config_path="configs", config_name="branin_rs", version_base="1.1")
def branin(cfg: DictConfig):
    x0 = cfg.x0
    x1 = cfg.x1
    a = 1.0
    b = 5.1 / (4.0 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)
    ret = a * (x1 - b * x0**2 + c * x0 - r) ** 2 + s * (1 - t) * np.cos(x0) + s

    return ret


if __name__ == "__main__":
    branin()
