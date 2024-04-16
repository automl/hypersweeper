#!/usr/bin/env python
"""Tests for `hypersweeper` package."""

from hydra.core.plugins import Plugins
from hydra.plugins.sweeper import Sweeper

from hydra_plugins.hyper_pbt import HyperPBT
from hydra_plugins.hyper_smac import HyperSMAC


def test_smac_discovery() -> None:
    assert HyperSMAC.__name__ in [
        x.__name__ for x in Plugins.instance().discover(Sweeper)
    ], "HyperSMAC not found in Hydra Sweeper plugins"

def test_pbt_discovery() -> None:
    assert HyperPBT.__name__ in [
        x.__name__ for x in Plugins.instance().discover(Sweeper)
    ], "HyperPBT not found in Hydra Sweeper plugins"