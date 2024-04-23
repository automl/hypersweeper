#!/usr/bin/env python
"""Tests for `hypersweeper` package."""
from __future__ import annotations

from hydra.core.plugins import Plugins
from hydra.plugins.sweeper import Sweeper


def test_discovery() -> None:
    assert "Hypersweeper" in [
        x.__name__ for x in Plugins.instance().discover(Sweeper)
    ], "Hypersweeper not found in Hydra Sweeper plugins"
