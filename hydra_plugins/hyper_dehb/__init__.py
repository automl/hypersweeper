import importlib

if (spec := importlib.util.find_spec("dehb")) is not None:
    from .config import HyperDEHBConfig

    __all__ = ["HyperDEHBConfig"]
