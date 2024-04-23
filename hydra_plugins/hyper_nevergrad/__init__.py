import importlib

if (spec := importlib.util.find_spec("nevergrad")) is not None:
    from .config import HyperNevergradConfig

    __all__ = ["HyperNevergradConfig"]
