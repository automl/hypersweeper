import importlib

if (spec := importlib.util.find_spec("neps")) is not None:
    from .config import HyperNEPSConfig

    __all__ = ["HyperNEPSConfig"]
