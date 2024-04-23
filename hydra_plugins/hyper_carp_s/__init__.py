import importlib

if (spec := importlib.util.find_spec("carps")) is not None:
    from .config import HyperCARPSConfig

    __all__ = ["HyperCARPSConfig"]
