import importlib

if (spec := importlib.util.find_spec("hebo")) is not None:
    from .config import HyperHEBOConfig

    __all__ = ["HyperHEBOConfig"]
