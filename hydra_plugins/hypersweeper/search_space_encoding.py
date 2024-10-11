"""Search space encoding between hydra and configspace."""

from __future__ import annotations

import json
from io import StringIO

from ConfigSpace import ConfigurationSpace
from omegaconf import DictConfig, ListConfig


class JSONCfgEncoder(json.JSONEncoder):
    """Encode DictConfigs.

    Convert DictConfigs to normal dicts.
    """

    def default(self, obj):
        """Convert DictConfigs to normal dicts."""
        if isinstance(obj, DictConfig):
            return dict(obj)
        if isinstance(obj, ListConfig):
            parsed_list = []
            for o in obj:
                parsed = o
                if isinstance(o, DictConfig):
                    parsed = dict(parsed)
                elif isinstance(o, ListConfig):
                    parsed = list(parsed)
                parsed_list.append(parsed)

            return parsed_list  # [dict(o) for o in obj]
        return json.JSONEncoder.default(self, obj)


def search_space_to_config_space(search_space: str | DictConfig) -> ConfigurationSpace:
    """Convert hydra search space to SMAC's configuration space.

    See the [ConfigSpace docs](https://automl.github.io/ConfigSpace/master/API-Doc.html#)
    for information of how to define a configuration (search) space.

    In a yaml (hydra) config file, the smac.search space must take the form of:

    search_space:
        hyperparameters:
            hyperparameter_name_0:
                key1: value1
                ...
            hyperparameter_name_1:
                key1: value1
                key2: value2
                ...


    Parameters
    ----------
    search_space : Union[str, DictConfig, ConfigurationSpace]
        The search space, either a DictConfig from a hydra yaml config file,
        or a path to a json configuration space file
        in the format required of ConfigSpace.
        If it already is a ConfigurationSpace, just optionally seed it.
    seed : Optional[int]
        Optional seed to seed configuration space.


    Example of a json-serialized ConfigurationSpace file.
    {
      "hyperparameters": [
        {
          "name": "x0",
          "type": "uniform_float",
          "log": false,
          "lower": -512.0,
          "upper": 512.0,
          "default": -3.0
        },
        {
          "name": "x1",
          "type": "uniform_float",
          "log": false,
          "lower": -512.0,
          "upper": 512.0,
          "default": -4.0
        }
      ],
      "conditions": [],
      "forbiddens": [],
      "python_module_version": "0.4.17",
      "json_format_version": 0.2
    }


    Returns:
    -------
    ConfigurationSpace
    """
    if isinstance(search_space, str):
        with open(search_space) as f:
            jason_string = f.read()
        cs = ConfigurationSpace.from_json(StringIO(jason_string))
    elif isinstance(search_space, DictConfig):
        # reorder hyperparameters as List[Dict]
        hyperparameters = []
        for name, cfg in search_space.hyperparameters.items():
            cfg["name"] = name
            hyperparameters.append(cfg)
        search_space.hyperparameters = hyperparameters

        jason_string = json.dumps(search_space, cls=JSONCfgEncoder)
        cs = ConfigurationSpace.from_json(StringIO(jason_string))
    elif type(search_space) is ConfigurationSpace:
        cs = search_space
    else:
        raise ValueError(f"search_space must be of type str or DictConfig. Got {type(search_space)}.")

    if "seed" in search_space:
        cs.seed(seed=search_space.seed)

    return cs
