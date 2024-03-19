from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from hydra.core.config_store import ConfigStore


@dataclass
class SMACSweeperConfig:
    _target_: str = "hydra_plugins.hydra_smac_sweeper.smac_sweeper.SMACSweeper"
    search_space: Optional[Dict] = field(default_factory=dict)
    resume: Optional[str] = None
    optimizer: Optional[str] = "smac"
    budget: Optional[Any] = None
    budget_variable: Optional[str] = None
    loading_variable: Optional[str] = None
    saving_variable: Optional[str] = None
    smac_kwargs: Optional[Dict] = field(default_factory=dict)


ConfigStore.instance().store(
    group="hydra/sweeper",
    name="SMAC",
    node=SMACSweeperConfig,
    provider="hydra_smac_sweeper",
)
