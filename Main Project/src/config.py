import os
import yaml
from typing import Any, Dict

class Config:
    """Configuration loader class for parsing project settings from config.yaml."""
    def __init__(self, config_path: str = "project/configs/config.yaml") -> None:
        if not os.path.exists(config_path):
            # Try backup path relative to current dir
            backup_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs/config.yaml")
            if os.path.exists(backup_path):
                config_path = backup_path
            else:
                raise FileNotFoundError(f"Configuration file not found at {config_path}")
                
        with open(config_path, "r") as f:
            self._cfg: Dict[str, Any] = yaml.safe_load(f)

    @property
    def data(self) -> Dict[str, str]:
        return self._cfg.get("data", {})

    @property
    def features(self) -> Dict[str, Any]:
        return self._cfg.get("features", {})

    @property
    def training(self) -> Dict[str, Any]:
        return self._cfg.get("training", {})

    @property
    def models(self) -> Dict[str, Dict[str, Any]]:
        return self._cfg.get("models", {})

    @property
    def outputs(self) -> Dict[str, str]:
        return self._cfg.get("outputs", {})
