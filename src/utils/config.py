from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "config.yaml"

DEFAULTS: Dict[str, Any] = {
    "empty_threshold": 5,
    "full_threshold": 2,
    "decision_threshold": 0.60,
    "horizon_shifts": 6,
    "horizon_minutes": 30,
    "train_fraction": 0.8,
    "empty_threshold_candidates": [2, 3, 4, 5],
}


def load_config() -> Dict[str, Any]:
    """Load config.yaml with safe defaults."""
    if not CONFIG_PATH.exists():
        return DEFAULTS.copy()

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    config = DEFAULTS.copy()
    config.update(data)
    return config
