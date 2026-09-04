"""Load the project's central configuration file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load ``config.yaml`` into a plain dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
