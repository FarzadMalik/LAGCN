"""YAML configuration loading and CLI argument merging."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file and return it as a nested dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    return cfg


def merge_cli_args(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Override *cfg* values with any non-None CLI arguments.

    Only top-level keys that already exist in *cfg* are overwritten.
    """
    cfg = copy.deepcopy(cfg)
    for key, value in vars(args).items():
        if value is not None and key in cfg:
            cfg[key] = value
    return cfg
