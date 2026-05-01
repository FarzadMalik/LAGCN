"""Lightweight logging helpers (console + optional W&B)."""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, Optional


def setup_logger(
    name: str = "sig2vec_pp",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Create a logger that writes to *stderr* and optionally to a file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "[%(asctime)s %(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def log_metrics(
    metrics: Dict[str, Any],
    step: int,
    logger: Optional[logging.Logger] = None,
    use_wandb: bool = False,
) -> None:
    """Log a dict of metrics to the console and optionally to W&B."""
    if logger is None:
        logger = setup_logger()
    parts = [f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()]
    logger.info("step=%d  %s", step, "  ".join(parts))

    if use_wandb:
        try:
            import wandb  # noqa: F811

            wandb.log(metrics, step=step)
        except ImportError:
            logger.warning("wandb not installed — skipping remote logging")
