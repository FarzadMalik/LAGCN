"""Verification metrics: EER, DET curves, FNMR@FMR thresholds."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d


def compute_det_curve(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    n_thresholds: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Detection-Error Tradeoff (DET) curve.

    Parameters
    ----------
    genuine_scores : (N_g,) similarity scores for genuine pairs
    impostor_scores : (N_i,) similarity scores for impostor pairs
    n_thresholds : resolution of the threshold sweep

    Returns
    -------
    fmr  : (n_thresholds,) False Match Rate
    fnmr : (n_thresholds,) False Non-Match Rate
    thresholds : (n_thresholds,)
    """
    all_scores = np.concatenate([genuine_scores, impostor_scores])
    lo, hi = float(all_scores.min()), float(all_scores.max())
    thresholds = np.linspace(lo, hi, n_thresholds)

    fmr = np.empty(n_thresholds, dtype=np.float64)
    fnmr = np.empty(n_thresholds, dtype=np.float64)

    n_gen = len(genuine_scores)
    n_imp = len(impostor_scores)

    for i, t in enumerate(thresholds):
        fmr[i] = np.sum(impostor_scores >= t) / n_imp
        fnmr[i] = np.sum(genuine_scores < t) / n_gen

    return fmr, fnmr, thresholds


def compute_eer(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
) -> float:
    """Equal Error Rate — the operating point where FMR == FNMR."""
    fmr, fnmr, _ = compute_det_curve(genuine_scores, impostor_scores)
    try:
        eer = brentq(lambda x: interp1d(fmr, fnmr)(x) - x, fmr.min(), fmr.max())
    except ValueError:
        idx = np.argmin(np.abs(fmr - fnmr))
        eer = float((fmr[idx] + fnmr[idx]) / 2)
    return float(eer)


def compute_fnmr_at_fmr(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    target_fmr: float = 0.01,
) -> float:
    """FNMR at a given FMR operating point."""
    fmr, fnmr, _ = compute_det_curve(genuine_scores, impostor_scores)
    valid = fmr <= target_fmr
    if not np.any(valid):
        return 1.0
    idx = np.where(valid)[0][np.argmax(fmr[valid])]
    return float(fnmr[idx])
