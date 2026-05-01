"""Time-series preprocessing utilities for online signature data."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


def normalize_time_series(
    series: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-feature z-score normalisation.

    Parameters
    ----------
    series : ndarray of shape (T, D)
        Raw time-series where T is the number of time-steps and D is the
        feature dimension.
    mean, std : optional pre-computed statistics (D,).  When *None* they are
        computed from *series*.

    Returns
    -------
    normed : ndarray (T, D)
    mean   : ndarray (D,)
    std    : ndarray (D,)
    """
    if mean is None:
        mean = series.mean(axis=0)
    if std is None:
        std = series.std(axis=0)
        std[std < 1e-8] = 1.0
    normed = (series - mean) / std
    return normed, mean, std


def resample_time_series(series: np.ndarray, target_len: int) -> np.ndarray:
    """Resample a variable-length time-series to *target_len* via linear
    interpolation.

    Parameters
    ----------
    series : ndarray (T, D)
    target_len : desired output length

    Returns
    -------
    resampled : ndarray (target_len, D)
    """
    T, D = series.shape
    if T == target_len:
        return series
    t_orig = np.linspace(0, 1, T)
    t_new = np.linspace(0, 1, target_len)
    resampled = np.empty((target_len, D), dtype=np.float32)
    for d in range(D):
        f = interp1d(t_orig, series[:, d], kind="linear")
        resampled[:, d] = f(t_new)
    return resampled


def pad_or_truncate(series: np.ndarray, max_len: int) -> tuple[np.ndarray, int]:
    """Pad with zeros or truncate to *max_len*.

    Returns
    -------
    padded : ndarray (max_len, D)
    actual_length : int
    """
    T, D = series.shape
    actual = min(T, max_len)
    padded = np.zeros((max_len, D), dtype=np.float32)
    padded[:actual] = series[:actual]
    return padded, actual


FEATURE_COLUMNS = {
    "x": 0,
    "y": 1,
    "p": 2,
    "t": 3,
    "u": 4,
}


def select_features(
    raw: np.ndarray, feature_names: list[str]
) -> np.ndarray:
    """Select a subset of columns from the raw MSDS time-series array.

    The MSDS .txt files are assumed to have columns ordered as
    X, Y, P, T, U (pen-up/down).
    """
    cols = [FEATURE_COLUMNS[f] for f in feature_names]
    return raw[:, cols]
