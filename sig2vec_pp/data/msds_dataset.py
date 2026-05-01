"""PyTorch datasets for the MSDS-ChS signature dataset.

Supports three modalities:
  * **Online**  – variable-length time-series (X, Y, P, T, U)
  * **Offline** – rendered signature images (PNG)
  * **Multimodal** – both online + offline returned together

The MSDS-ChS directory layout (after extraction) is::

    MSDS-ChS/
      session1/
        0/
          images/   g_0_0.png  f_0_0.png  …
          series/   g_0_0.txt  f_0_0.txt  …
        1/ …
      session2/ …

File naming: ``{flag}_{user}_{index}.{ext}``
  * flag: ``g`` = genuine, ``f`` = skilled forgery
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .preprocessing import (
    normalize_time_series,
    pad_or_truncate,
    select_features,
)


# ─── helpers ────────────────────────────────────────────────────────

def _discover_samples(
    root: Path,
    sessions: Sequence[int],
    user_ids: Sequence[int] | None = None,
) -> List[Dict]:
    """Walk the MSDS directory tree and build a flat list of sample dicts.

    Each dict has keys: ``user``, ``session``, ``flag`` (g/f), ``index``,
    ``series_path``, ``image_path``.
    """
    samples: List[Dict] = []
    for sess in sessions:
        sess_dir = root / f"session{sess}"
        if not sess_dir.is_dir():
            continue
        dirs = sorted(
            [d for d in sess_dir.iterdir() if d.is_dir()],
            key=lambda p: int(p.name),
        )
        for user_dir in dirs:
            uid = int(user_dir.name)
            if user_ids is not None and uid not in user_ids:
                continue
            series_dir = user_dir / "series"
            images_dir = user_dir / "images"
            if series_dir.is_dir():
                for txt in sorted(series_dir.glob("*.txt")):
                    flag, u, idx = txt.stem.split("_")
                    img_path = images_dir / f"{flag}_{u}_{idx}.png"
                    samples.append(
                        {
                            "user": uid,
                            "session": sess,
                            "flag": flag,
                            "index": int(idx),
                            "series_path": str(txt),
                            "image_path": str(img_path) if img_path.exists() else None,
                        }
                    )
            elif images_dir.is_dir():
                for png in sorted(images_dir.glob("*.png")):
                    flag, u, idx = png.stem.split("_")
                    samples.append(
                        {
                            "user": uid,
                            "session": sess,
                            "flag": flag,
                            "index": int(idx),
                            "series_path": None,
                            "image_path": str(png),
                        }
                    )
    return samples


def _load_series(path: str) -> np.ndarray:
    """Load an MSDS ``.txt`` time-series file.

    Expected columns: X  Y  P  T  U  (whitespace-delimited).
    """
    return np.loadtxt(path, dtype=np.float32)


# ─── Online dataset ─────────────────────────────────────────────────

class MSDSOnlineDataset(Dataset):
    """Online (time-series) modality of MSDS-ChS.

    Each sample returns ``(series_tensor, length, label, user_id)`` where
    * *series_tensor* has shape ``(max_seq_len, D)``
    * *length* is the un-padded sequence length
    * *label* is 1 for genuine, 0 for forgery
    * *user_id* is the integer user identifier
    """

    def __init__(
        self,
        root: str | Path,
        sessions: Sequence[int] = (1, 2),
        user_ids: Sequence[int] | None = None,
        feature_names: Sequence[str] = ("x", "y", "p"),
        max_seq_len: int = 800,
        global_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        self.root = Path(root)
        self.feature_names = list(feature_names)
        self.max_seq_len = max_seq_len
        self.global_mean, self.global_std = (None, None) if global_stats is None else global_stats

        self.samples = _discover_samples(self.root, sessions, user_ids)
        self.samples = [s for s in self.samples if s["series_path"] is not None]
        if len(self.samples) == 0:
            raise RuntimeError(f"No online samples found under {self.root}")

    # ── stats ────────────────────────────────────────────────────────

    def compute_global_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute per-feature mean/std over the whole dataset (for z-score)."""
        all_vals: list[np.ndarray] = []
        for s in self.samples:
            raw = _load_series(s["series_path"])
            feats = select_features(raw, self.feature_names)
            all_vals.append(feats)
        concatenated = np.concatenate(all_vals, axis=0)
        self.global_mean = concatenated.mean(axis=0)
        self.global_std = concatenated.std(axis=0)
        self.global_std[self.global_std < 1e-8] = 1.0
        return self.global_mean, self.global_std

    # ── dataset interface ────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        info = self.samples[idx]
        raw = _load_series(info["series_path"])
        feats = select_features(raw, self.feature_names)
        feats, mean, std = normalize_time_series(feats, self.global_mean, self.global_std)
        padded, length = pad_or_truncate(feats, self.max_seq_len)

        series_tensor = torch.from_numpy(padded)  # (max_seq_len, D)
        label = 1 if info["flag"] == "g" else 0
        return series_tensor, length, label, info["user"]


# ─── Offline dataset ────────────────────────────────────────────────

_DEFAULT_IMG_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((128, 256)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)


class MSDSOfflineDataset(Dataset):
    """Offline (rendered-image) modality of MSDS-ChS.

    Returns ``(image_tensor, label, user_id)``.
    """

    def __init__(
        self,
        root: str | Path,
        sessions: Sequence[int] = (1, 2),
        user_ids: Sequence[int] | None = None,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform or _DEFAULT_IMG_TRANSFORM
        self.samples = _discover_samples(self.root, sessions, user_ids)
        self.samples = [s for s in self.samples if s["image_path"] is not None]
        if len(self.samples) == 0:
            raise RuntimeError(f"No offline (image) samples found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        info = self.samples[idx]
        img = Image.open(info["image_path"]).convert("L")
        img_tensor = self.transform(img)
        label = 1 if info["flag"] == "g" else 0
        return img_tensor, label, info["user"]


# ─── Multimodal dataset ─────────────────────────────────────────────

class MSDSMultiModalDataset(Dataset):
    """Joint online + offline dataset.

    Returns ``(series_tensor, length, image_tensor, label, user_id)``.
    Only samples that have *both* modalities are included.
    """

    def __init__(
        self,
        root: str | Path,
        sessions: Sequence[int] = (1, 2),
        user_ids: Sequence[int] | None = None,
        feature_names: Sequence[str] = ("x", "y", "p"),
        max_seq_len: int = 800,
        global_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        image_transform: transforms.Compose | None = None,
    ) -> None:
        self.root = Path(root)
        self.feature_names = list(feature_names)
        self.max_seq_len = max_seq_len
        self.global_mean, self.global_std = (None, None) if global_stats is None else global_stats
        self.image_transform = image_transform or _DEFAULT_IMG_TRANSFORM

        all_samples = _discover_samples(self.root, sessions, user_ids)
        self.samples = [
            s for s in all_samples
            if s["series_path"] is not None and s["image_path"] is not None
        ]
        if len(self.samples) == 0:
            raise RuntimeError(
                f"No multimodal samples (need both series + image) under {self.root}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        info = self.samples[idx]

        raw = _load_series(info["series_path"])
        feats = select_features(raw, self.feature_names)
        feats, _, _ = normalize_time_series(feats, self.global_mean, self.global_std)
        padded, length = pad_or_truncate(feats, self.max_seq_len)
        series_tensor = torch.from_numpy(padded)

        img = Image.open(info["image_path"]).convert("L")
        img_tensor = self.image_transform(img)

        label = 1 if info["flag"] == "g" else 0
        return series_tensor, length, img_tensor, label, info["user"]


# ─── Task-batch sampler (episodic) ──────────────────────────────────

class TaskBatchSampler:
    """Episodic sampler that yields batches structured by task.

    Each task selects one user, then draws
      * 1 anchor genuine
      * *n_genuine* genuine references
      * *n_forgery* skilled forgeries

    The resulting batch contains ``n_tasks * (1 + n_genuine + n_forgery)``
    samples.
    """

    def __init__(
        self,
        dataset: MSDSOnlineDataset,
        n_tasks: int = 4,
        n_genuine: int = 5,
        n_forgery: int = 10,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.n_tasks = n_tasks
        self.n_genuine = n_genuine
        self.n_forgery = n_forgery
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed)

        self.user_genuine: Dict[int, list] = {}
        self.user_forgery: Dict[int, list] = {}
        for i, s in enumerate(dataset.samples):
            uid = s["user"]
            if s["flag"] == "g":
                self.user_genuine.setdefault(uid, []).append(i)
            else:
                self.user_forgery.setdefault(uid, []).append(i)

        self.valid_users = [
            u
            for u in self.user_genuine
            if len(self.user_genuine[u]) >= 1 + n_genuine
            and len(self.user_forgery.get(u, [])) >= n_forgery
        ]
        if len(self.valid_users) == 0:
            raise RuntimeError("No users have enough genuine+forgery samples for the requested task size")
        self._build_epoch()

    def _build_epoch(self):
        users = self.valid_users.copy()
        if self.shuffle:
            self.rng.shuffle(users)
        self.batches: list[list[int]] = []
        for start in range(0, len(users) - self.n_tasks + 1, self.n_tasks):
            batch_users = users[start : start + self.n_tasks]
            indices: list[int] = []
            for u in batch_users:
                g_pool = self.user_genuine[u]
                f_pool = self.user_forgery[u]
                chosen_g = self.rng.choice(g_pool, 1 + self.n_genuine, replace=False)
                chosen_f = self.rng.choice(f_pool, self.n_forgery, replace=False)
                indices.append(int(chosen_g[0]))  # anchor
                indices.extend(chosen_g[1:].tolist())
                indices.extend(chosen_f.tolist())
            self.batches.append(indices)

    def __iter__(self):
        self._build_epoch()
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def collate_online(batch):
    """Custom collate for :class:`MSDSOnlineDataset`.

    Returns numpy arrays matching the original Sig2Vec training loop
    interface: ``(sigs, lengths, labels)``.
    """
    sigs, lengths, labels, users = zip(*batch)
    sigs = np.stack([s.numpy() for s in sigs], axis=0).astype(np.float32)
    lengths = np.array(lengths, dtype=np.int64)
    labels = np.array(labels, dtype=np.int64)
    return sigs, lengths, labels
