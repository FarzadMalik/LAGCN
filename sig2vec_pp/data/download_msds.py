#!/usr/bin/env python
"""Download helper and preprocessing pipeline for the MSDS-ChS dataset.

**The MSDS dataset requires an approved application.**  This script cannot
automatically download it — instead it validates the local directory layout
and runs any needed preprocessing.

How to obtain MSDS
------------------
1. Visit https://github.com/HCIILAB/MSDS
2. Download and fill in the Application Form and Legal Commitment.
3. Submit via the SCUT DLVC Lab portal:
       http://121.41.49.212:9000/apply/msds
4. After approval (typically 3–5 business days) you will receive a
   download link and decompression password.
5. Extract the archive so the layout matches::

       data/MSDS/
         MSDS-ChS/
           session1/
             0/
               images/  g_0_0.png …
               series/  g_0_0.txt …
             1/ …
           session2/ …
         MSDS-TDS/  (optional)

Then run this script to verify and preprocess::

    python -m sig2vec_pp.data.download_msds --root data/MSDS/MSDS-ChS
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def verify_layout(root: Path) -> bool:
    """Check that the MSDS-ChS directory looks correct."""
    ok = True
    for sess in [1, 2]:
        sess_dir = root / f"session{sess}"
        if not sess_dir.is_dir():
            print(f"[WARN] Missing directory: {sess_dir}")
            ok = False
            continue
        user_dirs = sorted(
            [d for d in sess_dir.iterdir() if d.is_dir()], key=lambda p: int(p.name)
        )
        if len(user_dirs) == 0:
            print(f"[WARN] No user directories in {sess_dir}")
            ok = False
        else:
            print(f"[OK]   session{sess}: {len(user_dirs)} users")
            sample_user = user_dirs[0]
            series = list((sample_user / "series").glob("*.txt")) if (sample_user / "series").is_dir() else []
            images = list((sample_user / "images").glob("*.png")) if (sample_user / "images").is_dir() else []
            print(f"       sample user {sample_user.name}: {len(series)} series, {len(images)} images")
    return ok


def compute_dataset_stats(root: Path, feature_names=("x", "y", "p")):
    """Compute and cache global mean/std for the time-series features."""
    from sig2vec_pp.data.preprocessing import select_features

    feature_col_map = {"x": 0, "y": 1, "p": 2, "t": 3, "u": 4}
    cols = [feature_col_map[f] for f in feature_names]

    all_vals = []
    for sess in [1, 2]:
        sess_dir = root / f"session{sess}"
        if not sess_dir.is_dir():
            continue
        for user_dir in sorted(sess_dir.iterdir()):
            series_dir = user_dir / "series"
            if not series_dir.is_dir():
                continue
            for txt in sorted(series_dir.glob("*.txt")):
                data = np.loadtxt(txt, dtype=np.float32)
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                all_vals.append(data[:, cols])

    if not all_vals:
        print("[WARN] No series files found — cannot compute stats.")
        return

    cat = np.concatenate(all_vals, axis=0)
    mean = cat.mean(axis=0)
    std = cat.std(axis=0)
    std[std < 1e-8] = 1.0

    out = root / "global_stats.npz"
    np.savez(out, mean=mean, std=std, features=list(feature_names))
    print(f"[OK]   Saved global stats to {out}")
    print(f"       mean = {mean}")
    print(f"       std  = {std}")


def main():
    parser = argparse.ArgumentParser(description="Verify / preprocess MSDS-ChS")
    parser.add_argument("--root", type=str, default="data/MSDS/MSDS-ChS")
    parser.add_argument("--compute-stats", action="store_true", default=True)
    parser.add_argument("--features", nargs="+", default=["x", "y", "p"])
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"[ERROR] Root directory not found: {root}")
        print()
        print("The MSDS dataset must be obtained via application.")
        print("See: https://github.com/HCIILAB/MSDS")
        sys.exit(1)

    print(f"Verifying MSDS-ChS layout at: {root}\n")
    ok = verify_layout(root)

    if ok and args.compute_stats:
        print("\nComputing global statistics ...")
        compute_dataset_stats(root, feature_names=args.features)

    if ok:
        print("\n[DONE] Dataset is ready for training.")
    else:
        print("\n[WARN] Some issues found — please check the directory layout.")


if __name__ == "__main__":
    main()
