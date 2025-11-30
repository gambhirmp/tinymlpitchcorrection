#!/usr/bin/env python3
"""
Plot pitch (f0 in cents) before and after correction for a subset of clips.
- Selects the first N clips in val/test (prefers originals; excludes augmented files with '_ps').
- Uses SavedModel to predict per-frame shift_cents and computes corrected f0 as f0_cents + shift.
- Saves PNG plots under artifacts/plots/{clip_id}.png
"""
from pathlib import Path
import csv
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

ROOT = Path("/Users/mayagambhir/3600_final")
PROC = ROOT / "data/processed/features"
META = ROOT / "metadata"
OUTP = ROOT / "artifacts/plots"
OUTP.mkdir(parents=True, exist_ok=True)


def load_feature_norm():
    with open(META / "feature_norm.json", "r") as f:
        norm = json.load(f)
    mu = np.array(norm["feature_mean"] or [0.0] * 64, np.float32)
    sd = np.array(norm["feature_std"] or [1.0] * 64, np.float32)
    fps = int(norm.get("frames_per_second", 100))
    return mu, sd, fps


def load_clip_map():
    """Return dict clip_id -> path_rel from metadata/clips.csv."""
    m = {}
    csv_path = META / "clips.csv"
    if not csv_path.exists():
        return m
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            m[row["clip_id"]] = row["path_rel"]
    return m


def list_all_npz():
    files = []
    for split in ["val", "test", "train"]:
        files.extend(glob.glob(str(PROC / split / "*.npz")))
    return sorted(files)


def main():
    mu, sd, fps = load_feature_norm()
    clip_map = load_clip_map()
    serve = tf.saved_model.load(str(ROOT / "artifacts/baseline/saved_model"))

    candidates = []
    for npz_path in list_all_npz():
        clip_id = Path(npz_path).stem
        raw_rel = clip_map.get(clip_id, "")
        # Prefer original takes (exclude augmented with '_ps')
        if "_ps" in raw_rel:
            continue
        candidates.append((clip_id, npz_path))
    if not candidates:
        print("No feature files found to plot.")
        return

    # Limit to first N to avoid too many plots
    MAX_PLOTS = 12
    for clip_id, npz_path in candidates[:MAX_PLOTS]:
        arr = np.load(npz_path)
        f0_cents = arr["f0_cents"].astype(np.float32)
        voiced = (arr["voiced"].astype(np.float32) > 0.5)
        X = (arr["logmel"].astype(np.float32) - mu[None, :]) / (sd[None, :] + 1e-8)
        pred = serve.serve(tf.convert_to_tensor(X[None, ...], tf.float32))["shift_cents"].numpy()[0, :, 0]
        # Compute corrected f0 in cents (simple sum for visualization)
        f0_corr = f0_cents + pred
        T = len(f0_cents)
        t = np.arange(T) / float(fps)

        plt.figure(figsize=(10, 3))
        # Plot only voiced frames to reduce clutter
        plt.plot(t[voiced], f0_cents[voiced], label="Original f0 (cents)", alpha=0.7)
        plt.plot(t[voiced], f0_corr[voiced], label="Corrected f0 (cents)", alpha=0.8)
        plt.xlabel("Time (s)")
        plt.ylabel("Pitch (cents vs A4=440)")
        plt.title(f"{clip_id} - Maya (before vs after)")
        plt.legend(loc="best")
        plt.tight_layout()
        out_png = OUTP / f"{clip_id}.png"
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"Wrote plot: {out_png}")


if __name__ == "__main__":
    main()


