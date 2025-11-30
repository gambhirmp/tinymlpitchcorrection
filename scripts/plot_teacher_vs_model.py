#!/usr/bin/env python3
"""
Plot original pitch vs teacher-corrected vs model-corrected for a subset of clips.
Bottom panel shows:
  - raw f0 (pyin)              : blue (thin)
  - original trend (low-pass)  : gray dashed
  - teacher-corrected trend    : green
  - model-corrected trend      : orange
Also prints per-clip MAE (cents) between teacher- and model-corrected trends.
Saves to artifacts/plots/teacher_vs_model_{clip_id}.png
"""
from pathlib import Path
import csv
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
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
    sr = int(norm.get("sample_rate_hz", 16000))
    return mu, sd, fps, sr


def load_clip_map():
    m = {}
    p = META / "clips.csv"
    if not p.exists():
        return m
    with open(p, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            m[row["clip_id"]] = str((ROOT / row["path_rel"]).resolve())
    return m


def lowpass_exponential(x: np.ndarray, cutoff_hz: float, fps: int) -> np.ndarray:
    dt = 1.0 / fps
    rc = 1.0 / (2.0 * np.pi * max(cutoff_hz, 1e-3))
    alpha = dt / (rc + dt)
    y = np.zeros_like(x, dtype=np.float32)
    prev = x[0]
    for i in range(len(x)):
        prev = prev + alpha * (x[i] - prev)
        y[i] = prev
    return y


def list_npz_subset(limit=8):
    paths = []
    for split in ["val", "test", "train"]:
        paths.extend(glob.glob(str(PROC / split / "*.npz")))
    return sorted(paths)[:limit]


def main():
    mu, sd, fps, sr = load_feature_norm()
    clip_map = load_clip_map()
    serve = tf.saved_model.load(str(ROOT / "artifacts/baseline/saved_model"))

    for npz_path in list_npz_subset(limit=10):
        npz_path = Path(npz_path)
        clip_id = npz_path.stem
        raw_path = clip_map.get(clip_id)
        # Only originals: skip augmented files that contain '_ps' in the raw path
        raw_rel = None
        for k, v in clip_map.items():
            if k == clip_id:
                raw_rel = v
                break
        if raw_rel and "_ps" in raw_rel:
            continue
        if not raw_path or not Path(raw_path).exists():
            print(f"[WARN] missing raw audio for {clip_id}, skipping.")
            continue

        arr = np.load(npz_path)
        f0 = arr["f0_cents"].astype(np.float32)
        teacher_shift = arr["target_shift"].astype(np.float32)
        X = (arr["logmel"].astype(np.float32) - mu[None, :]) / (sd[None, :] + 1e-8)
        model_shift = serve.serve(tf.convert_to_tensor(X[None, ...], tf.float32))["shift_cents"].numpy()[0, :, 0]

        trend = lowpass_exponential(f0, cutoff_hz=4.0, fps=fps)
        teacher_corr = trend + teacher_shift
        model_corr = trend + model_shift

        mae_cents = float(np.mean(np.abs(teacher_corr - model_corr)))

        y, _ = sf.read(raw_path)
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = y / (np.max(np.abs(y)) + 1e-9)
        t_audio = np.arange(len(y)) / sr
        t_frames = np.arange(len(f0)) / float(fps)

        fig = plt.figure(figsize=(12, 5))
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(t_audio, y, color="#333", linewidth=0.8)
        ax1.set_title(f"{clip_id} – waveform (normalized)")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")

        ax2 = plt.subplot(2, 1, 2)
        ax2.plot(t_frames, f0, label="raw f0 (pyin)", color="#1f77b4", alpha=0.5, linewidth=0.8)
        ax2.plot(t_frames, trend, label="trend (low-pass)", color="#888", linestyle="--", linewidth=1.2)
        ax2.plot(t_frames, teacher_corr, label="teacher-corrected trend", color="#2ca02c", linewidth=2.0)
        ax2.plot(t_frames, model_corr, label="model-corrected trend", color="#ff7f0e", linewidth=2.0, alpha=0.9)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Pitch (cents vs A4=440)")
        ax2.legend(loc="best", ncol=2)
        ax2.set_title(f"Teacher vs Model (MAE={mae_cents:.2f} cents)")

        plt.tight_layout()
        out = OUTP / f"teacher_vs_model_{clip_id}.png"
        plt.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Wrote: {out}  |  MAE={mae_cents:.2f} cents")


if __name__ == "__main__":
    main()


