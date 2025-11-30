#!/usr/bin/env python3
"""
Plot original waveform and pitch tracks (before/after) for a subset of clips.
Figure layout:
  - Top: time-domain waveform (normalized), seconds on x-axis
  - Bottom: pitch (cents vs A4=440):
      * raw f0 (pyin)          : blue
      * trend (low-pass ~4 Hz) : gray dashed
      * corrected trend        : orange
      * (optional) corrected f0: thin orange, raw f0 + predicted shift
Outputs PNGs to artifacts/plots/wave_pitch_{clip_id}.png
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


def list_npz_subset(limit=6):
    paths = []
    for split in ["val", "test", "train"]:
        paths.extend(glob.glob(str(PROC / split / "*.npz")))
    return sorted(paths)[:limit]


def main():
    mu, sd, fps, sr = load_feature_norm()
    clip_map = load_clip_map()
    serve = tf.saved_model.load(str(ROOT / "artifacts/baseline/saved_model"))

    for npz_path in list_npz_subset(limit=8):
        npz_path = Path(npz_path)
        clip_id = npz_path.stem
        raw_path = clip_map.get(clip_id)
        if not raw_path or not Path(raw_path).exists():
            print(f"[WARN] missing raw audio for {clip_id}, skipping.")
            continue

        arr = np.load(npz_path)
        f0 = arr["f0_cents"].astype(np.float32)
        X = (arr["logmel"].astype(np.float32) - mu[None, :]) / (sd[None, :] + 1e-8)
        pred = serve.serve(tf.convert_to_tensor(X[None, ...], tf.float32))["shift_cents"].numpy()[0, :, 0]
        trend = lowpass_exponential(f0, cutoff_hz=4.0, fps=fps)
        corrected_trend = trend + pred
        corrected_f0 = f0 + pred

        y, _ = sf.read(raw_path)
        if y.ndim > 1:
            y = y.mean(axis=1)
        t_audio = np.arange(len(y)) / sr
        t_frames = np.arange(len(f0)) / float(fps)
        y_norm = y / (np.max(np.abs(y)) + 1e-9)

        fig = plt.figure(figsize=(12, 4.5))
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(t_audio, y_norm, color="#444", linewidth=0.8)
        ax1.set_title(f"{clip_id} – waveform (normalized)")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")

        ax2 = plt.subplot(2, 1, 2)
        ax2.plot(t_frames, f0, label="raw f0 (cents)", color="#1f77b4", alpha=0.8)
        ax2.plot(t_frames, trend, label="trend (low-pass)", color="#888888", linestyle="--", linewidth=1.2)
        ax2.plot(t_frames, corrected_trend, label="corrected trend", color="#ff7f0e", linewidth=2.0)
        ax2.plot(t_frames, corrected_f0, label="corrected f0 (raw+shift)", color="#ff7f0e", alpha=0.4, linewidth=0.8)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Pitch (cents vs A4=440)")
        ax2.legend(loc="best", ncol=2)
        plt.tight_layout()
        out = OUTP / f"wave_pitch_{clip_id}.png"
        plt.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Wrote: {out}")


if __name__ == "__main__":
    main()


