#!/usr/bin/env python3
"""
Render before/after WAVs for a few clips using the SavedModel predictions.
For a quick listen, applies a constant median shift (in semitones) per clip.
Outputs:
  artifacts/listen/{clip_id}_before.wav
  artifacts/listen/{clip_id}_after.wav
"""
from pathlib import Path
import csv
import json
import glob
import numpy as np
import soundfile as sf
import librosa
import tensorflow as tf

ROOT = Path("/Users/mayagambhir/3600_final")
PROC = ROOT / "data/processed/features"
META = ROOT / "metadata"
ART = ROOT / "artifacts/listen"
ART.mkdir(parents=True, exist_ok=True)


def load_feature_norm():
    with open(META / "feature_norm.json", "r") as f:
        norm = json.load(f)
    mu = np.array(norm["feature_mean"] or [0.0] * 64, np.float32)
    sd = np.array(norm["feature_std"] or [1.0] * 64, np.float32)
    return mu, sd


def load_clip_map():
    """Return dict clip_id -> raw path (absolute)."""
    m = {}
    csv_path = META / "clips.csv"
    if not csv_path.exists():
        return m
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip_id = row["clip_id"]
            path_rel = row["path_rel"]
            m[clip_id] = str((ROOT / path_rel).resolve())
    return m


def pick_npz(limit=3):
    files = sorted(glob.glob(str(PROC / "val" / "*.npz")))
    if len(files) < limit:
        files += sorted(glob.glob(str(PROC / "test" / "*.npz")))
    return files[:limit]


def main():
    mu, sd = load_feature_norm()
    clip_map = load_clip_map()
    serve = tf.saved_model.load(str(ROOT / "artifacts/baseline/saved_model"))
    sr = 16000
    picked = pick_npz(limit=5)
    if not picked:
        print("No feature files found.")
        return
    for npz_path in picked:
        npz_path = Path(npz_path)
        clip_id = npz_path.stem
        arr = np.load(npz_path)
        X = (arr["logmel"].astype(np.float32) - mu[None, :]) / (sd[None, :] + 1e-8)
        pred = serve.serve(tf.convert_to_tensor(X[None, ...], tf.float32))["shift_cents"].numpy()[0, :, 0]
        median_cents = float(np.median(pred))
        n_steps = median_cents / 100.0
        raw_path = clip_map.get(clip_id)
        if not raw_path or not Path(raw_path).exists():
            print(f"[WARN] Missing raw path for {clip_id}, skipping.")
            continue
        y, _ = librosa.load(raw_path, sr=sr, mono=True)
        # Write before
        out_before = ART / f"{clip_id}_before.wav"
        sf.write(str(out_before), y.astype(np.float32), sr)
        # Apply constant shift for quick listen
        y_shift = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
        out_after = ART / f"{clip_id}_after.wav"
        sf.write(str(out_after), y_shift.astype(np.float32), sr)
        print(f"Wrote: {out_before.name} (orig), {out_after.name} (median {median_cents:.1f} cents)")


if __name__ == "__main__":
    main()


