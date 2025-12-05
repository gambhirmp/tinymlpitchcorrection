#!/usr/bin/env python3
"""
Compare original (raw f0), teacher-corrected, and TFLite model-corrected pitch trends.

- Iterates over the test split from metadata/splits.json
- Loads data/processed/features/test/{clip_id}.npz
- Recomputes trend with a specified EMA cutoff (should match preprocessing)
- Teacher-corrected = trend + target_shift (saved during preprocessing)
- TFLite model-corrected = trend + model_pred_shift
- Saves PNG per clip and prints MAE (cents) between teacher- and model-corrected trends
"""
from pathlib import Path
import argparse
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

ROOT = Path("/Users/mayagambhir/3600_final")
PROC_BASE = ROOT / "data/processed/features"
META = ROOT / "metadata"
PLOTS = ROOT / "artifacts/plots"
PLOTS.mkdir(parents=True, exist_ok=True)


def load_meta():
    with open(META / "feature_norm.json", "r") as f:
        norm = json.load(f)
    fps = int(norm.get("frames_per_second", 100))
    mu = np.array(norm["feature_mean"] or [0.0] * 64, dtype=np.float32)
    sd = np.array(norm["feature_std"] or [1.0] * 64, dtype=np.float32)

    with open(META / "splits.json", "r") as f:
        splits = json.load(f)

    clip_map = {}
    csv_path = META / "clips.csv"
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                clip_map[row["clip_id"]] = str((ROOT / row["path_rel"]).resolve())

    return mu, sd, fps, splits, clip_map


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


def run_tflite(tflite_path: Path, x_norm: np.ndarray) -> np.ndarray:
    """
    x_norm: float32 array [T, 64] normalized using feature_mean/std
    Returns float32 shift_cents [T]
    """
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Some models may have multiple outputs; pick the one containing 'shift' if present
    out_idx = 0
    for i, od in enumerate(output_details):
        name = od.get("name", "")
        if "shift" in name or "shift_cents" in name:
            out_idx = i
            break

    # Resize to [1, T, 64]
    t = x_norm.shape[0]
    in_idx = input_details[0]["index"]
    in_shape = list(input_details[0]["shape"])
    # Expect [1, any, 64] or [1, T, 64]
    if in_shape[1] != t:
        new_shape = [1, t, x_norm.shape[1]]
        interpreter.resize_tensor_input(in_idx, new_shape, strict=False)
    interpreter.allocate_tensors()

    # Prepare input tensor according to quantization
    in_info = interpreter.get_input_details()[0]
    scale, zp = in_info.get("quantization", (0.0, 0))
    if in_info["dtype"] == np.int8 and scale not in (None, 0.0):
        q = np.round(x_norm / scale + zp).astype(np.int8)
        q = np.clip(q, -128, 127)
        data = q[None, ...]
    else:
        data = x_norm[None, ...].astype(in_info["dtype"])

    interpreter.set_tensor(in_idx, data)
    interpreter.invoke()

    out_info = interpreter.get_output_details()[out_idx]
    out = interpreter.get_tensor(out_info["index"])
    # out expected shape [1, T, 1] or [1, T]
    if out.ndim == 3:
        out = out[0, :, 0]
    elif out.ndim == 2:
        out = out[0, :]
    else:
        out = out.reshape(-1)

    # Dequantize if needed
    oscale, ozp = out_info.get("quantization", (0.0, 0))
    if out_info["dtype"] == np.int8 and oscale not in (None, 0.0):
        out = (out.astype(np.float32) - ozp) * oscale
    else:
        out = out.astype(np.float32)
    return out


def main():
    parser = argparse.ArgumentParser(description="Plot original vs teacher vs TFLite model on test set")
    parser.add_argument("--tflite", type=str,
                        default=str(ROOT / "artifacts/full/tflite/full_melody.tflite"),
                        help="Path to TFLite model file")
    parser.add_argument("--cutoff_hz", type=float, default=2.5,
                        help="EMA cutoff used in preprocessing (for reconstructing trend)")
    parser.add_argument("--splits", type=str, default="test,val",
                        help="Comma-separated splits to include (e.g., 'test,val').")
    parser.add_argument("--limit", type=int, default=10_000, help="Max clips to plot (default: all)")
    args = parser.parse_args()

    mu, sd, fps, splits, _ = load_meta()
    split_names = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    clip_ids = []
    for s in split_names:
        clip_ids.extend(splits.get(s, []))

    tflite_path = Path(args.tflite)
    plotted = 0
    maes = []
    for cid in clip_ids:
        npz_path = None
        for s in split_names:
            p = PROC_BASE / s / f"{cid}.npz"
            if p.exists():
                npz_path = p
                break
        if not npz_path.exists():
            continue
        arr = np.load(str(npz_path))
        f0 = arr["f0_cents"].astype(np.float32)
        X = arr["logmel"].astype(np.float32)
        teacher_shift = arr["target_shift"].astype(np.float32)
        # Reconstruct trend consistent with preprocessing settings
        trend = lowpass_exponential(f0, cutoff_hz=args.cutoff_hz, fps=fps)
        teacher_corr = trend + teacher_shift  # equals snapped semitone trajectory with hysteresis

        # Normalize features
        Xn = (X - mu[None, :]) / (sd[None, :] + 1e-8)
        model_shift = run_tflite(tflite_path, Xn)
        model_corr = trend + model_shift

        t = np.arange(len(f0)) / float(fps)
        mae = float(np.mean(np.abs(teacher_corr - model_corr)))
        maes.append(mae)

        plt.figure(figsize=(11, 3.6))
        plt.plot(t, f0, label="original f0", color="#1f77b4", alpha=0.4, linewidth=0.8)
        plt.plot(t, teacher_corr, label="teacher-corrected", color="#2ca02c", linewidth=1.8)
        plt.plot(t, model_corr, label="tflite-corrected", color="#ff7f0e", linewidth=1.8, alpha=0.9)
        plt.xlabel("Time (s)")
        plt.ylabel("Pitch (cents)")
        plt.title(f"{cid} – MAE(model, teacher)={mae:.2f} cents")
        plt.legend(loc="best", ncol=3)
        plt.tight_layout()
        out = PLOTS / f"tflite_teacher_vs_original_{cid}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Wrote: {out}")

        plotted += 1
        if plotted >= args.limit:
            break

    if maes:
        print(f"[Summary] Plotted {plotted} clips. Mean MAE={np.mean(maes):.2f} cents; Median MAE={np.median(maes):.2f} cents.")


if __name__ == "__main__":
    main()


