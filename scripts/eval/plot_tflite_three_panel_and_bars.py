#!/usr/bin/env python3
"""
Three-panel per-clip plots (original, teacher, TFLite) and category bar chart of deviation.

- Uses the test split from metadata/splits.json
- For each test clip:
    * Loads data/processed/features/test/{clip_id}.npz
    * Reconstructs trend with specified EMA cutoff (match preprocessing)
    * teacher_corr = trend + target_shift
    * model_corr = trend + tflite_pred_shift
    * Saves 3-panel PNG: artifacts/plots/tflite_three_panel_{clip_id}.png
- Aggregates MAE(model_corr, teacher_corr) by category with groups:
    * "twinkle" if raw path or clip_id contains "twinkle"
    * "scales" if metadata category == "scales"
    * otherwise "notes"
  Saves bar chart with mean ± SEM: artifacts/plots/tflite_mae_by_category.png
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


def load_norm_and_meta():
    with open(META / "feature_norm.json", "r") as f:
        norm = json.load(f)
    fps = int(norm.get("frames_per_second", 100))
    mu = np.array(norm["feature_mean"] or [0.0] * 64, dtype=np.float32)
    sd = np.array(norm["feature_std"] or [1.0] * 64, dtype=np.float32)
    with open(META / "splits.json", "r") as f:
        splits = json.load(f)
    csv_rows = {}
    with open(META / "clips.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_rows[row["clip_id"]] = row
    return mu, sd, fps, splits, csv_rows


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
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    out_idx = 0
    for i, od in enumerate(output_details):
        n = od.get("name", "")
        if "shift" in n or "shift_cents" in n:
            out_idx = i
            break
    t = x_norm.shape[0]
    in_idx = input_details[0]["index"]
    in_info = input_details[0]
    if in_info["shape"][1] != t:
        interpreter.resize_tensor_input(in_idx, [1, t, x_norm.shape[1]], strict=False)
    interpreter.allocate_tensors()
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
    if out.ndim == 3:
        out = out[0, :, 0]
    elif out.ndim == 2:
        out = out[0, :]
    out_scale, out_zp = out_info.get("quantization", (0.0, 0))
    if out_info["dtype"] == np.int8 and out_scale not in (None, 0.0):
        out = (out.astype(np.float32) - out_zp) * out_scale
    else:
        out = out.astype(np.float32)
    return out


def infer_group(clip_id: str, row: dict) -> str:
    raw_rel = row.get("path_rel", "").lower()
    name = (raw_rel + " " + clip_id.lower())
    if "twinkle" in name:
        return "twinkle"
    if row.get("category", "") == "scales" or "scales" in name:
        return "scales"
    return "notes"


def main():
    parser = argparse.ArgumentParser(description="3-panel plots and category bar chart for TFLite vs teacher")
    parser.add_argument("--tflite", type=str,
                        default=str(ROOT / "artifacts/full/tflite/full_melody.tflite"))
    parser.add_argument("--cutoff_hz", type=float, default=2.5)
    parser.add_argument("--splits", type=str, default="test,val",
                        help="Comma-separated splits to include (e.g., 'test,val').")
    parser.add_argument("--limit", type=int, default=10_000)
    args = parser.parse_args()

    mu, sd, fps, splits, csv_rows = load_norm_and_meta()
    tflite_path = Path(args.tflite)
    split_names = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    clip_ids = []
    for s in split_names:
        clip_ids.extend(splits.get(s, []))

    maes_by_group = {"notes": [], "scales": [], "twinkle": []}
    plotted = 0
    for cid in clip_ids:
        # Find which split contains the file
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
        trend = lowpass_exponential(f0, cutoff_hz=args.cutoff_hz, fps=fps)
        teacher_corr = trend + teacher_shift
        Xn = (X - mu[None, :]) / (sd[None, :] + 1e-8)
        model_shift = run_tflite(tflite_path, Xn)
        model_corr = trend + model_shift
        t = np.arange(len(f0)) / float(fps)

        # 3-panel plot
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), sharex=True, sharey=True)
        axes[0].plot(t, f0, color="#1f77b4", linewidth=0.9)
        axes[0].set_title("original (raw f0)")
        axes[0].set_ylabel("Pitch (cents)")
        axes[1].plot(t, teacher_corr, color="#2ca02c", linewidth=1.6)
        axes[1].set_title("teacher-corrected")
        axes[2].plot(t, model_corr, color="#ff7f0e", linewidth=1.6)
        axes[2].set_title("tflite-corrected")
        for ax in axes:
            ax.set_xlabel("Time (s)")
            ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
        plt.tight_layout()
        out = PLOTS / f"tflite_three_panel_{cid}.png"
        plt.savefig(out, dpi=150)
        plt.close(fig)

        # accumulate MAE
        mae = float(np.mean(np.abs(teacher_corr - model_corr)))
        group = infer_group(cid, csv_rows.get(cid, {}))
        maes_by_group.setdefault(group, []).append(mae)
        plotted += 1
        if plotted >= args.limit:
            break

    # Bar chart by group (mean ± SEM)
    groups = ["notes", "scales", "twinkle"]
    means = [np.mean(maes_by_group[g]) if maes_by_group[g] else 0.0 for g in groups]
    sems = []
    for g in groups:
        arr = np.array(maes_by_group[g], dtype=np.float32)
        sems.append(float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0)

    plt.figure(figsize=(6.5, 4))
    xs = np.arange(len(groups))
    plt.bar(xs, means, yerr=sems, capsize=6, color=["#6baed6", "#74c476", "#fd8d3c"])
    plt.xticks(xs, groups)
    plt.ylabel("MAE vs teacher (cents)")
    for i, m in enumerate(means):
        plt.text(xs[i], m + (sems[i] if sems[i] else 0) + 0.5, f"{m:.1f}", ha="center", va="bottom", fontsize=9)
    plt.title("Deviation from teacher by category (mean ± SEM)")
    plt.tight_layout()
    out_bar = PLOTS / "tflite_mae_by_category.png"
    plt.savefig(out_bar, dpi=150)
    plt.close()
    print(f"[Summary] Saved 3-panel plots for {plotted} clips.")
    for g in groups:
        print(f"  {g}: n={len(maes_by_group[g])}, mean={np.mean(maes_by_group[g]) if maes_by_group[g] else 0.0:.2f}c")
    print(f"Wrote: {out_bar}")


if __name__ == "__main__":
    main()


