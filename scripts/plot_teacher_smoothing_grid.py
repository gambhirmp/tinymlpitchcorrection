#!/usr/bin/env python3
"""
Plot side-by-side panels showing the effect of teacher smoothing cutoffs on pitch trend (cents).

For each selected clip, produce a 1x5 grid:
 - Panel 1: raw f0 (pyin)
 - Panel 2-5: teacher-corrected trend at LP cutoffs (e.g., 5, 10, 25, 50 Hz)

Saves to artifacts/plots/teacher_smoothing_grid_{clip_id}.png
"""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/Users/mayagambhir/3600_final")
PROC = ROOT / "data/processed/features"
META = ROOT / "metadata"
PLOTP = ROOT / "artifacts/plots"
PLOTP.mkdir(parents=True, exist_ok=True)


def load_norm_and_fps():
    with open(META / "feature_norm.json", "r") as f:
        norm = json.load(f)
    fps = int(norm.get("frames_per_second", 100))
    return fps


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


def main():
    parser = argparse.ArgumentParser(description="Plot side-by-side teacher smoothing panels.")
    parser.add_argument("--contains", type=str, default="", help="Only process raw paths containing this substring.")
    parser.add_argument("--cutoffs", type=str, default="5,10,25,50",
                        help="Comma-separated low-pass cutoffs in Hz for panels 2-5.")
    parser.add_argument("--limit", type=int, default=4, help="Max number of clips to process.")
    args = parser.parse_args()

    fps = load_norm_and_fps()
    clip_map = load_clip_map()
    cutoffs = []
    for token in str(args.cutoffs).split(","):
        try:
            cutoffs.append(float(token.strip()))
        except Exception:
            pass
    # Ensure exactly 4 cutoffs for 4 panels
    if len(cutoffs) < 4:
        cutoffs = (cutoffs + [5.0, 10.0, 25.0, 50.0])[:4]
    else:
        cutoffs = cutoffs[:4]

    needle = args.contains.lower().strip()

    processed = 0
    candidates = []
    for split in ["val", "test", "train"]:
        for p in sorted((PROC / split).glob("*.npz")):
            cid = p.stem
            raw = clip_map.get(cid, "")
            if needle and needle not in raw.lower():
                continue
            candidates.append((cid, p, raw))

    for cid, npz_path, raw_path in candidates:
        if processed >= args.limit:
            break
        arr = np.load(str(npz_path))
        f0 = arr["f0_cents"].astype(np.float32)
        voiced = (arr["voiced"].astype(np.float32) > 0.5)
        t_frames = np.arange(len(f0)) / float(fps)

        fig, axes = plt.subplots(1, 5, figsize=(16, 3.5), sharex=True, sharey=True)
        # Panel 1: raw f0
        axes[0].plot(t_frames, f0, color="#1f77b4", linewidth=0.8)
        axes[0].set_title("original (raw f0)")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Pitch (cents)")

        # Panels 2-5: teacher-corrected trends
        for i, c in enumerate(cutoffs, start=1):
            trend = lowpass_exponential(f0, cutoff_hz=c, fps=fps)
            snapped = np.round(trend / 100.0) * 100.0
            teacher_shift = snapped - trend
            corrected_trend = trend + teacher_shift
            axes[i].plot(t_frames, corrected_trend, color="#ff7f0e", linewidth=1.3)
            # Light background raw for context
            axes[i].plot(t_frames, f0, color="#1f77b4", alpha=0.2, linewidth=0.6)
            axes[i].set_title(f"LP {c:g} Hz")
            axes[i].set_xlabel("Time (s)")

        # Make Y range a bit padded
        y_min = np.nanmin(f0[voiced]) if np.any(voiced) else np.nanmin(f0)
        y_max = np.nanmax(f0[voiced]) if np.any(voiced) else np.nanmax(f0)
        pad = 100.0
        for ax in axes:
            ax.set_ylim(y_min - pad, y_max + pad)
            ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

        plt.tight_layout()
        out_plot = PLOTP / f"teacher_smoothing_grid_{cid}.png"
        plt.savefig(out_plot, dpi=150)
        plt.close(fig)
        print(f"Wrote: {out_plot}")
        processed += 1


if __name__ == "__main__":
    main()


