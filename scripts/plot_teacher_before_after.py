#!/usr/bin/env python3
"""
Make 2-panel plots: original raw f0 vs teacher-corrected trend with low EMA cutoff.

Panel L: raw f0 (cents)
Panel R: teacher-corrected trend using EMA cutoff and optional hysteresis/hold-time

Saves PNGs to artifacts/plots/teacher_before_after_{clip_id}.png
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


def load_fps() -> int:
    with open(META / "feature_norm.json", "r") as f:
        norm = json.load(f)
    return int(norm.get("frames_per_second", 100))


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


def hysteretic_snap(trend: np.ndarray, deadband_cents: float, min_frames: int) -> np.ndarray:
    if len(trend) == 0:
        return trend
    snapped = np.zeros_like(trend, dtype=np.float32)
    note = np.round(trend[0] / 100.0) * 100.0
    streak = 0
    for i in range(len(trend)):
        x = float(trend[i])
        if abs(x - note) <= deadband_cents:
            streak = 0
        else:
            cand = np.round(x / 100.0) * 100.0
            if cand != note:
                streak += 1
                if streak >= max(1, int(min_frames)):
                    note = cand
                    streak = 0
        snapped[i] = note
    return snapped


def main():
    parser = argparse.ArgumentParser(description="2-panel teacher before/after plot")
    parser.add_argument("--contains", type=str, default="", help="Only process raw paths containing this substring.")
    parser.add_argument("--cutoff_hz", type=float, default=2.5, help="Low-pass cutoff for trend (Hz)")
    parser.add_argument("--hysteresis", action="store_true", help="Use hysteresis + hold-time snapping.")
    parser.add_argument("--deadband_cents", type=float, default=50.0, help="Deadband for hysteresis (cents).")
    parser.add_argument("--min_hold_frames", type=int, default=0, help="Min frames before switching note (0 => 8% of a second).")
    parser.add_argument("--limit", type=int, default=6, help="Max number of clips to process.")
    args = parser.parse_args()

    fps = load_fps()
    clip_map = load_clip_map()
    needle = args.contains.lower().strip()
    hold_frames = args.min_hold_frames if args.min_hold_frames > 0 else int(0.08 * fps)

    processed = 0
    candidates = []
    for split in ["val", "test", "train"]:
        for p in sorted((PROC / split).glob("*.npz")):
            cid = p.stem
            raw = clip_map.get(cid, "")
            if needle and needle not in raw.lower():
                continue
            candidates.append((cid, p))

    for cid, npz_path in candidates:
        if processed >= args.limit:
            break
        arr = np.load(str(npz_path))
        f0 = arr["f0_cents"].astype(np.float32)
        voiced = (arr["voiced"].astype(np.float32) > 0.5)
        t = np.arange(len(f0)) / float(fps)

        trend = lowpass_exponential(f0, cutoff_hz=args.cutoff_hz, fps=fps)
        if args.hysteresis:
            snapped = hysteretic_snap(trend, deadband_cents=args.deadband_cents, min_frames=hold_frames)
        else:
            snapped = np.round(trend / 100.0) * 100.0
        corrected_trend = trend + (snapped - trend)

        fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharex=True, sharey=True)
        axes[0].plot(t, f0, color="#1f77b4", linewidth=0.9)
        axes[0].set_title("original (raw f0)")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Pitch (cents)")

        axes[1].plot(t, corrected_trend, color="#ff7f0e", linewidth=1.6)
        axes[1].plot(t, f0, color="#1f77b4", alpha=0.25, linewidth=0.6)
        axes[1].set_title(f"teacher (LP {args.cutoff_hz:g} Hz{' + hyst' if args.hysteresis else ''})")
        axes[1].set_xlabel("Time (s)")

        # Y padding
        sel = f0[voiced] if np.any(voiced) else f0
        y_min = np.nanmin(sel)
        y_max = np.nanmax(sel)
        pad = 100.0
        for ax in axes:
            ax.set_ylim(y_min - pad, y_max + pad)
            ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

        plt.tight_layout()
        out = PLOTP / f"teacher_before_after_{cid}.png"
        plt.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Wrote: {out}")
        processed += 1


if __name__ == "__main__":
    main()


