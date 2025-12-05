#!/usr/bin/env python3
"""
Render and plot teacher-corrected audio with different smoothing levels (low-pass cutoffs).

For each selected clip:
 - Load f0_cents and voiced mask from data/processed/features/*.npz
 - For each cutoff (Hz), compute trend = lowpass(f0_cents, cutoff)
 - Compute teacher shift = round(trend/100)*100 - trend (cents)
 - Plot raw f0, trend, and corrected trends for each cutoff
 - Synthesize "teacher-corrected" audio by piecewise-constant pitch shifting per note segment
   (segments defined by changes in snapped_trend). Saves WAV per cutoff.

Notes:
 - This is an offline renderer for demonstration; piecewise-constant pitch shifting uses
   librosa.effects.pitch_shift per segment with simple crossfades. It will introduce seam
   artifacts at boundaries but suffices to audition how different smoothing cutoffs sound.
"""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt

ROOT = Path("/Users/mayagambhir/3600_final")
PROC = ROOT / "data/processed/features"
META = ROOT / "metadata"
OUTP = ROOT / "artifacts/listen"
PLOTP = ROOT / "artifacts/plots"
OUTP.mkdir(parents=True, exist_ok=True)
PLOTP.mkdir(parents=True, exist_ok=True)


def load_feature_norm():
    """(Unused placeholder retained for compatibility)."""
    with open(META / "feature_norm.json", "r") as f:
        return json.load(f)


def load_norm_and_fps():
    with open(META / "feature_norm.json", "r") as f:
        norm = f.read()
    meta = json.loads(norm)
    mu = np.array(meta["feature_mean"] or [0.0] * 64, dtype=np.float32)
    sd = np.array(meta["feature_std"] or [1.0] * 64, dtype=np.float32)
    fps = int(meta.get("frames_per_second", 100))
    sr = int(meta.get("sample_rate_hz", 16000))
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


def piecewise_pitch_shift(y, sr, shift_cents, voiced, fps, fade_ms=20):
    """
    Apply piecewise-constant pitch shifting based on per-frame shift_cents (cents).
    Segments are contiguous runs where the rounded target semitone stays constant
    (or where voiced==0). For unvoiced frames, copy original audio segment.
    """
    hop = int(round(sr / fps))
    n_frames = len(shift_cents)
    # Compute snapped semitone centers to define segments
    snapped = np.round(shift_cents / 100.0)  # in semitone units
    # Identify boundaries where snapped or voiced changes
    boundaries = [0]
    for i in range(1, n_frames):
        if (snapped[i] != snapped[i - 1]) or (voiced[i] != voiced[i - 1]):
            boundaries.append(i)
    boundaries.append(n_frames)

    out = np.zeros_like(y, dtype=np.float32)
    fade = int(sr * (fade_ms / 1000.0))
    if fade < 1:
        fade = 1

    def segment_samples(fi0, fi1):
        # Map frame indices to sample indices using hop; clamp to signal length
        start = int(fi0 * hop)
        end = int(min(len(y), fi1 * hop))
        return start, end

    for s_idx in range(len(boundaries) - 1):
        f0 = boundaries[s_idx]
        f1 = boundaries[s_idx + 1]
        s, e = segment_samples(f0, f1)
        if s >= e:
            continue
        seg = y[s:e]
        if seg.size == 0:
            continue
        if voiced[f0] <= 0.5:
            # Unvoiced: copy as-is
            seg_out = seg.astype(np.float32, copy=False)
        else:
            step_semi = float(np.median(shift_cents[f0:f1]) / 100.0)
            try:
                seg_out = librosa.effects.pitch_shift(seg.astype(np.float32), sr=sr, n_steps=step_semi)
            except Exception:
                # Fallback: if pitch shift fails, copy original
                seg_out = seg.astype(np.float32, copy=False)
        # Ensure same length (librosa keeps length but guard anyway)
        if len(seg_out) != (e - s):
            # Pad or trim
            if len(seg_out) > (e - s):
                seg_out = seg_out[: (e - s)]
            else:
                pad = (e - s) - len(seg_out)
                seg_out = np.pad(seg_out, (0, pad))
        # Overlap-add with simple crossfade
        if s > 0 and fade > 0:
            left = max(0, s - fade)
            n = min(s - left, len(seg_out))
            if n > 0:
                w = np.linspace(0.0, 1.0, n, dtype=np.float32)
                out[s - n:s] = out[s - n:s] * (1 - w) + seg_out[:n] * w
            # Write the remainder of this segment after the crossfade
            rem = seg_out[n:]
            end_pos = min(len(out), s + len(rem))
            out[s:end_pos] = rem[: end_pos - s]
        else:
            end_pos = min(e, s + len(seg_out))
            out[s:end_pos] = seg_out[: end_pos - s]
    return out


def infer_group(raw_path: str) -> str:
    lower = raw_path.lower()
    if "twinkle" in lower:
        return "twinkle"
    if "scale" in lower or "scales" in lower:
        return "scales"
    if "tune" in lower or "tunes" in lower:
        return "tunes"
    return "notes"


def main():
    parser = argparse.ArgumentParser(description="Render teacher-corrected audio with different smoothing levels.")
    parser.add_argument("--contains", type=str, default="", help="Only process raw paths containing this substring.")
    parser.add_argument("--cutoffs", type=str, default="2,3,4,6",
                        help="Comma-separated low-pass cutoffs in Hz, e.g., '2,3,4,6'.")
    parser.add_argument("--limit", type=int, default=4, help="Max number of clips to process.")
    parser.add_argument("--hysteresis", action="store_true", help="Use hysteresis + hold-time snapping.")
    parser.add_argument("--deadband_cents", type=float, default=50.0, help="Deadband for hysteresis (cents).")
    parser.add_argument("--min_hold_frames", type=int, default=0, help="Min frames to hold before switching note (0 => 8% of a second).")
    args = parser.parse_args()

    mu, sd, fps, sr = load_norm_and_fps()
    clip_map = load_clip_map()
    cutoffs = []
    for token in str(args.cutoffs).split(","):
        try:
            cutoffs.append(float(token.strip()))
        except Exception:
            pass
    if not cutoffs:
        cutoffs = [4.0]
    needle = args.contains.lower().strip()

    processed = 0
    # Gather candidate npz files
    candidates = []
    for split in ["val", "test", "train"]:
        for p in sorted((PROC / split).glob("*.npz")):
            cid = p.stem
            raw = clip_map.get(cid, "")
            if not raw:
                continue
            if needle and needle not in raw.lower():
                continue
            candidates.append((cid, p, raw))

    for cid, npz_path, raw_path in candidates:
        if processed >= args.limit:
            break
        if not Path(raw_path).exists():
            continue

        arr = np.load(str(npz_path))
        f0 = arr["f0_cents"].astype(np.float32)
        voiced = (arr["voiced"].astype(np.float32) > 0.5)

        # Load audio
        y, sr0 = sf.read(raw_path)
        if isinstance(y, np.ndarray) and y.ndim > 1:
            y = y.mean(axis=1)
        if sr0 != sr:
            y = librosa.resample(y, orig_sr=sr0, target_sr=sr)
        t_frames = np.arange(len(f0)) / float(fps)
        group = infer_group(raw_path)
        clip_dir = OUTP / group / cid
        clip_dir.mkdir(parents=True, exist_ok=True)

        # Save original (resampled) for A/B listening
        out_wav_orig = clip_dir / "original.wav"
        try:
            sf.write(str(out_wav_orig), y.astype(np.float32), sr)
        except Exception:
            pass

        # Plot setup
        plt.figure(figsize=(12, 5))
        # Plot raw f0
        trend_default = lowpass_exponential(f0, cutoff_hz=4.0, fps=fps)
        plt.plot(t_frames, f0, color="#1f77b4", alpha=0.35, linewidth=0.8, label="raw f0 (pyin)")
        plt.plot(t_frames, trend_default, color="#888", linestyle="--", linewidth=1.0, label="trend 4 Hz")

        # For each cutoff, compute corrected trend and render audio
        for c in cutoffs:
            trend = lowpass_exponential(f0, cutoff_hz=c, fps=fps)
            if args.hysteresis:
                hold_frames = args.min_hold_frames if args.min_hold_frames > 0 else int(0.08 * fps)
                # Local hysteretic snap (mirror of preprocessing behavior)
                note = np.round(trend[0] / 100.0) * 100.0
                snapped = np.zeros_like(trend, dtype=np.float32)
                streak = 0
                for i in range(len(trend)):
                    x = float(trend[i])
                    if abs(x - note) <= args.deadband_cents:
                        streak = 0
                    else:
                        cand = np.round(x / 100.0) * 100.0
                        if cand != note:
                            streak += 1
                            if streak >= max(1, int(hold_frames)):
                                note = cand
                                streak = 0
                    snapped[i] = note
            else:
                snapped = np.round(trend / 100.0) * 100.0
            teacher_shift = snapped - trend  # cents
            corrected_trend = trend + teacher_shift
            plt.plot(t_frames, corrected_trend, linewidth=1.5, label=f"teacher-corr (LP {c:g} Hz)")

            # Synthesize audio with piecewise-constant pitch shift
            y_corr = piecewise_pitch_shift(y, sr, teacher_shift, voiced.astype(np.float32), fps=fps, fade_ms=20)
            freq_tag = f"{c:.2f}".rstrip("0").rstrip(".")
            out_wav = clip_dir / f"teacher_lp{freq_tag}Hz.wav"
            sf.write(str(out_wav), y_corr, sr)

        plt.xlabel("Time (s)")
        plt.ylabel("Pitch (cents vs A4=440)")
        plt.title(f"{cid} – teacher smoothing variants")
        plt.legend(loc="best", ncol=2)
        out_plot = PLOTP / f"teacher_smoothing_{cid}.png"
        plt.tight_layout()
        plt.savefig(out_plot, dpi=150)
        plt.close()
        print(f"Wrote: {out_plot}")

        processed += 1


if __name__ == "__main__":
    main()


