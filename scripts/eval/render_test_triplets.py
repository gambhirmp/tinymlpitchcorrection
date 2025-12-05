#!/usr/bin/env python3
"""
Render per-test clip triplets: original.wav, teacher.wav, tflite.wav
organized under artifacts/listen/{group}/{clip_id}/

Definitions:
- teacher: uses preprocessing labels (target_shift) with low-EMA trend to produce
          a snapped semitone trajectory; renders piecewise-constant pitch shift.
- tflite: runs the full-melody TFLite model, builds corrected trend as trend + pred,
          segments by rounded semitone, and renders piecewise-constant pitch shifts.
"""
from pathlib import Path
import argparse
import json
import csv
import numpy as np
import soundfile as sf
import librosa
import tensorflow as tf

ROOT = Path("/Users/mayagambhir/3600_final")
PROC = ROOT / "data/processed/features" / "test"
META = ROOT / "metadata"
OUT_ROOT = ROOT / "artifacts/listen"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def load_meta():
    with open(META / "feature_norm.json", "r") as f:
        norm = json.load(f)
    fps = int(norm.get("frames_per_second", 100))
    sr = int(norm.get("sample_rate_hz", 16000))
    mu = np.array(norm["feature_mean"] or [0.0] * 64, dtype=np.float32)
    sd = np.array(norm["feature_std"] or [1.0] * 64, dtype=np.float32)
    with open(META / "splits.json", "r") as f:
        splits = json.load(f)
    clips = {}
    with open(META / "clips.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clips[row["clip_id"]] = row
    return mu, sd, fps, sr, splits, clips


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


def run_tflite(model_path: Path, X_norm: np.ndarray) -> np.ndarray:
    """Return predicted shift_cents [T]."""
    interp = tf.lite.Interpreter(model_path=str(model_path))
    inp = interp.get_input_details()[0]
    out_all = interp.get_output_details()
    out_idx = 0
    for i, od in enumerate(out_all):
        name = od.get("name", "")
        if "shift" in name or "shift_cents" in name:
            out_idx = i
            break
    t = X_norm.shape[0]
    if inp["shape"][1] != t:
        interp.resize_tensor_input(inp["index"], [1, t, X_norm.shape[1]], strict=False)
    interp.allocate_tensors()
    scale, zp = inp.get("quantization", (0.0, 0))
    if inp["dtype"] == np.int8 and scale not in (None, 0.0):
        q = np.round(X_norm / scale + zp).astype(np.int8)
        q = np.clip(q, -128, 127)
        data = q[None, ...]
    else:
        data = X_norm[None, ...].astype(inp["dtype"])
    interp.set_tensor(inp["index"], data)
    interp.invoke()
    out_info = interp.get_output_details()[out_idx]
    out = interp.get_tensor(out_info["index"])
    if out.ndim == 3:
        out = out[0, :, 0]
    elif out.ndim == 2:
        out = out[0, :]
    o_scale, o_zp = out_info.get("quantization", (0.0, 0))
    if out_info["dtype"] == np.int8 and o_scale not in (None, 0.0):
        out = (out.astype(np.float32) - o_zp) * o_scale
    else:
        out = out.astype(np.float32)
    return out


def piecewise_pitch_shift(y: np.ndarray, sr: int, fps: int,
                          shift_cents: np.ndarray,
                          segment_ids: np.ndarray,
                          voiced_mask: np.ndarray,
                          fade_ms: float = 30.0) -> np.ndarray:
    """Render with constant shift per segment_id; copy unvoiced."""
    hop = int(round(sr / fps))
    n_frames = len(shift_cents)
    # Build boundaries where segment id or voicing changes
    boundaries = [0]
    for i in range(1, n_frames):
        if (segment_ids[i] != segment_ids[i - 1]) or (voiced_mask[i] != voiced_mask[i - 1]):
            boundaries.append(i)
    boundaries.append(n_frames)
    out = np.zeros_like(y, dtype=np.float32)
    fade = int(sr * (fade_ms / 1000.0))
    if fade < 1:
        fade = 1
    for s_idx in range(len(boundaries) - 1):
        f0 = boundaries[s_idx]
        f1 = boundaries[s_idx + 1]
        start = int(f0 * hop)
        end = int(min(len(y), f1 * hop))
        if start >= end:
            continue
        seg = y[start:end]
        if seg.size == 0:
            continue
        if voiced_mask[f0] <= 0.5:
            seg_out = seg.astype(np.float32, copy=False)
        else:
            step_semi = float(np.median(shift_cents[f0:f1]) / 100.0)
            try:
                seg_out = librosa.effects.pitch_shift(seg.astype(np.float32), sr=sr, n_steps=step_semi)
            except Exception:
                seg_out = seg.astype(np.float32, copy=False)
        # match length
        if len(seg_out) != (end - start):
            if len(seg_out) > (end - start):
                seg_out = seg_out[: end - start]
            else:
                seg_out = np.pad(seg_out, (0, (end - start) - len(seg_out)))
        # crossfade
        if start > 0 and fade > 0:
            left = max(0, start - fade)
            n = min(start - left, len(seg_out))
            if n > 0:
                w = np.linspace(0.0, 1.0, n, dtype=np.float32)
                out[start - n:start] = out[start - n:start] * (1 - w) + seg_out[:n] * w
            rem = seg_out[n:]
            end_pos = min(len(out), start + len(rem))
            out[start:end_pos] = rem[: end_pos - start]
        else:
            out[start:end] = seg_out
    return out


def infer_group(row: dict, clip_id: str) -> str:
    raw_rel = (row.get("path_rel", "") or "").lower()
    name = (raw_rel + " " + clip_id.lower())
    if "twinkle" in name:
        return "twinkle"
    if row.get("category", "") == "scales" or "scales" in name:
        return "scales"
    return "notes"


def main():
    parser = argparse.ArgumentParser(description="Render original/teacher/tflite triplets for test set.")
    parser.add_argument("--tflite", type=str,
                        default=str(ROOT / "artifacts/full/tflite/full_melody.tflite"))
    parser.add_argument("--cutoff_hz", type=float, default=2.5,
                        help="EMA cutoff to reconstruct trend (match preprocessing).")
    parser.add_argument("--limit", type=int, default=100000, help="Max clips to render (default: all).")
    args = parser.parse_args()

    mu, sd, fps, sr, splits, clips = load_meta()
    tflite_path = Path(args.tflite)
    test_ids = splits.get("test", [])

    rendered = 0
    for cid in test_ids:
        npz = PROC / f"{cid}.npz"
        if not npz.exists():
            continue
        row = clips.get(cid, {})
        raw_path = ROOT / row.get("path_rel", "")
        if not raw_path.exists():
            continue
        # Load audio mono @ sr
        y, sr0 = sf.read(str(raw_path))
        if isinstance(y, np.ndarray) and y.ndim > 1:
            y = y.mean(axis=1)
        if sr0 != sr:
            y = librosa.resample(y, orig_sr=sr0, target_sr=sr)
        # Load features/labels
        arr = np.load(str(npz))
        f0 = arr["f0_cents"].astype(np.float32)
        voiced = (arr["voiced"].astype(np.float32) > 0.5)
        X = arr["logmel"].astype(np.float32)
        target_shift = arr["target_shift"].astype(np.float32)
        trend = lowpass_exponential(f0, cutoff_hz=args.cutoff_hz, fps=fps)
        teacher_corr = trend + target_shift
        teacher_seg = np.round(teacher_corr / 100.0)  # segment ids by snapped semitone

        # Model
        Xn = (X - mu[None, :]) / (sd[None, :] + 1e-8)
        model_shift = run_tflite(tflite_path, Xn)
        model_corr = trend + model_shift
        model_seg = np.round(model_corr / 100.0)

        # Render
        group = infer_group(row, cid)
        out_dir = OUT_ROOT / group / cid
        out_dir.mkdir(parents=True, exist_ok=True)
        # Original
        sf.write(str(out_dir / "original.wav"), y.astype(np.float32), sr)
        # Teacher
        y_teacher = piecewise_pitch_shift(y, sr, fps, target_shift, teacher_seg, voiced.astype(np.float32), fade_ms=30.0)
        sf.write(str(out_dir / "teacher.wav"), y_teacher, sr)
        # TFLite
        y_model = piecewise_pitch_shift(y, sr, fps, model_shift, model_seg, voiced.astype(np.float32), fade_ms=30.0)
        sf.write(str(out_dir / "tflite.wav"), y_model, sr)

        rendered += 1
        if rendered >= args.limit:
            break
    print(f"[OK] Rendered triplets for {rendered} test clips into {OUT_ROOT}/<group>/<clip_id>/")


if __name__ == "__main__":
    main()


