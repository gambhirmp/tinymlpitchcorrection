#!/usr/bin/env python3
"""
Preprocessing pipeline:
- Scan data/raw/{notes,scales,tunes}/*/*.wav
- Resample -> loudness normalize -> trim silence
- Compute log-mel features (64 bands) at 16 kHz, win=512, hop=160
- Estimate f0 with librosa.pyin; compute voiced mask
- Build vibrato-preserving labels:
  trend = lowpass(f0_cents, ~4 Hz), snapped = snap_to_scale_or_chromatic(trend), target_shift = snapped - trend
- Save per-clip NPZ in data/processed/features/{train,val,test}/{clip_id}.npz
- Compute global normalization across all clips and write metadata/feature_norm.json
- Write metadata/clips.csv and metadata/splits.json
"""
import argparse
import json
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import soundfile as sf
import librosa

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROC_DIR = DATA_DIR / "processed" / "features"
META_DIR = ROOT / "metadata"

# params
SAMPLE_RATE = 16000  # sample rate for all audio (Hz)
WIN_S = 512          # window size 
HOP_S = 160          # hop length 
FPS = int(SAMPLE_RATE / HOP_S)  # frames per second for features
N_MELS = 64          # number of mel filterbank channels
FMIN = 50.0          # minimum frequency (Hz) for mel spectrograms
FMAX = 2000.0        # maximum frequency (Hz) for mel spectrograms


def ensure_dirs() -> None:
    for d in [
        RAW_DIR / "notes",
        RAW_DIR / "scales",
        RAW_DIR / "tunes",
        INTERIM_DIR / "resampled",
        INTERIM_DIR / "trimmed",
        PROC_DIR / "train",
        PROC_DIR / "val",
        PROC_DIR / "test",
        META_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)

# normalizes the audio to -18 dB 
def db_normalize(audio: np.ndarray, target_db: float = -18.0, eps: float = 1e-9) -> np.ndarray:
    rms = np.sqrt(np.mean(audio**2) + eps)
    current_db = 20.0 * np.log10(rms + eps)
    gain_db = target_db - current_db
    gain = 10 ** (gain_db / 20.0)
    return np.clip(audio * gain, -1.0, 1.0)

# removes silence from the audio
def trim_silence(audio: np.ndarray, top_db: float = 40.0) -> np.ndarray:
    yt, _ = librosa.effects.trim(audio, top_db=top_db)
    return yt

# computes the log-mel spectrogram
def compute_logmel(y: np.ndarray, sr: int) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=WIN_S,
        hop_length=HOP_S,
        win_length=WIN_S,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
        center=True,
        window="hann",
    )
    logmel = librosa.power_to_db(S, ref=1.0, top_db=80.0)
    return logmel.T.astype(np.float32)  # [T, 64]

# estimates the fundamental frequency (f0) using the pyin algorithm
def estimate_f0(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        frame_length=WIN_S,
        hop_length=HOP_S,
        center=True,
    )
    # Interpolate small gaps
    f0 = f0.astype(np.float32)
    mask = np.isnan(f0)
    if np.any(~mask):
        idx = np.arange(len(f0))
        f0[mask] = np.interp(idx[mask], idx[~mask], f0[~mask])
    else:
        f0[:] = 0.0
    voiced = (voiced_prob > 0.5).astype(np.float32)
    return f0, voiced

# converts the frequency in Hz to cents
def hz_to_cents(f0_hz: np.ndarray, ref_hz: float = 440.0, eps: float = 1e-6) -> np.ndarray:
    return 1200.0 * np.log2((f0_hz + eps) / ref_hz)

# applies a low-pass filter to the signal
def lowpass_exponential(x: np.ndarray, cutoff_hz: float, fps: int) -> np.ndarray:
    # First-order low-pass filter with cutoff in Hz (on frame-rate domain)
    # alpha derived from RC filter: alpha = dt / (RC + dt), with RC = 1/(2*pi*fc)
    dt = 1.0 / fps
    rc = 1.0 / (2.0 * np.pi * max(cutoff_hz, 1e-3))
    alpha = dt / (rc + dt)
    y = np.zeros_like(x, dtype=np.float32)
    prev = x[0]
    for i in range(len(x)):
        prev = prev + alpha * (x[i] - prev)
        y[i] = prev
    return y

# rounds the frequency in cents to the nearest 100 cents
def snap_to_chromatic(cents: np.ndarray) -> np.ndarray:
    # Round to nearest 100 cents
    return np.round(cents / 100.0) * 100.0

# snaps the trend to the nearest 100-cent step with hysteresis and minimum hold-time
def hysteretic_snap(trend: np.ndarray, deadband_cents: float, min_frames: int) -> np.ndarray:
    """
    Snap trend (cents) to 100-cent steps with hysteresis and minimum hold-time.
    - Only switch notes when the candidate semitone is different and the trend
      stays outside the ±deadband around the current note for >= min_frames.
    """
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

# builds the targets for the pitch correction model
def build_targets(
    f0_cents: np.ndarray,
    voiced: np.ndarray,
    cutoff_hz: float = 4.0,
    deadband_cents: float = 50.0,
    min_hold_frames: int = 8,
) -> np.ndarray:
    trend = lowpass_exponential(f0_cents, cutoff_hz=cutoff_hz, fps=FPS)
    snapped = hysteretic_snap(trend, deadband_cents=deadband_cents, min_frames=min_hold_frames)
    target_shift = snapped - trend
    target_shift = target_shift * (voiced > 0.5)  # zero on unvoiced
    return target_shift.astype(np.float32)


@dataclass
class ClipMeta:
    clip_id: str
    category: str
    singer_id: str
    take_id: str
    path_rel: str
    sample_rate: int
    duration_sec: float
    scale_id: Optional[str] = None

# iterates over the raw wav files
def iter_raw_wavs() -> List[Tuple[str, Path]]:
    """
    Return list of (category, wav_path) scanning raw data recursively to include nested folders
    such as notes/<note>/Maya/*.wav.
    """
    pairs: List[Tuple[str, Path]] = []
    for category in ["notes", "scales", "tunes"]:
        d = RAW_DIR / category
        if not d.exists():
            continue
        for wav_path in sorted(d.rglob("*.wav")):
            pairs.append((category, wav_path))
    return pairs

# processes a single clip
def process_clip(
    category: str,
    wav_path: Path,
    cutoff_hz: float,
    deadband_cents: float,
    min_hold_frames: int,
) -> Tuple[ClipMeta, Dict[str, np.ndarray]]:
    singer_id = wav_path.parent.name
    take_id = wav_path.stem
    y, sr = sf.read(wav_path)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE, res_type="soxr_hq")
        sr = SAMPLE_RATE
    y = db_normalize(y)
    y = trim_silence(y)
    duration_sec = len(y) / sr
    # saves the interim 
    resampled_path = INTERIM_DIR / "resampled" / f"{category}_{singer_id}_{take_id}.wav"
    resampled_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(resampled_path), y, sr)

    # Features
    logmel = compute_logmel(y, sr)  # [T, 64]
    f0_hz, voiced = estimate_f0(y, sr)
    f0_cents = hz_to_cents(f0_hz)
    # builds the targets for the pitch correction model
    target_shift = build_targets(
        f0_cents,
        voiced,
        cutoff_hz=cutoff_hz,
        deadband_cents=deadband_cents,
        min_hold_frames=min_hold_frames,
    )
    # truncates the features to the shortest length
    T = min(len(logmel), len(f0_cents), len(target_shift), len(voiced))
    logmel = logmel[:T]
    f0_cents = f0_cents[:T]
    target_shift = target_shift[:T]
    voiced = voiced[:T]

    # generates a unique clip id
    clip_id = f"{category}_{singer_id}_{take_id}_{uuid.uuid4().hex[:8]}"
    # creates a metadata object
    meta = ClipMeta(
        clip_id=clip_id,
        category=category,
        singer_id=singer_id,
        take_id=take_id,
        path_rel=str(wav_path.relative_to(ROOT)),
        sample_rate=sr,
        duration_sec=float(duration_sec),
        scale_id=None,
    )
    # creates a dictionary of the features: logmel, f0_cents, voiced, target_shift
    arrays = {
        "logmel": logmel.astype(np.float32),
        "f0_cents": f0_cents.astype(np.float32),
        "voiced": voiced.astype(np.float32),
        "target_shift": target_shift.astype(np.float32),
    }
    return meta, arrays

# writes the features to a npz file
def write_npz(split: str, clip_id: str, arrays: Dict[str, np.ndarray]) -> Path:
    out_path = PROC_DIR / split / f"{clip_id}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    return out_path

# updates the clips csv file
def update_clips_csv(rows: List[ClipMeta]) -> None:
    csv_path = META_DIR / "clips.csv"
    have_header = csv_path.exists() and csv_path.stat().st_size > 0
    with open(csv_path, "a", encoding="utf-8") as f:
        if not have_header:
            f.write("clip_id,category,singer_id,take_id,path_rel,sample_rate,duration_sec,scale_id\n")
        for r in rows:
            f.write(
                f"{r.clip_id},{r.category},{r.singer_id},{r.take_id},{r.path_rel},"
                f"{r.sample_rate},{r.duration_sec},{'' if r.scale_id is None else r.scale_id}\n"
            )

# performs a stratified split of the clips
def stratified_split(metas: List[ClipMeta], seed: int = 13) -> Dict[str, List[str]]:
    rng = np.random.RandomState(seed)
    by_singer: Dict[str, List[ClipMeta]] = {}
    for m in metas:
        by_singer.setdefault(m.singer_id, []).append(m)
    train_ids: List[str] = []
    val_ids: List[str] = []
    test_ids: List[str] = []
    for singer, items in by_singer.items():
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, int(0.1 * n))
        n_val = max(1, int(0.1 * n))
        test_ids.extend([m.clip_id for m in items[:n_test]])
        val_ids.extend([m.clip_id for m in items[n_test:n_test + n_val]])
        train_ids.extend([m.clip_id for m in items[n_test + n_val:]])
    return {"train": train_ids, "val": val_ids, "test": test_ids}

# computes the feature normalization
def compute_feature_norm(all_feature_paths: List[Path]) -> Tuple[List[float], List[float]]:
    # Stream over files to compute mean/std per mel bin
    sum_vec = np.zeros((N_MELS,), dtype=np.float64)
    sumsq_vec = np.zeros((N_MELS,), dtype=np.float64)
    count = 0
    for p in all_feature_paths:
        data = np.load(p)
        logmel = data["logmel"]  # [T, 64]
        sum_vec += logmel.sum(axis=0)
        sumsq_vec += np.square(logmel).sum(axis=0)
        count += logmel.shape[0]
    mean = (sum_vec / max(count, 1)).astype(float).tolist()
    var = (sumsq_vec / max(count, 1) - np.square(sum_vec / max(count, 1))).astype(float)
    std = np.sqrt(np.maximum(var, 1e-8)).tolist()
    return mean, std


def write_json(path: Path, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def main(args: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--cutoff_hz", type=float, default=4.0, help="Low-pass cutoff for trend (Hz)")
    parser.add_argument("--limit_per_category", type=int, default=0, help="Limit number of files per category (0=all)")
    parser.add_argument("--deadband_cents", type=float, default=50.0, help="Hysteresis deadband around current note (cents)")
    parser.add_argument("--min_hold_frames", type=int, default=int(0.08 * FPS), help="Minimum frames to hold before switching note")
    parsed = parser.parse_args(args)

    ensure_dirs()
    pairs = iter_raw_wavs()
    # Optional limit to speed up first run
    if parsed.limit_per_category > 0:
        limited = []
        seen_count: Dict[str, int] = {}
        for cat, p in pairs:
            seen_count[cat] = seen_count.get(cat, 0) + 1
            if seen_count[cat] <= parsed.limit_per_category:
                limited.append((cat, p))
        pairs = limited

    metas: List[ClipMeta] = []
    arrays_by_id: Dict[str, Dict[str, np.ndarray]] = {}

    # processes the clips
    for category, wav_path in pairs:
        try:
            meta, arrays = process_clip(
                category,
                wav_path,
                cutoff_hz=parsed.cutoff_hz,
                deadband_cents=parsed.deadband_cents,
                min_hold_frames=parsed.min_hold_frames,
            )
            metas.append(meta)
            arrays_by_id[meta.clip_id] = arrays
        except Exception as e:
            print(f"[WARN] Failed to process {wav_path}: {e}", file=sys.stderr)

    if not metas:
        print("[INFO] No clips found. Place WAV files under data/raw and rerun.")
        return

    # splits the clips
    splits = stratified_split(metas, seed=parsed.seed)
    write_json(META_DIR / "splits.json", splits)

    # saves the npzs ( augmented clips train-only)
    for m in metas:
        split = "train"
        if m.clip_id in splits["val"]:
            split = "val"
        elif m.clip_id in splits["test"]:
            split = "test"
        if "_ps" in m.take_id:
            split = "train"
        write_npz(split, m.clip_id, arrays_by_id[m.clip_id])

    update_clips_csv(metas)

    # normalization 
    train_paths = [PROC_DIR / "train" / f"{cid}.npz" for cid in splits["train"]]
    mean, std = compute_feature_norm(train_paths)
    write_json(
        META_DIR / "feature_norm.json",
        {
            "feature_mean": mean,
            "feature_std": std,
            "frames_per_second": FPS,
            "mel_bands": N_MELS,
            "sample_rate_hz": SAMPLE_RATE,
            "stft_window_samples": WIN_S,
            "stft_hop_samples": HOP_S,
        },
    )
    print(f"[OK] Processed {len(metas)} clips. Train/val/test: "
          f"{len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")


if __name__ == "__main__":
    main()


