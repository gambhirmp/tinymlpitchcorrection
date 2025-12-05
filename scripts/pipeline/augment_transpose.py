#!/usr/bin/env python3
"""
Augment raw note WAVs by pitch-shifting to new notes and octaves.
- Input root: data/raw/notes/**/*.wav
- Output: same folders with filename suffix _ps{+/-semitones}.wav
- Skips files that already look augmented (_ps... in stem)
"""
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

ROOT = Path("/Users/mayagambhir/3600_final")
SRC = ROOT / "data/raw/notes"
SR = 16000
# Include octaves and musically common intervals
SEMITONES = [-12, -7, -5, -3, 3, 5, 7, 12]


def norm_peak(y: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(y)) + 1e-9)
    if peak > 1.0:
        y = y / peak
    return y.astype(np.float32)


def is_augmented_path(p: Path) -> bool:
    return "_ps" in p.stem


def main() -> None:
    wavs = sorted(SRC.rglob("*.wav"))
    if not wavs:
        print(f"No WAV files found under {SRC}")
        return
    written = 0
    skipped = 0
    for wav in wavs:
        if is_augmented_path(wav):
            skipped += 1
            continue
        try:
            y, _ = librosa.load(str(wav), sr=SR, mono=True)
        except Exception as e:
            print(f"[WARN] load failed {wav}: {e}")
            continue
        for s in SEMITONES:
            try:
                y2 = librosa.effects.pitch_shift(y, sr=SR, n_steps=s)
                y2 = norm_peak(y2)
                out = wav.with_name(f"{wav.stem}_ps{s:+d}.wav")
                out.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(out), y2, SR)
                written += 1
            except Exception as e:
                print(f"[WARN] augment failed {wav} semitones {s}: {e}")
                continue
    print(f"Augmentation done. New files: {written}, skipped existing augmented: {skipped}")


if __name__ == "__main__":
    main()


