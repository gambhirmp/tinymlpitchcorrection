#!/usr/bin/env python3
"""
Augment raw WAVs by pitch-shifting to new notes and octaves.
- Input roots:
  - data/raw/notes/**/*.wav
  - data/raw/scales/**/*.wav (if present) or data/raw/notes/scales/*.wav
  - data/raw/tunes/**/*.wav (if present) or data/raw/notes/twinkle*/**/*.wav
- Output: same folders with filename suffix _ps{+/-semitones}.wav
- Skips files that already look augmented (_ps... in stem)
"""
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

ROOT = Path("/Users/mayagambhir/3600_final")
NOTES = ROOT / "data/raw/notes"
SCALES = ROOT / "data/raw/scales"
TUNES = ROOT / "data/raw/tunes"
SR = 16000
# includes octaves and musically common intervals
SEMITONES = [-12, -7, -5, -3, 3, 5, 7, 12]


# normalizes the audio to the peak amplitude
def norm_peak(y: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(y)) + 1e-9)
    if peak > 1.0:
        y = y / peak
    return y.astype(np.float32)

# checks if the file is already augmented
def is_augmented_path(p: Path) -> bool:
    return "_ps" in p.stem


def main() -> None:
    # aggregates all candidate roots that exist
    roots = []
    for r in [NOTES, SCALES, TUNES]:
        if r.exists():
            roots.append(r)
    # fallbacks for datasets placed under notes/
    nested_scales = NOTES / "scales"
    if nested_scales.exists():
        roots.append(nested_scales)
    for name in ["twinkle twinkle little star", "twinkle", "tunes"]:
        nested_tunes = NOTES / name
        if nested_tunes.exists():
            roots.append(nested_tunes)

    # de-duplicates the files
    seen = set()
    wavs: list[Path] = []
    for r in roots:
        for p in r.rglob("*.wav"):
            if p.resolve() not in seen:
                seen.add(p.resolve())
                wavs.append(p)

    wavs = sorted(wavs)
    if not wavs:
        print(f"No WAV files found under any roots: {roots or '[none]'}")
        return
    written = 0
    skipped = 0
    # augments the wav files
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


