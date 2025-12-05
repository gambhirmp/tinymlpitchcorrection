#!/usr/bin/env python3
"""
Organize existing artifacts/listen WAVs into per-clip folders:
  artifacts/listen/{group}/{clip_id}/original.wav
  artifacts/listen/{group}/{clip_id}/teacher_lp{X}Hz.wav

Group is inferred from the raw path in metadata/clips.csv if available,
falling back to substring heuristics on the clip_id.
"""
from pathlib import Path
import csv
import shutil

ROOT = Path("/Users/mayagambhir/3600_final")
META = ROOT / "metadata"
LISTEN = ROOT / "artifacts/listen"


def load_clip_map():
    m = {}
    p = META / "clips.csv"
    if not p.exists():
        return m
    with open(p, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            m[row["clip_id"]] = (ROOT / row["path_rel"]).as_posix()
    return m


def infer_group_from_path(raw_path: str, clip_id: str) -> str:
    lower = (raw_path or "").lower() + " " + clip_id.lower()
    if "twinkle" in lower:
        return "twinkle"
    if "scale" in lower or "scales" in lower:
        return "scales"
    if "tune" in lower or "tunes" in lower:
        return "tunes"
    return "notes"


def main():
    clip_map = load_clip_map()
    wavs = sorted(LISTEN.glob("*.wav"))
    moved = 0
    for wav in wavs:
        name = wav.stem  # e.g., "{clip_id}_teacher_lp4Hz" or "{clip_id}_original"
        # Split into clip_id and suffix by finding last _teacher_ or _original
        suffix = ""
        clip_id = name
        if "_teacher_lp" in name:
            idx = name.rfind("_teacher_lp")
            clip_id, suffix = name[:idx], name[idx + 1 :]  # keep suffix like "teacher_lp4Hz"
        elif name.endswith("_original"):
            idx = name.rfind("_original")
            clip_id, suffix = name[:idx], "original"

        raw_path = clip_map.get(clip_id, "")
        group = infer_group_from_path(raw_path, clip_id)
        dest_dir = LISTEN / group / clip_id
        dest_dir.mkdir(parents=True, exist_ok=True)

        if suffix == "original":
            dest = dest_dir / "original.wav"
        elif suffix.startswith("teacher_lp"):
            dest = dest_dir / f"{suffix}.wav"
        else:
            # Unknown pattern; keep name inside the folder as-is
            dest = dest_dir / f"{name}.wav"

        try:
            shutil.move(str(wav), str(dest))
            moved += 1
        except Exception:
            # If already exists or failed, skip
            pass

    print(f"Moved {moved} files into structured folders under artifacts/listen/")


if __name__ == "__main__":
    main()


