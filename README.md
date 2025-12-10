# Pitch Correction – Quickstart (Full Melody + Baseline)

It shows how to add data, augment/transpose, preprocess, train, where the TFLite ends up, and what its inputs/outputs are.

## TL;DR – Full Melody (commands)

```bash
# 0) Create/activate Python 3.11 venv (once)
/opt/homebrew/opt/python@3.11/bin/python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# 1) Put WAVs under data/raw/{notes,scales,tunes}/... (16 kHz mono recommended)
# Optional convert from m4a/mp3:
# ffmpeg -i input.m4a -ar 16000 -ac 1 -sample_fmt s16 output.wav

# 2) Augment/transpose (applies to notes, scales, twinkle/tunes if present)
# Writes siblings with suffix _ps{+/-semitones}.wav
python scripts/augment_transpose.py

# 3) Preprocess → features + teacher labels (.npz)
# Note: augmented (_ps*) are forced into the train split
python scripts/preprocess.py

# 4) Train/export full melody (TCN → int8 TFLite)
jupyter lab   # run notebooks/full_melody_correction.ipynb (Run All)
# or batch:
python -m jupyter nbconvert --to notebook --execute \
  notebooks/full_melody_correction.ipynb \
  --ExecutePreprocessor.timeout=-1 \
  --output notebooks/full_melody_correction.executed.ipynb

# Artifacts
ls -lh artifacts/full/tflite/full_melody.tflite
```

Artifacts (full melody):
- SavedModel: `artifacts/full/saved_model/`
- TFLite: `artifacts/full/tflite/full_melody.tflite`

TFLite I/O (see also `metadata/tflite_metadata.json`):
- Input: `features` int8, shape `[1, 1, 64]` (one normalized mel frame)
- Outputs: `shift_cents` int8 `[1,1,1]` (~2 cents/LSB), `confidence` int8 `[1,1,1]`
- Feature settings: sr=16000, window=512, hop=160, n_mels=64; normalize using `metadata/feature_norm.json`

---

# Baseline Pitch Correction – Launchpad

```bash
# 0) Activate env
source ./.venv311/bin/activate

# 1) Put WAVs under data/raw/notes/... (16 kHz mono recommended)
# Optional convert from m4a:
# ffmpeg -i input.m4a -ar 16000 -ac 1 -sample_fmt s16 output.wav

# 2) augmentation/make some extra data (transpose notes to new keys/octaves)
python scripts/augment_transpose.py

# 3) Preprocess → features + teacher labels (.npz)
python scripts/preprocess.py

# 4) Train/export baseline
jupyter lab   # run notebooks/baseline_single_note.ipynb
# or batch:
jupyter nbconvert --to notebook --execute \
  notebooks/baseline_single_note.ipynb \
  --ExecutePreprocessor.timeout=7200 \
  --output notebooks/baseline_single_note.executed.ipynb
```

## Data: upload and organize

- Put audio in `data/raw/notes/<NOTE>/.../*.wav`. Best: 16 kHz mono, PCM16 WAV.
- If your recordings are `.m4a`, decode to WAV before preprocessing (see ffmpeg command above).
- You can nest singer/take folders (the pipeline scans recursively).

## Augmentation (optional)

- Run `scripts/augment_transpose.py` to pitch‑shift existing WAVs by semitone steps (includes ±12 for octaves).
- New files are written alongside originals with suffix `_ps+N.wav` or `_ps-N.wav`.
- Re‑run preprocessing afterwards so labels match the augmented audio.

## Preprocessing (what it produces)

Command: `python scripts/preprocess.py`
- Resample to 16 kHz, normalize loudness, trim silence.
- Compute 64‑band log‑mel features at 100 fps (32 ms window = 512 samples; 10 ms hop = 160 samples).
- Estimate per‑frame f0 with pYIN; convert to cents.
- Build vibrato‑preserving teacher labels:
  - trend = low‑pass(f0_cents, ≈4 Hz); snapped = chromatic snap(trend)
  - target_shift = snapped − trend (cents), zero on unvoiced
- Write one `.npz` per clip to `data/processed/features/{train,val,test}/` with:
  - `logmel [T,64]`, `f0_cents [T]`, `voiced [T]`, `target_shift [T]`
- Update metadata:
  - `metadata/feature_norm.json` (mean/std, sr=16000, fps=100, window/hop, n_mels)
  - `metadata/clips.csv` (index), `metadata/splits.json` (train/val/test)

## Train the baseline

Open `notebooks/baseline_single_note.ipynb` and “Run All”. It:
- Loads/normalizes mel windows; filters to sustained notes.
- Trains a tiny GRU to predict `target_shift` (cents) per frame.
- Exports both SavedModel and TFLite artifacts.

Artifacts:
- SavedModel: `artifacts/baseline/saved_model`
- TFLite: `artifacts/baseline/tflite/baseline_single_note.tflite`

## TFLite model I/O

- File: `artifacts/baseline/tflite/baseline_single_note.tflite`
- Ops: uses SELECT TF OPS (Flex) because of GRU; the app must include the Select TF Ops delegate.
- Input (streaming per hop):
  - name: `features`
  - shape: `[1, 1, 64]` (float32)
  - content: one normalized log‑mel frame (normalize with `feature_norm.json` mean/std)
- Output:
  - `shift_cents`: `[1, 1, 1]` float32 — per‑frame correction in cents (≈ −300..+300 typical)
  - `confidence`: `[1, 1, 1]` float32 — optional gating strength in [0,1]
- Feature settings (must match phone feature extractor):
  - sr=16000, window=512 (Hann ~32 ms), hop=160 (10 ms), n_mels=64, dB/log power

## Deployment..?

Useful checks:
- Numeric sanity (SavedModel): run the quick tests in the notebook or use the scripts in `scripts/` to render A/B audio and plots under `artifacts/listen` and `artifacts/plots`.

---
