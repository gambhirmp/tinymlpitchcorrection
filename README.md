# Pitch Correction 

This shows how to add data, augment/transpose, preprocess, train, where the TFLite ends up, and what its inputs/outputs are.

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

# 4) Train/export full melody (TCN → int8 TFLite) using full_melody_correction.ipynb

# Artifacts
ls -lh artifacts/full/tflite/full_melody.tflite
```

Artifacts (full melody):
- SavedModel: `artifacts/full/saved_model/`
- TFLite: `artifacts/full/tflite/full_melody.tflite`

TFLite I/O (see also `metadata/tflite_metadata.json`):
- Input: `features` int8, shape `[1, 1, 64]` (one normalized mel frame)
- Outputs: `shift_cents` int8 `[1,1,1]` (~2 cents/LSB), `confidence` int8 `[1,1,1]`
- Feature settings: sr=16000, window=512, hop=40, n_mels=32; normalize using `metadata/feature_norm.json`

