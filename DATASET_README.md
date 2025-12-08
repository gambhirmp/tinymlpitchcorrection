# Dataset layout

Place recorded audio here in the following structure:

```
/Users/mayagambhir/3600_final/
  data/
    raw/
      notes/{singer_id}/{take_id}.wav
      scales/{singer_id}/{take_id}.wav
      tunes/{singer_id}/{take_id}.wav
    interim/
      resampled/...
      trimmed/...
    processed/
      features/
        train/{clip_id}.npz
        val/{clip_id}.npz
        test/{clip_id}.npz
  metadata/
    clips.csv
    splits.json
    feature_norm.json
    tflite_metadata.json
```

Audio requirements:
- Mono WAV files at 16 kHz (or any rate that will be resampled to 16 kHz).
- Reasonable loudness; avoid clipping; record in a quiet room.

Notes:
- `clips.csv` describes every clip and where it came from.
- `splits.json` defines the train/val/test split.
- `feature_norm.json` holds global normalization statistics for features.
- `tflite_metadata.json` describes the model’s expected inputs/outputs on device.




