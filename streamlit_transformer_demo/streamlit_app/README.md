# Transformer Fault Detection — Live Demo

Real inference through both tracks of the edge ML pipeline, using the actual
trained models (not simulated).

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at http://localhost:8501

## What it does

- **Sidebar**: pick one of four built-in example signals (Healthy, Partial
  discharge, Arcing, Mechanical wear — drawn from the real validation set),
  or upload your own raw `.npy` signal (60,000 samples).
- **Track 1** (LSTM autoencoder, TFLite): computes real reconstruction error
  and compares against τ* = 2.708. Shown as a gauge.
- **Track 2** (XGBoost): only runs if Track 1 flags a fault. Shows real class
  probabilities for partial discharge / arcing / mechanical wear.
- Footer strip shows the headline reported metrics (AUC, macro F1, model
  size, target hardware).

## Files

- `app.py` — the Streamlit app
- `preprocess_signal.py`, `feature_extraction.py`, `feature_extraction_track1.py`
  — your original pipeline scripts, used unmodified
- `models/` — your trained model files (tflite, xgboost json, scaler,
  ranked indices, threshold, label mapping) and the four bundled demo signals

## Notes

- The four built-in example signals are already preprocessed (bandpass +
  standardized), so the app skips `preprocess_signal.py` for those and runs
  it only on raw uploads.
- Verified end-to-end before delivery: all four example classes produce the
  correct Track 1 verdict and, where applicable, the correct Track 2 fault
  type.
