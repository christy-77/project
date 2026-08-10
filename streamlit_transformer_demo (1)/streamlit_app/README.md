# AE Fault Monitor — Control Room View

A continuously-running live monitoring panel, not a click-through demo.
The screen streams signals on its own — mostly healthy, with faults injected
randomly (you don't pick them) — and visibly runs the two-stage pipeline on
each: Track 1 (anomaly gate) first, then Track 2 (fault classification) only
if Track 1 flags something.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at http://localhost:8501 and starts streaming immediately — nothing to
click. Just leave it open on a second monitor/projector during the defense.

## What's real vs. what's staged

- **Real inference**: every number shown (Track 1 reconstruction error, Track
  2 class probabilities, fault type) comes from your actual trained models
  (`track1_k20_v3_native.tflite`, `xgb_production_model.json`), run for real
  on your actual `preprocess_signal.py` / `feature_extraction*.py` pipeline.
- **Staged for a smooth live demo**: rather than calling the models fresh on
  every 1.1s screen tick (slow, and risks a stall mid-defense), the app
  precomputes ~240 signals once at startup — your 4 real validated example
  signals with light jitter (small noise + shift) applied for variety, each
  one run through the real models — then plays that back smoothly. This was
  verified: jitter doesn't change any signal's true classification (see the
  build process notes if you want to re-check).
- **You never pick the fault** — a random step is chosen to be a fault
  (~1 in 9 signals, randomly one of the three fault types), same as
  real monitoring.

## Panel layout

- System status (MONITORING / FAULT, blinking on alert)
- Signals monitored / faults detected counters
- Live waveform (green = healthy, red = fault)
- Track 1 and Track 2 status boxes that visibly light up in sequence
- Fault probability bars (only populate once Track 2 finishes)
- Fault log — timestamp, fault type, confidence — **fault events only**,
  healthy signals are not logged (matches what a control-room engineer
  actually wants to see)

## Files

- `app.py` — the app
- `preprocess_signal.py`, `feature_extraction.py`, `feature_extraction_track1.py`
  — your original pipeline scripts, unmodified
- `models/` — trained model files + the four real demo signals
