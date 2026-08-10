"""
Live demo: edge ML fault detection for power transformers via acoustic emission.

Real inference, both tracks:
  Track 1 -- LSTM autoencoder (TFLite), anomaly gate
  Track 2 -- XGBoost, fault classification (only runs if Track 1 flags a fault)

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import json
import io

import joblib
import numpy as np
import streamlit as st
import xgboost as xgb
from ai_edge_litert.interpreter import Interpreter
import plotly.graph_objects as go

from preprocess_signal import preprocess_signal
from feature_extraction import extract_features, features_to_vector
from feature_extraction_track1 import prepare_track1_input

# ---------------------------------------------------------------------------
# Page setup + styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Transformer Fault Detection — Live Demo",
    page_icon="\u26a1",
    layout="wide",
)

NAVY = "#16213E"
STEEL = "#3E5C76"
AMBER = "#E8A33D"
TEAL = "#2E9E86"
CORAL = "#C4453F"
LIGHT = "#F5F7FA"

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {NAVY}; }}
    section[data-testid="stSidebar"] {{ background-color: #101a33; }}
    h1, h2, h3, p, label, .stMarkdown, span {{ color: #F5F7FA !important; }}
    .stat-card {{
        background-color: #1E2C4F;
        border-radius: 10px;
        padding: 18px 20px;
        border: 1px solid #33456b;
        text-align: center;
    }}
    .stat-value {{ font-size: 30px; font-weight: 700; color: {AMBER}; }}
    .stat-label {{ font-size: 13px; color: #A9B6C9; margin-top: 4px; }}
    .verdict-healthy {{
        background-color: rgba(46,158,134,0.15); border: 1px solid {TEAL};
        border-radius: 10px; padding: 20px; text-align: center;
    }}
    .verdict-fault {{
        background-color: rgba(196,69,63,0.15); border: 1px solid {CORAL};
        border-radius: 10px; padding: 20px; text-align: center;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Model loading (cached so this only happens once)
# ---------------------------------------------------------------------------
MODELS_DIR = "models"


@st.cache_resource
def load_pipeline():
  interp = Interpreter(model_path=f"{MODELS_DIR}/track1_k20_v3_native.tflite")
  interp.allocate_tensors()
  
  xgb_model = xgb.Booster()
  xgb_model.load_model(f"{MODELS_DIR}/xgb_production_model.json")

  scaler = joblib.load(f"{MODELS_DIR}/track1_temporal_scaler_v3.pkl")
  ranked_indices = np.load(f"{MODELS_DIR}/ranked_indices_temporal_v3.npy")
  top_k_indices = ranked_indices[:20]
  
  with open(f"{MODELS_DIR}/threshold.json") as f:
      tau = json.load(f)["track1_roc_optimal_threshold"]
  
  with open(f"{MODELS_DIR}/label_mapping.json") as f:
      label_map = json.load(f)
  
  return {
          "interp": interp,
          "in_detail": interp.get_input_details()[0],
          "out_detail": interp.get_output_details()[0],
          "xgb_model": xgb_model,
          "scaler": scaler,
          "top_k_indices": top_k_indices,
        "tau": tau,
        "label_map": label_map,
    }


@st.cache_resource
def load_demo_samples():
    return dict(np.load(f"{MODELS_DIR}/demo_samples.npz"))


pipeline = load_pipeline()
demo_samples = load_demo_samples()

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def run_track1(preprocessed_signal, state):
    x = prepare_track1_input(preprocessed_signal, state["scaler"], state["top_k_indices"])
    state["interp"].set_tensor(state["in_detail"]["index"], x)
    state["interp"].invoke()
    recon = state["interp"].get_tensor(state["out_detail"]["index"])
    mse = float(np.mean((x - recon) ** 2))
    is_fault = mse > state["tau"]
    return is_fault, mse


def run_track2(preprocessed_signal, state):
    feats = extract_features(preprocessed_signal)
    vec = np.array(features_to_vector(feats), dtype=np.float32).reshape(1, -1)
    dmat = xgb.DMatrix(vec, feature_names=list(feats.keys()))
    probs = state["xgb_model"].predict(dmat)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = state["label_map"][str(pred_idx)]
    class_probs = {state["label_map"][str(i)]: float(p) for i, p in enumerate(probs)}
    return pred_label, class_probs, feats


def diagnose(preprocessed_signal, state):
    is_fault, mse = run_track1(preprocessed_signal, state)
    result = {"track1_error": mse, "tau": state["tau"], "is_fault": is_fault}
    if is_fault:
        pred_label, class_probs, feats = run_track2(preprocessed_signal, state)
        result.update({"fault_type": pred_label, "class_probabilities": class_probs, "features": feats})
    return result


# ---------------------------------------------------------------------------
# Sidebar — signal selection
# ---------------------------------------------------------------------------
st.sidebar.markdown("## Signal source")
mode = st.sidebar.radio(
    "Choose an input", ["Built-in example", "Upload a signal"], label_visibility="collapsed"
)

raw_or_preprocessed = None
signal_label = None
needs_preprocessing = False

if mode == "Built-in example":
    choice = st.sidebar.selectbox(
        "Example signal",
        ["Healthy", "Partial discharge", "Arcing", "Mechanical wear"],
    )
    key_map = {
        "Healthy": "healthy",
        "Partial discharge": "partial_discharge",
        "Arcing": "arcing",
        "Mechanical wear": "mechanical",
    }
    raw_or_preprocessed = demo_samples[key_map[choice]]
    signal_label = choice
    needs_preprocessing = False  # bundled samples are already preprocessed
    st.sidebar.caption("These four windows are drawn from the actual validation set.")
else:
    uploaded = st.sidebar.file_uploader("Upload a .npy file (60,000-sample raw AE window)", type=["npy"])
    if uploaded is not None:
        raw_or_preprocessed = np.load(io.BytesIO(uploaded.read()))
        signal_label = uploaded.name
        needs_preprocessing = True
        st.sidebar.caption("Uploaded signals are treated as raw and run through the full preprocessing pipeline (bandpass 20–400 kHz + standardization).")

st.sidebar.markdown("---")
st.sidebar.markdown("### System summary")
st.sidebar.markdown(
    f"""
    **AUC** 0.999 &nbsp;·&nbsp; **F1** 0.993 (Track 1)
    **Macro F1** 0.9139 ± 0.0087 (Track 2, nested CV)
    **Model size** 262.7 KB &nbsp;·&nbsp; **Target** Raspberry Pi 4B
    """
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("# \u26a1 Transformer Fault Detection — Live Demo")
st.markdown(
    "Acoustic emission signal in, verdict out — real inference through both tracks of the edge pipeline."
)
st.markdown("")

if raw_or_preprocessed is None:
    st.info("Choose a built-in example signal or upload one from the sidebar to run the pipeline.")
    st.stop()

# ---------------------------------------------------------------------------
# Run the pipeline
# ---------------------------------------------------------------------------
if needs_preprocessing:
    clean_signal = preprocess_signal(raw_or_preprocessed)
else:
    clean_signal = raw_or_preprocessed

with st.spinner("Running Track 1 and Track 2..."):
    result = diagnose(clean_signal, pipeline)

# ---------------------------------------------------------------------------
# Waveform plot
# ---------------------------------------------------------------------------
st.markdown(f"### Signal: {signal_label}")
fig = go.Figure()
t = np.arange(len(clean_signal)) / 1_000_000 * 1000  # ms
fig.add_trace(go.Scatter(x=t, y=clean_signal, line=dict(color=AMBER, width=0.8), name="AE signal"))
fig.update_layout(
    height=260,
    margin=dict(l=10, r=10, t=10, b=10),
    plot_bgcolor="#1E2C4F",
    paper_bgcolor="#1E2C4F",
    font=dict(color="#F5F7FA"),
    xaxis_title="Time (ms)",
    yaxis_title="Amplitude (standardized)",
    xaxis=dict(gridcolor="#33456b"),
    yaxis=dict(gridcolor="#33456b"),
)
st.plotly_chart(fig, width='stretch')

# ---------------------------------------------------------------------------
# Track 1 result
# ---------------------------------------------------------------------------
st.markdown("### Track 1 — anomaly detection")
col1, col2 = st.columns([1, 2])

with col1:
    if result["is_fault"]:
        st.markdown(
            f"""<div class="verdict-fault">
                <div style="font-size:22px; font-weight:700; color:{CORAL};">FAULT DETECTED</div>
                <div style="margin-top:6px; color:#A9B6C9;">Reconstruction error above threshold</div>
                </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""<div class="verdict-healthy">
                <div style="font-size:22px; font-weight:700; color:{TEAL};">HEALTHY</div>
                <div style="margin-top:6px; color:#A9B6C9;">Reconstruction error within normal range</div>
                </div>""",
            unsafe_allow_html=True,
        )

with col2:
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=result["track1_error"],
            number={"suffix": "", "font": {"color": "#F5F7FA"}},
            gauge={
                "axis": {"range": [0, max(20, result["track1_error"] * 1.2)], "tickcolor": "#A9B6C9"},
                "bar": {"color": CORAL if result["is_fault"] else TEAL},
                "steps": [
                    {"range": [0, result["tau"]], "color": "#274a3f"},
                    {"range": [result["tau"], max(20, result["track1_error"] * 1.2)], "color": "#4a2b28"},
                ],
                "threshold": {
                    "line": {"color": AMBER, "width": 3},
                    "thickness": 0.8,
                    "value": result["tau"],
                },
            },
        )
    )
    gauge.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="#1E2C4F",
        font=dict(color="#F5F7FA"),
    )
    st.plotly_chart(gauge, width='stretch')
    st.caption(f"Reconstruction error (MSE) = {result['track1_error']:.4f}  ·  \u03c4* = {result['tau']}")

# ---------------------------------------------------------------------------
# Track 2 result (only if Track 1 flagged a fault)
# ---------------------------------------------------------------------------
if result["is_fault"]:
    st.markdown("### Track 2 — fault classification")
    probs = result["class_probabilities"]
    labels = list(probs.keys())
    values = [probs[k] for k in labels]

    bar = go.Figure(
        go.Bar(
            x=values,
            y=[l.replace("_", " ").title() for l in labels],
            orientation="h",
            marker_color=[AMBER if l == result["fault_type"] else STEEL for l in labels],
            text=[f"{v:.1%}" for v in values],
            textposition="outside",
        )
    )
    bar.update_layout(
        height=220,
        margin=dict(l=10, r=40, t=10, b=10),
        plot_bgcolor="#1E2C4F",
        paper_bgcolor="#1E2C4F",
        font=dict(color="#F5F7FA"),
        xaxis=dict(range=[0, 1], gridcolor="#33456b", tickformat=".0%"),
        yaxis=dict(gridcolor="#33456b"),
    )
    st.plotly_chart(bar, width='stretch')
    st.markdown(
        f"**Predicted fault type:** <span style='color:{AMBER}; font-weight:700;'>"
        f"{result['fault_type'].replace('_', ' ').title()}</span>",
        unsafe_allow_html=True,
    )
else:
    st.caption("Track 2 does not run — Track 1 already confirmed the signal is healthy.")

# ---------------------------------------------------------------------------
# Footer stat strip
# ---------------------------------------------------------------------------
st.markdown("---")
c1, c2, c3, c4 = st.columns(4)
for col, value, label in zip(
    [c1, c2, c3, c4],
    ["0.999", "0.9139 \u00b1 0.0087", "262.7 KB", "Raspberry Pi 4B"],
    ["Track 1 AUC", "Track 2 macro F1 (nested CV)", "Track 1 model size", "Target hardware"],
):
    with col:
        st.markdown(
            f"""<div class="stat-card">
                <div class="stat-value">{value}</div>
                <div class="stat-label">{label}</div>
                </div>""",
            unsafe_allow_html=True,
        )
