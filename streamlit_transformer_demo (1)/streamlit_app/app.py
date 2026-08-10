"""
Live control-room demo: edge ML fault detection for power transformers via
acoustic emission.

Continuously streams signals (mostly healthy, with rare random faults),
running REAL inference through both tracks:
  Track 1 -- LSTM autoencoder (TFLite), anomaly gate
  Track 2 -- XGBoost, fault classification (only runs if Track 1 flags a fault)

All numbers shown come from real model inference, computed once at startup
on jittered variants of the real validated signals, then replayed smoothly
so the live view never stalls waiting on inference.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import os
import json
from datetime import datetime

import joblib
import numpy as np
import streamlit as st
import xgboost as xgb
from ai_edge_litert.interpreter import Interpreter
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# ---------------------------------------------------------------------------
# Page setup + dark terminal styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AE Fault Monitor — Control Room",
    page_icon="\u26a1",
    layout="wide",
)

BG = "#080B08"
PANEL = "#0E1410"
BORDER = "#1E3324"
GREEN = "#39FF6A"
GREEN_DIM = "#1E8A45"
GREEN_MUTED = "#5C8A6B"
AMBER = "#FFB020"
RED = "#FF4B3E"
TEXT_MUTED = "#6B8F76"

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {BG}; }}
    section[data-testid="stSidebar"] {{ background-color: {PANEL}; border-right: 1px solid {BORDER}; }}
    * {{ font-family: 'Consolas', 'Menlo', 'Courier New', monospace !important; }}
    h1, h2, h3, p, label, span, div {{ color: {GREEN} !important; }}
    .panel {{
        background-color: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 16px 20px;
    }}
    .panel-title {{
        font-size: 11px; letter-spacing: 2px; color: {TEXT_MUTED} !important;
        margin-bottom: 8px; text-transform: uppercase;
    }}
    .status-monitoring {{ color: {GREEN} !important; font-size: 30px; font-weight: 700; }}
    .status-fault {{ color: {RED} !important; font-size: 30px; font-weight: 700; }}
    .track-box {{
        border: 1px solid {BORDER}; border-radius: 6px; padding: 12px 16px;
        text-align: center; transition: all 0.3s;
    }}
    .track-idle {{ opacity: 0.35; }}
    .track-active {{ border-color: {AMBER}; box-shadow: 0 0 12px rgba(255,176,32,0.35); }}
    .track-healthy {{ border-color: {GREEN}; box-shadow: 0 0 12px rgba(57,255,106,0.3); }}
    .track-fault {{ border-color: {RED}; box-shadow: 0 0 12px rgba(255,75,62,0.35); }}
    .log-fault {{ color: {RED} !important; }}
    .log-time {{ color: {TEXT_MUTED} !important; }}
    .blink {{ animation: blink 1.1s infinite; }}
    @keyframes blink {{ 50% {{ opacity: 0.25; }} }}
    hr {{ border-color: {BORDER}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# ---------------------------------------------------------------------------
# Model + pipeline loading
# ---------------------------------------------------------------------------


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


def run_track1(preprocessed_signal, state, prepare_fn):
    x = prepare_fn(preprocessed_signal, state["scaler"], state["top_k_indices"])
    state["interp"].set_tensor(state["in_detail"]["index"], x)
    state["interp"].invoke()
    recon = state["interp"].get_tensor(state["out_detail"]["index"])
    mse = float(np.mean((x - recon) ** 2))
    return mse > state["tau"], mse


def run_track2(preprocessed_signal, state, extract_fn, vec_fn):
    feats = extract_fn(preprocessed_signal)
    vec = np.array(vec_fn(feats), dtype=np.float32).reshape(1, -1)
    dmat = xgb.DMatrix(vec, feature_names=list(feats.keys()))
    probs = state["xgb_model"].predict(dmat)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = state["label_map"][str(pred_idx)]
    class_probs = {state["label_map"][str(i)]: float(p) for i, p in enumerate(probs)}
    return pred_label, class_probs


# ---------------------------------------------------------------------------
# Precompute a real-inference stream (jittered real signals, cached once)
# ---------------------------------------------------------------------------


@st.cache_resource
def build_stream(_pipeline, n_steps=240, fault_every_avg=9, seed=7):
    from feature_extraction_track1 import prepare_track1_input
    from feature_extraction import extract_features, features_to_vector

    samples = dict(np.load(f"{MODELS_DIR}/demo_samples.npz"))
    fault_classes = ["arcing", "mechanical", "partial_discharge"]
    rng = np.random.default_rng(seed)

    def jitter(sig):
        noise = rng.normal(0, 0.03, size=sig.shape).astype(np.float32)
        shift = int(rng.integers(-200, 200))
        return np.roll(sig, shift) + noise

    stream = []
    for i in range(n_steps):
        is_fault_step = rng.integers(0, fault_every_avg) == 0
        cls = rng.choice(fault_classes) if is_fault_step else "healthy"
        sig = jitter(samples[cls])

        is_fault, mse = run_track1(sig, _pipeline, prepare_track1_input)
        entry = {"class": cls, "signal": sig, "track1_mse": mse, "is_fault": is_fault}

        if is_fault:
            fault_type, class_probs = run_track2(
                sig, _pipeline, extract_features, features_to_vector
            )
            entry["fault_type"] = fault_type
            entry["class_probs"] = class_probs

        stream.append(entry)

    return stream


pipeline = load_pipeline()
stream = build_stream(pipeline)

# ---------------------------------------------------------------------------
# Session state: position in stream + sub-phase + running log
# ---------------------------------------------------------------------------
if "pos" not in st.session_state:
    st.session_state.pos = 0
    st.session_state.phase = 0
    st.session_state.log = []
    st.session_state.logged_positions = set()
    st.session_state.total_seen = 0
    st.session_state.total_faults = 0

st_autorefresh(interval=1100, key="tick")

current = stream[st.session_state.pos % len(stream)]
is_fault = current["is_fault"]
max_phase = 3 if is_fault else 1

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("## \u26a1 AE FAULT MONITOR — CONTROL ROOM VIEW")
st.markdown("")

col_status, col_stats = st.columns([1.3, 2])
with col_status:
    fault_now = is_fault and st.session_state.phase >= 1
    status_class = "status-fault blink" if fault_now else "status-monitoring"
    status_text = "FAULT" if fault_now else "MONITORING"
    st.markdown(
        f"""<div class="panel">
            <div class="panel-title">System status</div>
            <div class="{status_class}">{status_text}</div>
            </div>""",
        unsafe_allow_html=True,
    )
with col_stats:
    c1, c2, c3 = st.columns(3)
    for col, val, lab in zip(
        [c1, c2, c3],
        [st.session_state.total_seen, st.session_state.total_faults, "RPi 4B"],
        ["Signals monitored", "Faults detected", "Deployed on"],
    ):
        with col:
            st.markdown(
                f"""<div class="panel" style="text-align:center;">
                    <div class="panel-title">{lab}</div>
                    <div style="font-size:26px; font-weight:700;">{val}</div>
                    </div>""",
                unsafe_allow_html=True,
            )

st.markdown("")

# ---------------------------------------------------------------------------
# Main row: waveform + track flow  |  fault probabilities
# ---------------------------------------------------------------------------
left, right = st.columns([1.6, 1])

with left:
    st.markdown('<div class="panel-title">Live signal waveform</div>', unsafe_allow_html=True)
    sig = current["signal"]
    t = np.arange(0, len(sig), 40)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t, y=sig[t],
            line=dict(color=RED if fault_now else GREEN, width=1.1),
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor=PANEL,
        paper_bgcolor=PANEL,
        font=dict(color=GREEN_MUTED, size=10),
        xaxis=dict(gridcolor=BORDER, showticklabels=False),
        yaxis=dict(gridcolor=BORDER, showticklabels=False),
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.markdown("")
    tcol1, tcol2 = st.columns(2)
    with tcol1:
        if st.session_state.phase == 0:
            cls = "track-active"
            label = "ANALYZING\u2026"
        elif is_fault:
            cls = "track-fault"
            label = "FAULT DETECTED"
        else:
            cls = "track-healthy"
            label = "HEALTHY"
        st.markdown(
            f"""<div class="track-box {cls}">
                <div class="panel-title">Track 1 \u00b7 Anomaly Gate</div>
                <div style="font-size:16px; font-weight:700;">{label}</div>
                </div>""",
            unsafe_allow_html=True,
        )
    with tcol2:
        if not is_fault:
            cls, label = "track-idle", "IDLE — signal healthy"
        elif st.session_state.phase < 2:
            cls, label = "track-idle", "WAITING\u2026"
        elif st.session_state.phase == 2:
            cls, label = "track-active", "ANALYZING\u2026"
        else:
            cls, label = "track-fault", current.get("fault_type", "").replace("_", " ").upper()
        st.markdown(
            f"""<div class="track-box {cls}">
                <div class="panel-title">Track 2 \u00b7 Fault Classification</div>
                <div style="font-size:16px; font-weight:700;">{label}</div>
                </div>""",
            unsafe_allow_html=True,
        )

with right:
    st.markdown('<div class="panel-title">Last fault probabilities</div>', unsafe_allow_html=True)
    if is_fault and st.session_state.phase >= 3:
        probs = current["class_probs"]
        bar = go.Figure(
            go.Bar(
                x=[probs[k] for k in probs],
                y=[k.replace("_", " ").title() for k in probs],
                orientation="h",
                marker_color=[AMBER if k == current["fault_type"] else GREEN_DIM for k in probs],
                text=[f"{probs[k]:.0%}" for k in probs],
                textposition="outside",
                textfont=dict(color=GREEN),
            )
        )
        bar.update_layout(
            height=220,
            margin=dict(l=10, r=40, t=10, b=10),
            plot_bgcolor=PANEL,
            paper_bgcolor=PANEL,
            font=dict(color=GREEN_MUTED, size=11),
            xaxis=dict(range=[0, 1], gridcolor=BORDER, tickformat=".0%"),
            yaxis=dict(gridcolor=BORDER),
        )
        st.plotly_chart(bar, width="stretch", config={"displayModeBar": False})
    else:
        st.markdown(
            f"""<div class="panel" style="height:220px; display:flex; align-items:center;
                 justify-content:center; color:{TEXT_MUTED} !important;">
                 no active fault</div>""",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Fault log (fault events only)
# ---------------------------------------------------------------------------
st.markdown("")
st.markdown('<div class="panel-title">Fault log</div>', unsafe_allow_html=True)

if is_fault and st.session_state.phase == max_phase and st.session_state.pos not in st.session_state.logged_positions:
    st.session_state.log.insert(
        0,
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "fault_type": current["fault_type"].replace("_", " ").title(),
            "confidence": max(current["class_probs"].values()),
        },
    )
    st.session_state.log = st.session_state.log[:30]
    st.session_state.logged_positions.add(st.session_state.pos)
    st.session_state.total_faults += 1

if st.session_state.log:
    rows = "".join(
        f"""<div style="display:flex; gap:24px; padding:4px 0; border-bottom:1px solid {BORDER};">
            <span class="log-time" style="width:90px;">{e['time']}</span>
            <span class="log-fault" style="width:200px;">{e['fault_type']}</span>
            <span style="color:{TEXT_MUTED} !important;">{e['confidence']:.0%} confidence</span>
            </div>"""
        for e in st.session_state.log
    )
    st.markdown(f'<div class="panel">{rows}</div>', unsafe_allow_html=True)
else:
    st.markdown(
        f'<div class="panel" style="color:{TEXT_MUTED} !important;">No faults logged yet.</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Advance stream position / phase for next refresh tick
# ---------------------------------------------------------------------------
if st.session_state.phase < max_phase:
    st.session_state.phase += 1
else:
    st.session_state.pos += 1
    st.session_state.phase = 0
    st.session_state.total_seen += 1
