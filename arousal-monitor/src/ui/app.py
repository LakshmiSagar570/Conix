"""
src/ui/app.py
────────────────────────────────────────────────────────────────
Arousal Monitor — Streamlit dashboard

Layout:
  ┌──────────────────────────────────────┐
  │  Header                              │
  ├─────────────────────┬────────────────┤
  │  Live webcam feed   │  Score + bands │
  │  (streamlit-webrtc) │  Feature cards │
  ├─────────────────────┴────────────────┤
  │  Tension timeline graph              │
  ├──────────────────────────────────────┤
  │  Pattern analysis │ Session stats    │
  ├──────────────────────────────────────┤
  │  Ethical disclaimer                  │
  └──────────────────────────────────────┘

WebRTC handles the webcam in the browser and streams frames
to the Python backend transformer via aiortc.
"""

import sys
import os
import time
import threading
from datetime import datetime

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# ── Path setup ────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.cv.extractor import FacialExtractor, FeatureVector
from src.features.tension import TensionEngine, TensionState
from src.db.client import SessionLogger


# ── WebRTC config (public STUN server — works in Codespaces) ──
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


# ── Shared state (webrtc runs in a separate thread) ───────────
class _SharedState:
    def __init__(self):
        self._lock    = threading.Lock()
        self.features: FeatureVector | None = None

    def write(self, f):
        with self._lock:
            self.features = f

    def read(self) -> FeatureVector | None:
        with self._lock:
            return self.features


_shared = _SharedState()


# ── Video transformer ─────────────────────────────────────────
class ArousalTransformer(VideoTransformerBase):
    """Runs in its own thread. Processes each frame, writes features."""

    def __init__(self):
        self._extractor = FacialExtractor(fps=15)

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        features, annotated = self._extractor.process_frame(img)
        _shared.write(features)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ── Plotly tension graph ──────────────────────────────────────
def _tension_graph(timestamps: list, scores: list) -> go.Figure:
    fig = go.Figure()

    # Band background shading
    for y0, y1, col in [(0, 30, "#4ade80"), (30, 60, "#facc15"), (60, 100, "#f87171")]:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=col, opacity=0.05, line_width=0)

    if timestamps:
        t0 = timestamps[0]
        x  = [t - t0 for t in timestamps]
        y  = list(scores)

        # Main line
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="lines",
            line=dict(color="#60a5fa", width=2.5, shape="spline"),
            fill="tozeroy",
            fillcolor="rgba(96,165,250,0.08)",
            name="Tension Index",
        ))

        # Current value dot
        fig.add_trace(go.Scatter(
            x=[x[-1]], y=[y[-1]],
            mode="markers",
            marker=dict(size=9, color="#60a5fa",
                        line=dict(width=2, color="#0a0f1e")),
            showlegend=False,
        ))
    else:
        fig.add_annotation(
            text="Waiting for calibration…",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(color="#475569", size=13),
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", family="JetBrains Mono, monospace", size=11),
        xaxis=dict(
            title="Elapsed (s)",
            gridcolor="rgba(255,255,255,0.06)",
            showline=False, zeroline=False,
        ),
        yaxis=dict(
            title="Tension Index",
            range=[0, 100],
            gridcolor="rgba(255,255,255,0.06)",
            showline=False, zeroline=False,
            tickvals=[0, 30, 60, 100],
        ),
        margin=dict(l=10, r=10, t=8, b=40),
        height=260,
        showlegend=False,
    )
    return fig


# ── CSS / style ───────────────────────────────────────────────
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace !important;
}

.stApp { background: #080d1a !important; color: #e2e8f0; }

/* hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

.block-container { padding: 1.4rem 2rem 2rem !important; max-width: 1200px; }

.label {
    font-size: 10px;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #334155;
    margin-bottom: 6px;
}

.score-big {
    font-size: 80px;
    font-weight: 600;
    line-height: 1;
    text-align: center;
    transition: color 0.4s ease;
}

.band-text {
    font-size: 11px;
    letter-spacing: 3px;
    text-align: center;
    margin-top: 4px;
}

.metric-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 12px 14px;
    text-align: center;
}

.metric-val {
    font-size: 20px;
    font-weight: 600;
    color: #7dd3fc;
}

.pattern-box {
    background: rgba(96,165,250,0.07);
    border: 1px solid rgba(96,165,250,0.18);
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 12px;
    color: #93c5fd;
    margin-bottom: 8px;
}

.stat-box {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px;
    padding: 14px;
}

.stat-row { font-size: 12px; color: #94a3b8; margin: 4px 0; }

.disclaimer {
    background: rgba(250,204,21,0.04);
    border: 1px solid rgba(250,204,21,0.12);
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 11px;
    color: #64748b;
    text-align: center;
    margin-top: 1rem;
}

.hr { border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 0.8rem 0; }

.db-badge {
    display: inline-block;
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 999px;
    margin-left: 8px;
}
</style>
"""


# ── Main ──────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Arousal Monitor",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(STYLE, unsafe_allow_html=True)

    # ── Session state init ────────────────────────────────────
    if "engine"   not in st.session_state:
        st.session_state.engine    = TensionEngine()
    if "db"       not in st.session_state:
        st.session_state.db        = SessionLogger()
    if "last_log" not in st.session_state:
        st.session_state.last_log  = 0.0

    engine: TensionEngine = st.session_state.engine
    db: SessionLogger     = st.session_state.db

    # ── Header ────────────────────────────────────────────────
    h1, h2 = st.columns([4, 1])
    with h1:
        db_badge = (
            '<span class="db-badge" '
            'style="background:rgba(74,222,128,0.12);color:#4ade80">● DB</span>'
            if db.is_connected() else
            '<span class="db-badge" '
            'style="background:rgba(100,116,139,0.12);color:#475569">● No DB</span>'
        )
        st.markdown(
            f"<h2 style='margin:0;color:#e2e8f0'>🧠 AROUSAL MONITOR{db_badge}</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='color:#334155;font-size:11px;letter-spacing:2px;"
            "margin:-6px 0 0'>REAL-TIME PHYSIOLOGICAL SIGNAL VISUALIZATION</p>",
            unsafe_allow_html=True,
        )
    with h2:
        st.markdown(
            f"<p style='color:#334155;font-size:11px;text-align:right;"
            f"margin-top:16px'>{datetime.now().strftime('%H:%M · %d %b %Y')}</p>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    # ── Row 1: Webcam | Score ─────────────────────────────────
    cam_col, score_col = st.columns([1.2, 1], gap="large")

    with cam_col:
        st.markdown("<div class='label'>LIVE FEED</div>", unsafe_allow_html=True)
        ctx = webrtc_streamer(
            key="arousal-stream",
            video_transformer_factory=ArousalTransformer,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": False},
            async_transform=True,
        )

        # Calibration bar
        if not engine.calibrated:
            prog = engine._calibration_progress()
            if prog > 0:
                st.progress(prog, text=f"Calibrating baseline — {int(prog * 100)}%")
            else:
                st.info("Start video to begin calibration (30 s)", icon="ℹ️")
        else:
            st.success("✓ Baseline calibrated", icon=None)

    with score_col:
        # Read latest features
        features = _shared.read()

        # Update engine
        state: TensionState = engine.update(features)

        # Log to DB every 5 s
        now = time.time()
        if state.calibrated and (now - st.session_state.last_log) >= 5:
            db.log(features, state.score)
            st.session_state.last_log = now

        # Score display
        score_color = state.band_color if state.calibrated else "#1e3a5f"
        label_text  = state.band       if state.calibrated else "CALIBRATING"

        st.markdown(
            f"<div class='label' style='text-align:center;margin-top:12px'>"
            f"TENSION INDEX</div>"
            f"<div class='score-big' style='color:{score_color}'>"
            f"{state.score:.0f}</div>"
            f"<div class='band-text' style='color:{score_color}'>"
            f"{label_text}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Feature metric cards
        if features:
            c1, c2 = st.columns(2)
            c3, c4 = st.columns(2)
            for col, label, val in [
                (c1, "BLINKS/MIN",  f"{features.blink_rate:.1f}"),
                (c2, "GAZE",        f"{features.gaze_aversion:.3f}"),
                (c3, "PUPIL VAR",   f"{features.pupil_variability:.3f}"),
                (c4, "HEAD JITTER", f"{features.head_jitter:.2f}"),
            ]:
                col.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='label'>{label}</div>"
                    f"<div class='metric-val'>{val}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<p style='color:#334155;font-size:12px;text-align:center;"
                "margin-top:20px'>No face detected</p>",
                unsafe_allow_html=True,
            )

    # ── Row 2: Timeline ───────────────────────────────────────
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown("<div class='label'>TENSION TIMELINE</div>", unsafe_allow_html=True)

    timestamps, scores = engine.get_history()
    st.plotly_chart(
        _tension_graph(timestamps, scores),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    # ── Row 3: Pattern | Stats ────────────────────────────────
    p_col, s_col = st.columns([1.5, 1], gap="large")

    with p_col:
        st.markdown("<div class='label'>PATTERN ANALYSIS</div>", unsafe_allow_html=True)
        pattern = engine.detect_pattern()
        msg = pattern or "No significant patterns detected"
        st.markdown(f"<div class='pattern-box'>{msg}</div>", unsafe_allow_html=True)

        # Band legend
        for color, rng, txt in [
            ("#4ade80", "0–30",   "Low Arousal"),
            ("#facc15", "30–60",  "Moderate Arousal"),
            ("#f87171", "60–100", "High Arousal"),
        ]:
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;margin:4px 0'>"
                f"<div style='width:9px;height:9px;border-radius:50%;"
                f"background:{color};flex-shrink:0'></div>"
                f"<span style='font-size:11px;color:#475569'>"
                f"{rng} · {txt}</span></div>",
                unsafe_allow_html=True,
            )

    with s_col:
        st.markdown("<div class='label'>SESSION STATS</div>", unsafe_allow_html=True)
        stats = engine.session_stats()
        st.markdown(
            f"<div class='stat-box'>"
            f"<div class='stat-row'>Peak &nbsp; <b style='color:#f87171'>"
            f"{stats['peak']}</b></div>"
            f"<div class='stat-row'>Mean &nbsp; <b style='color:#60a5fa'>"
            f"{stats['mean']}</b></div>"
            f"<div class='stat-row'>Samples &nbsp; <b style='color:#e2e8f0'>"
            f"{stats['samples']}</b></div>"
            f"<div class='stat-row' style='margin-top:8px;font-size:10px;color:#334155'>"
            f"Session ID<br>"
            f"<span style='color:#475569'>{db.session_id[:18]}…</span></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Disclaimer ────────────────────────────────────────────
    st.markdown(
        "<div class='disclaimer'>"
        "⚠️ This system visualizes arousal patterns only. "
        "It does not detect lies, intent, emotions, or mental states. "
        "Not a medical or psychological instrument."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Auto-rerun while streaming ────────────────────────────
    if ctx and ctx.state.playing:
        time.sleep(0.8)
        st.rerun()


if __name__ == "__main__":
    main()