"""
src/ui/app.py — Arousal Monitor (camera_input version, no av/webrtc)
"""

import sys, os, time
from datetime import datetime
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import cv2
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.cv.extractor import FacialExtractor
from src.features.tension import TensionEngine
from src.db.client import SessionLogger

st.set_page_config(page_title="Arousal Monitor", page_icon="🧠",
                   layout="wide", initial_sidebar_state="collapsed")

if "extractor" not in st.session_state:
    st.session_state.extractor = FacialExtractor(fps=10)
if "engine"    not in st.session_state:
    st.session_state.engine    = TensionEngine()
if "db"        not in st.session_state:
    st.session_state.db        = SessionLogger()
if "last_log"  not in st.session_state:
    st.session_state.last_log  = 0.0

extractor = st.session_state.extractor
engine    = st.session_state.engine
db        = st.session_state.db

def tension_graph(timestamps, scores):
    fig = go.Figure()
    for y0,y1,col in [(0,30,"#4ade80"),(30,60,"#facc15"),(60,100,"#f87171")]:
        fig.add_hrect(y0=y0,y1=y1,fillcolor=col,opacity=0.05,line_width=0)
    if timestamps:
        t0 = timestamps[0]
        x  = [t-t0 for t in timestamps]
        fig.add_trace(go.Scatter(x=x, y=list(scores),
            mode="lines", line=dict(color="#60a5fa",width=2.5,shape="spline"),
            fill="tozeroy", fillcolor="rgba(96,165,250,0.08)"))
        fig.add_trace(go.Scatter(x=[x[-1]],y=[scores[-1]],
            mode="markers",
            marker=dict(size=9,color="#60a5fa",line=dict(width=2,color="#000")),
            showlegend=False))
    else:
        fig.add_annotation(text="Waiting for calibration…",
            xref="paper",yref="paper",x=0.5,y=0.5,showarrow=False,
            font=dict(color="#475569",size=13))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8",size=11),
        xaxis=dict(title="Elapsed (s)",gridcolor="rgba(255,255,255,0.06)",showline=False,zeroline=False),
        yaxis=dict(title="Tension Index",range=[0,100],gridcolor="rgba(255,255,255,0.06)",
                   showline=False,zeroline=False,tickvals=[0,30,60,100]),
        margin=dict(l=10,r=10,t=8,b=40), height=240, showlegend=False)
    return fig

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&display=swap');
html,body,[class*="css"]{font-family:'JetBrains Mono',monospace!important}
.stApp{background:#080d1a!important;color:#e2e8f0}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding:1.2rem 2rem 2rem!important;max-width:1200px}
.label{font-size:10px;letter-spacing:2.5px;text-transform:uppercase;color:#334155;margin-bottom:6px}
.score-big{font-size:76px;font-weight:600;line-height:1;text-align:center}
.band-text{font-size:11px;letter-spacing:3px;text-align:center;margin-top:4px}
.metric-card{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);border-radius:10px;padding:12px 14px;text-align:center}
.metric-val{font-size:20px;font-weight:600;color:#7dd3fc}
.pattern-box{background:rgba(96,165,250,.07);border:1px solid rgba(96,165,250,.18);border-radius:8px;padding:10px 14px;font-size:12px;color:#93c5fd;margin-bottom:8px}
.stat-box{background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.06);border-radius:8px;padding:14px}
.stat-row{font-size:12px;color:#94a3b8;margin:4px 0}
.disclaimer{background:rgba(250,204,21,.04);border:1px solid rgba(250,204,21,.12);border-radius:8px;padding:10px 16px;font-size:11px;color:#64748b;text-align:center;margin-top:1rem}
.hr{border:none;border-top:1px solid rgba(255,255,255,.06);margin:.8rem 0}
</style>
""", unsafe_allow_html=True)

# Header
c1, c2 = st.columns([4,1])
with c1:
    db_dot = "🟢" if db.is_connected() else "⚫"
    st.markdown(f"<h2 style='margin:0;color:#e2e8f0'>🧠 AROUSAL MONITOR {db_dot}</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#334155;font-size:11px;letter-spacing:2px;margin:-6px 0 0'>REAL-TIME PHYSIOLOGICAL SIGNAL VISUALIZATION</p>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<p style='color:#334155;font-size:11px;text-align:right;margin-top:16px'>{datetime.now().strftime('%H:%M · %d %b')}</p>", unsafe_allow_html=True)

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

cam_col, score_col = st.columns([1.2, 1], gap="large")

with cam_col:
    st.markdown("<div class='label'>LIVE FEED</div>", unsafe_allow_html=True)
    img_file = st.camera_input("", label_visibility="collapsed")
    annotated_placeholder = st.empty()
    prog = engine._calibration_progress()
    if not engine.calibrated:
        if prog > 0:
            st.progress(prog, text=f"Calibrating baseline — {int(prog*100)}%")
        else:
            st.info("Allow camera access then keep clicking the camera button to stream frames", icon="ℹ️")
    else:
        st.success("✓ Baseline calibrated")

with score_col:
    score_placeholder  = st.empty()
    metric_placeholder = st.empty()

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
st.markdown("<div class='label'>TENSION TIMELINE</div>", unsafe_allow_html=True)
graph_placeholder = st.empty()

p_col, s_col = st.columns([1.5,1], gap="large")
with p_col:
    st.markdown("<div class='label'>PATTERN ANALYSIS</div>", unsafe_allow_html=True)
    pattern_placeholder = st.empty()
with s_col:
    st.markdown("<div class='label'>SESSION STATS</div>", unsafe_allow_html=True)
    stats_placeholder = st.empty()

st.markdown("<div class='disclaimer'>⚠️ This system visualizes arousal patterns only. It does not detect lies, intent, emotions, or mental states. Not a medical or psychological instrument.</div>", unsafe_allow_html=True)

# Process frame
features = None
if img_file is not None:
    pil_img = Image.open(img_file)
    frame   = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    features, annotated = extractor.process_frame(frame)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    annotated_placeholder.image(annotated_rgb, use_column_width=True)

state = engine.update(features)

now = time.time()
if state.calibrated and (now - st.session_state.last_log) >= 5:
    db.log(features, state.score)
    st.session_state.last_log = now

score_color = state.band_color if state.calibrated else "#1e3a5f"
label_text  = state.band       if state.calibrated else "CALIBRATING"

score_placeholder.markdown(
    f"<div class='label' style='text-align:center;margin-top:12px'>TENSION INDEX</div>"
    f"<div class='score-big' style='color:{score_color}'>{state.score:.0f}</div>"
    f"<div class='band-text' style='color:{score_color}'>{label_text}</div>",
    unsafe_allow_html=True)

if features:
    metric_placeholder.markdown(
        f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px'>"
        f"<div class='metric-card'><div class='label'>BLINKS/MIN</div><div class='metric-val'>{features.blink_rate:.1f}</div></div>"
        f"<div class='metric-card'><div class='label'>GAZE</div><div class='metric-val'>{features.gaze_aversion:.3f}</div></div>"
        f"<div class='metric-card'><div class='label'>PUPIL VAR</div><div class='metric-val'>{features.pupil_variability:.3f}</div></div>"
        f"<div class='metric-card'><div class='label'>HEAD JITTER</div><div class='metric-val'>{features.head_jitter:.2f}</div></div>"
        f"</div>", unsafe_allow_html=True)
else:
    metric_placeholder.markdown("<p style='color:#334155;font-size:12px;text-align:center;margin-top:20px'>No face detected</p>", unsafe_allow_html=True)

timestamps, scores = engine.get_history()
graph_placeholder.plotly_chart(tension_graph(timestamps, scores),
    use_container_width=True, config={"displayModeBar": False})

pattern = engine.detect_pattern() or "No significant patterns detected"
pattern_placeholder.markdown(
    f"<div class='pattern-box'>{pattern}</div>"
    f"<div style='display:flex;flex-direction:column;gap:4px;margin-top:6px'>"
    f"<div style='display:flex;align-items:center;gap:8px'><div style='width:9px;height:9px;border-radius:50%;background:#4ade80'></div><span style='font-size:11px;color:#475569'>0–30 · Low Arousal</span></div>"
    f"<div style='display:flex;align-items:center;gap:8px'><div style='width:9px;height:9px;border-radius:50%;background:#facc15'></div><span style='font-size:11px;color:#475569'>30–60 · Moderate Arousal</span></div>"
    f"<div style='display:flex;align-items:center;gap:8px'><div style='width:9px;height:9px;border-radius:50%;background:#f87171'></div><span style='font-size:11px;color:#475569'>60–100 · High Arousal</span></div>"
    f"</div>", unsafe_allow_html=True)

stats = engine.session_stats()
stats_placeholder.markdown(
    f"<div class='stat-box'>"
    f"<div class='stat-row'>Peak &nbsp;<b style='color:#f87171'>{stats['peak']}</b></div>"
    f"<div class='stat-row'>Mean &nbsp;<b style='color:#60a5fa'>{stats['mean']}</b></div>"
    f"<div class='stat-row'>Samples &nbsp;<b style='color:#e2e8f0'>{stats['samples']}</b></div>"
    f"<div class='stat-row' style='margin-top:8px;font-size:10px;color:#334155'>Session<br><span style='color:#475569'>{db.session_id[:20]}…</span></div>"
    f"</div>", unsafe_allow_html=True)

# Auto-rerun to keep streaming
time.sleep(0.5)
st.rerun()