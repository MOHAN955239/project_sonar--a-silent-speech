"""
Project Sonar - Silent Speech Interface  (Full Feature Dashboard)
=================================================================
All features included:
  - Multi-channel EMG live signal (stacked)
  - Spectrogram
  - Signal RMS power per channel (bar chart)
  - Prediction engine + SVG confidence gauge
  - Confidence score metric cards
  - XAI analysis text + channel importance bar chart
  - Phoneme probability heatmap
  - AAC panel with quick-phrase shortcuts + TTS
  - Confidence history line graph
  - Word frequency pie/bar chart
  - Session stats (streak, accuracy %, totals)
  - CSV export
"""

import os, sys, time, io, json
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from model_utils import load_model, predict, get_channel_importance, PRETRAINED_TRANSDUCTION

# Matches subsampled EMG rate used in read_emg.py (pipeline); STFT axis labels assume this Fs.
EMG_SAMPLE_RATE_HZ = 689.06

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Project Sonar – Silent Speech",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
html, body, [class*="css"] { font-family:'Inter',sans-serif; }
.block-container { padding-top:.8rem; padding-bottom:.5rem; }

.metric-card {
    background:linear-gradient(145deg,#111827,#1a2236);
    border:1px solid #1e3a5f; border-radius:14px;
    padding:14px 18px; margin:5px 0;
    box-shadow:0 4px 18px rgba(0,0,0,.4);
    transition:transform .18s;
}
.metric-card:hover { transform:translateY(-2px); }
.metric-title { font-size:.69rem; color:#6b7db3; letter-spacing:1.1px; text-transform:uppercase; }
.metric-value { font-size:1.9rem; font-weight:800; color:#64ffda; margin:3px 0 2px; }
.metric-sub   { font-size:.76rem; color:#8892b0; }

.section-label {
    font-size:.72rem; color:#6b7db3; letter-spacing:1.2px;
    text-transform:uppercase; margin:14px 0 6px;
    border-left:3px solid #2962a8; padding-left:8px;
}

.aac-box {
    background:linear-gradient(145deg,#0b1520,#111f2e);
    border:2px solid #238636; border-radius:14px;
    padding:16px 18px; box-shadow:0 4px 22px rgba(35,134,54,.12);
}
.xai-box {
    background:linear-gradient(145deg,#0a1520,#0f1e30);
    border:1px solid #2962a8; border-radius:12px;
    padding:13px 16px; margin:8px 0;
    box-shadow:0 4px 14px rgba(41,98,168,.12);
}
.badge-real { background:#0a3d20; color:#64ffda; border:1px solid #238636;
              border-radius:8px; padding:2px 9px; font-size:.72rem; font-weight:600; }
.badge-mock { background:#3d2700; color:#ffd166; border:1px solid #c07800;
              border-radius:8px; padding:2px 9px; font-size:.72rem; font-weight:600; }
.hist-pill  { display:inline-block; background:#0f2640; color:#64ffda;
              border-radius:18px; padding:3px 11px; margin:3px;
              font-size:.78rem; border:1px solid #1e4a6a; }
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.45;transform:scale(1.5)} }
.dot-live { display:inline-block; width:8px; height:8px; border-radius:50%;
            background:#64ffda; animation:pulse 1.4s ease-in-out infinite; margin-right:6px; }
.dot-idle { display:inline-block; width:8px; height:8px; border-radius:50%;
            background:#ffd166; margin-right:6px; }
.demo-banner {
    background:linear-gradient(90deg,#1a1f2e,#0f1729);
    border:1px solid #3d4f6e; border-radius:10px;
    padding:10px 14px; margin:8px 0 0; font-size:.82rem; color:#a8b9d4;
}
button[kind="primary"], .stButton button { min-height:2.4rem; }

/* Hero title: avoid mid-title line breaks / clipping from tight flex rows */
.app-hero {
    padding:4px 0 2px;
    min-width:0;
}
.app-hero-title-row {
    display:flex;
    align-items:flex-start;
    flex-wrap:wrap;
    gap:8px;
}
.app-hero-brand-lock {
    display:inline-flex;
    align-items:center;
    gap:10px;
    flex-wrap:nowrap;
    min-width:0;
    max-width:100%;
}
.app-hero-title {
    margin:0;
    padding:0;
    font-size:clamp(1.35rem, 2.8vw + 0.6rem, 1.95rem);
    font-weight:800;
    color:#64ffda;
    line-height:1.15;
    letter-spacing:-0.02em;
    white-space:nowrap;
}
.app-hero-tagline {
    display:block;
    margin:6px 0 0 2px;
    color:#8892b0;
    font-size:clamp(.78rem, 1.2vw + .65rem, .95rem);
    font-weight:500;
    line-height:1.35;
    max-width:42rem;
}
.app-hero-meta {
    color:#6b7db3;
    font-size:.79rem;
    margin:8px 0 0 2px;
    line-height:1.45;
    max-width:48rem;
}
.hackathon-strip {
    height:3px; border-radius:2px;
    background:linear-gradient(90deg,#f59e0b,#64ffda,#2962a8);
    margin:0 0 10px 0; opacity:.92;
}
.judge-card {
    background:linear-gradient(145deg,#0c1628,#111827);
    border:1px solid #374151; border-radius:12px;
    padding:12px 14px; margin:0 0 10px 0;
    font-size:.88rem; color:#c5cee0; line-height:1.5;
}
.sidebar-brand-title {
    color:#64ffda !important;
    font-size:clamp(1rem, 2.5vw, 1.25rem) !important;
    font-weight:800 !important;
    line-height:1.2 !important;
    margin:0 !important;
    word-break:normal;
    overflow-wrap:normal;
    hyphens:none;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
_defaults = dict(
    prediction_made=False, prediction="—", confidence=0.0,
    history=[], aac_text="", emg_buffer=None,
    last_channel_imp={}, streak=0, total_predictions=0, high_conf_count=0,
    emg_seed=0,
)
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v if not isinstance(v, list) else []

# After Predict/Refresh: consume pending EMG refresh before any plots (stable snapshot + working Refresh)
if st.session_state.pop("_pending_emg_refresh", False):
    st.session_state.emg_seed = int(st.session_state.get("emg_seed", 0)) + 1

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
def _checkpoint_mtime_for_cache():
    """Bust Streamlit cache when weights appear or change on disk."""
    try:
        return os.path.getmtime(PRETRAINED_TRANSDUCTION)
    except OSError:
        return 0.0


@st.cache_resource(show_spinner="Loading Silent Speech model…")
def get_model(_checkpoint_mtime: float):
    return load_model()

bundle = get_model(_checkpoint_mtime_for_cache())

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
QUICK_PHRASES = {
    "Emergency": ["HELP", "STOP", "NO"],
    "Basic":     ["HELLO", "YES", "PLEASE", "THANK YOU"],
    "Needs":     ["WATER", "HELP"],
}
PHONEMES = ["sil","p","b","t","d","k","g","m","n","s","z","h","l","r","w","y","ae","iy","ah","ow"]

def generate_emg(t, channels=8, noise=1.0, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    base_freqs = [8,12,18,25,35,50,80,120]
    out = []
    for i in range(channels):
        f = base_freqs[i % len(base_freqs)]
        s = np.sin(2*np.pi*f*t/1000) + 0.3*np.sin(2*np.pi*f*2.3*t/1000+0.7)
        s += noise * rng.standard_normal(len(t)).astype(np.float64) * 0.35
        s *= float(rng.uniform(0.5, 1.2))
        out.append(s.astype(np.float32))
    return np.stack(out, axis=1)

def gauge_svg(conf, col):
    import math
    r, cx, cy = 48, 60, 60
    arc  = 2*math.pi*r*0.75
    dash = conf/100*arc; gap = arc-dash
    rot  = -225
    return f"""<div style="text-align:center;margin:0 auto;width:120px">
<svg viewBox="0 0 120 90" width="120" height="90">
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#1e3a5f" stroke-width="9"
    stroke-dasharray="{arc:.1f} {2*math.pi*r-arc:.1f}"
    stroke-dashoffset="{arc/4:.1f}" stroke-linecap="round"
    transform="rotate({rot} {cx} {cy})"/>
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{col}" stroke-width="9"
    stroke-dasharray="{dash:.1f} {gap+2*math.pi*r-arc:.1f}"
    stroke-dashoffset="{arc/4:.1f}" stroke-linecap="round"
    transform="rotate({rot} {cx} {cy})"/>
  <text x="{cx}" y="{cy+6}" text-anchor="middle" fill="{col}"
    font-size="17" font-weight="800" font-family="Inter,sans-serif">{conf:.0f}%</text>
  <text x="{cx}" y="{cy+20}" text-anchor="middle" fill="#8892b0"
    font-size="7.5" font-family="Inter,sans-serif">CONFIDENCE</text>
</svg></div>"""

def xai_text(word, conf, imp):
    top = sorted(imp, key=imp.get, reverse=True)[:3]
    ch  = ", ".join(f"{c}({imp[c]:.0f}%)" for c in top)
    if conf >= 90:
        return "🟢 High confidence", "#64ffda", \
               f"Strong distinct bursts in {ch} match the '{word}' archetype closely."
    elif conf >= 70:
        return "🟡 Moderate confidence", "#ffd166", \
               f"Core patterns for '{word}' detected in {ch}; minor cross-channel noise reduced certainty."
    else:
        return "🔴 Low confidence", "#e63946", \
               f"Noisy signal — {ch} had highest correlation but amplitude was sub-threshold for '{word}'."

def fig_settings(fig, bg="#080e1a"):
    fig.patch.set_facecolor(bg)

def ax_dark(ax, bg="#080e1a"):
    ax.set_facecolor(bg)
    for sp in ax.spines.values(): sp.set_color("#1e3a5f")
    ax.tick_params(colors="#8892b0")

def export_csv():
    if not st.session_state.history: return None
    buf = io.StringIO()
    pd.DataFrame(st.session_state.history).to_csv(buf, index=False)
    return buf.getvalue().encode()

def export_session_json(bundle):
    if not st.session_state.history:
        return None
    payload = {
        "project": "Project Sonar",
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_mode": "real" if bundle.get("real") else "demo",
        "device": bundle.get("device", "cpu"),
        "predictions": st.session_state.history,
    }
    return json.dumps(payload, indent=2).encode("utf-8")

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:8px 4px 4px;max-width:100%">
      <div style="font-size:2rem;line-height:1;margin-bottom:6px">🌊</div>
      <h2 class="sidebar-brand-title">Project&nbsp;Sonar</h2>
      <p style="color:#6b7db3;font-size:.73rem;margin:6px 0 0;line-height:1.35">Silent Speech Interface</p>
    </div>""", unsafe_allow_html=True)
    st.divider()

    badge = ('<span class="badge-real">🟢 Real PyTorch</span>' if bundle.get("real")
             else '<span class="badge-mock">🟡 Demo Mode</span>')
    dev = bundle.get("device", "cpu")
    st.markdown(f"**Model:** {badge}", unsafe_allow_html=True)
    if bundle.get("real"):
        st.caption(f"Running on **{dev}** · synthetic EMG is still demo data unless you stream real EMG.")
    else:
        reason = bundle.get("demo_reason")
        detail = (bundle.get("demo_detail") or "").strip()
        with st.expander("⬆️ Enable real PyTorch mode", expanded=True):
            if reason == "no_torch":
                st.markdown("Install PyTorch (CPU is fine), **restart Streamlit**, then reload this page.")
                st.code("pip install torch", language="bash")
            elif reason == "missing_weights":
                st.markdown(
                    "Unzip **`pretrained_models.zip`** into the project folder so this exact file exists "
                    "(create the folder if needed):"
                )
                st.code(detail, language="text")
                st.markdown(
                    "If the archive contains a nested folder, move files so the `.pt` sits at the path above. "
                    "Optional: add **`normalizers.pkl`** next to `app.py` for training-matched EMG scaling."
                )
            elif reason == "load_error":
                st.error("Could not load the checkpoint. Fix the error below or use Demo mode.")
                st.code(detail, language="text")
            else:
                st.caption("Place pretrained weights and install PyTorch — see README / Zenodo for the official release.")

    st.divider()
    with st.expander("⚙️ Signal", expanded=True):
        n_ch   = st.slider("EMG channels", 1, 8, 8)
        win    = st.slider("Window (samples)", 50, 400, 180)
        noise  = st.slider("Noise", 0.1, 3.0, 0.9, 0.1)
        st.caption("Synthetic demo EMG — stable until you change sliders or **Refresh**.")
        if st.button("🔄 New synthetic sample", use_container_width=True, key="sidebar_emg_refresh"):
            st.session_state._pending_emg_refresh = True
            st.rerun()

    with st.expander("🧠 Prediction", expanded=True):
        thresh    = st.slider("Confidence threshold %", 50, 95, 72, 5)
        autospeak = st.checkbox("Auto-speak on predict", False)
        auto_add  = st.checkbox("Auto-add word to AAC", False)

    st.divider()
    st.subheader("📊 Session Stats")
    total   = st.session_state.total_predictions
    hi      = st.session_state.high_conf_count
    acc     = hi/total*100 if total else 0
    c1, c2  = st.columns(2)
    c1.metric("Predictions", total)
    c2.metric("High-conf", f"{acc:.0f}%")
    st.markdown(f'<div style="color:#ffd166;font-size:.82rem">🔥 Streak: <b style="font-size:1.3rem">{st.session_state.streak}</b></div>', unsafe_allow_html=True)

    st.divider()
    csv = export_csv()
    jsn = export_session_json(bundle)
    if csv:
        st.download_button("⬇️ Export CSV", csv, "sonar_history.csv", "text/csv", use_container_width=True)
    if jsn:
        st.download_button("⬇️ Export JSON (judges)", jsn, "sonar_session.json", "application/json", use_container_width=True)

    with st.expander("🏁 Hackathon — 60s demo script", expanded=False):
        st.markdown(
            "1. **Signal wall** — EMG stack + spectrogram + RMS.\n"
            "2. **Predict** — confidence + word (enable real weights for PyTorch).\n"
            "3. **AAC** — quick phrase → **Speak** (browser TTS).\n"
            "4. **Export** CSV/JSON for judges · see **HACKATHON.md** for Q&A."
        )
    if st.button("🗑 Reset Session", use_container_width=True):
        for k, v in _defaults.items():
            st.session_state[k] = v if not isinstance(v, list) else []
        st.session_state._pending_emg_refresh = False
        st.rerun()

    with st.expander("❓ How to use", expanded=False):
        st.markdown(
            "1. Tune **Signal** sliders or load real weights for live EMG.\n"
            "2. Click **Predict** to classify the current window.\n"
            "3. Use **AAC** quick phrases or **Speak** for browser TTS.\n"
            "4. **Export CSV** or **JSON** saves this session’s log (JSON adds model mode for judges).\n"
            "5. See **HACKATHON.md** for a full demo script and pitch bullets."
        )

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
hc1, hc2 = st.columns([2.6, 1], gap="small")
with hc1:
    dot = '<span class="dot-live"></span>' if st.session_state.prediction_made else '<span class="dot-idle"></span>'
    st.markdown(f"""
    <div class="app-hero">
      <div class="app-hero-title-row">
        <div class="app-hero-brand-lock">
          {dot}
          <h1 class="app-hero-title">Project&nbsp;Sonar</h1>
        </div>
      </div>
      <span class="app-hero-tagline">Silent EMG → Speech AI</span>
      <p class="app-hero-meta">
        Research-grade silent speech recognition · Explainable AI · AAC interface
      </p>
    </div>""", unsafe_allow_html=True)
with hc2:
    st.markdown(f"""
    <div style="text-align:right;padding-top:10px">
      <span style="color:#6b7db3;font-size:.72rem">Session clock</span><br>
      <span style="color:#ccd6f6;font-size:1.05rem;font-weight:700">{time.strftime('%H:%M:%S')}</span>
    </div>""", unsafe_allow_html=True)
st.divider()

st.markdown('<div class="hackathon-strip" aria-hidden="true"></div>', unsafe_allow_html=True)
with st.expander("🏆 For judges — pitch, impact & what to click", expanded=True):
    st.markdown("""
<div class="judge-card">

**One-liner:** *Project Sonar* is an interactive **silent-speech command center** — EMG-style signals, neural prediction,
explainability-style context, and an **AAC + text-to-speech** path for urgent communication scenarios.

**Why it matters:** Surfaces a hard problem (silent speech is invisible) into visuals judges can follow in under a minute,
with a credible link to published **Voicing Silent Speech** research.

**Click path (live):** Signal panels → **Predict** → scroll to **AAC** → **Speak** → **Export CSV/JSON** (sidebar).

**Differentiators:** Full dashboard (not a single chart), session analytics, export for scoring/review, optional **real PyTorch** weights.

</div>
""", unsafe_allow_html=True)

if not bundle.get("real"):
    st.markdown(
        '<div class="demo-banner">🟡 <b>Demo mode</b> — charts use a deterministic synthetic EMG sample '
        "(change sliders or <b>Refresh</b> for a new sample). Add <code>pretrained_models/</code> for real inference.</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic EMG: deterministic for (channels, window, noise, emg_seed) — no flicker on every click
# ─────────────────────────────────────────────────────────────────────────────
def _emg_rng(n_ch, win, noise, seed):
    mix = (int(seed) * 1_000_003) ^ (int(win) * 2_654_443) ^ (int(n_ch) * 97) ^ int(float(noise) * 1_000)
    return np.random.default_rng(mix % (2**32))

t = np.arange(win, dtype=np.float64)
rng = _emg_rng(n_ch, win, noise, int(st.session_state.get("emg_seed", 0)))
emg = generate_emg(t, n_ch, noise, rng=rng)
st.session_state.emg_buffer = emg

# ═══════════════════════════════════════════════════════════════════
# ROW 1 — EMG Signal + Signal Power
# ═══════════════════════════════════════════════════════════════════
r1c1, r1c2 = st.columns([2.4, 1], gap="medium")

with r1c1:
    st.markdown('<div class="section-label">📡 Real-time Multi-channel EMG Signal</div>', unsafe_allow_html=True)
    fig = plt.figure(figsize=(10, max(n_ch*0.82+0.8, 3.2)), facecolor="#080e1a")
    gs  = gridspec.GridSpec(n_ch, 1, figure=fig, hspace=0.0)
    pal = plt.cm.plasma(np.linspace(0.15, 0.92, n_ch))
    for i in range(n_ch):
        ax = fig.add_subplot(gs[i])
        ax.plot(t, emg[:,i], color=pal[i], linewidth=0.88, alpha=0.95)
        ax.fill_between(t, emg[:,i], alpha=0.07, color=pal[i])
        ax.set_facecolor("#080e1a")
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_yticks([])
        ax.set_ylabel(f"CH{i+1}", color=pal[i], fontsize=6.5, rotation=0, labelpad=24, va="center", fontweight="600")
        ax.set_ylim(emg[:,i].min()-0.4, emg[:,i].max()+0.4)
        if i < n_ch-1: ax.set_xticks([])
        else: ax.tick_params(colors="#8892b0", labelsize=6.5)
    fig.text(0.5, 0.0, "Time (samples)", ha="center", color="#8892b0", fontsize=7)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with r1c2:
    st.markdown('<div class="section-label">⚡ Signal RMS Power</div>', unsafe_allow_html=True)
    rms  = np.sqrt(np.mean(emg**2, axis=0))
    chs  = [f"CH{i+1}" for i in range(n_ch)]
    rmax = float(rms.max()) if rms.size else 1.0
    cols_rms = plt.cm.plasma(rms / max(rmax, 1e-9))

    fig_rms, ax_rms = plt.subplots(figsize=(4, max(n_ch*0.82+0.8, 3.2)), facecolor="#080e1a")
    ax_dark(ax_rms)
    bars = ax_rms.barh(chs, rms, color=cols_rms, edgecolor="#080e1a", height=0.6)
    ax_rms.set_xlabel("RMS Amplitude", color="#8892b0", fontsize=7.5)
    ax_rms.tick_params(colors="#8892b0", labelsize=8)
    ax_rms.set_title("Per-Channel RMS Power", color="#ccd6f6", fontsize=8.5)
    ax_rms.invert_yaxis()
    for bar, val in zip(bars, rms):
        ax_rms.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
                    f"{val:.2f}", va="center", ha="left", color="#8892b0", fontsize=6.5)
    fig_rms.tight_layout()
    st.pyplot(fig_rms, use_container_width=True)
    plt.close(fig_rms)

    # Signal quality indicator
    avg_rms = float(rms.mean())
    q_col   = "#64ffda" if avg_rms > 0.5 else ("#ffd166" if avg_rms > 0.25 else "#e63946")
    q_label = "Excellent" if avg_rms > 0.5 else ("Moderate" if avg_rms > 0.25 else "Poor")
    st.markdown(f"""
    <div class="metric-card" style="margin-top:8px">
      <div class="metric-title">Signal Quality</div>
      <div class="metric-value" style="font-size:1.4rem;color:{q_col}">{q_label}</div>
      <div class="metric-sub">Avg RMS: {avg_rms:.3f}</div>
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# ROW 2 — Spectrogram + Controls + Confidence Metrics
# ═══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">📊 Frequency Analysis & Prediction Engine</div>', unsafe_allow_html=True)
r2c1, r2c2 = st.columns(2, gap="medium")

with r2c1:
    fig_sp, ax_sp = plt.subplots(figsize=(7, 2.2), facecolor="#080e1a")
    ax_dark(ax_sp)
    ax_sp.specgram(emg[:,0], Fs=EMG_SAMPLE_RATE_HZ, cmap="plasma", NFFT=32, noverlap=16)
    ax_sp.set_ylabel("Freq (Hz)", color="#8892b0", fontsize=7.5)
    ax_sp.set_xlabel("Time (s)", color="#8892b0", fontsize=7.5)
    ax_sp.tick_params(colors="#8892b0", labelsize=7)
    ax_sp.set_title("EMG Spectrogram – Channel 1", color="#ccd6f6", fontsize=9)
    st.caption(f"STFT uses Fs = {EMG_SAMPLE_RATE_HZ:.2f} Hz (same scale as the project’s EMG pipeline). Synthetic demo data may not match real articulatory EMG.")
    st.pyplot(fig_sp, use_container_width=True)
    plt.close(fig_sp)

with r2c2:
    bc1, bc2 = st.columns(2)
    with bc1:
        do_predict = st.button("🧠 Predict", use_container_width=True, type="primary")
    with bc2:
        do_refresh = st.button("🔄 Refresh", use_container_width=True, help="Draw a new synthetic EMG sample (demo)")

    if do_refresh:
        st.session_state._pending_emg_refresh = True
        st.toast("New synthetic sample…")
        st.rerun()

    if do_predict:
        with st.spinner("Analysing EMG…"):
            word, conf = predict(bundle, st.session_state.emg_buffer)
        st.session_state.update(dict(
            prediction_made=True, prediction=word, confidence=conf,
            last_channel_imp=get_channel_importance(st.session_state.emg_buffer),
        ))
        st.session_state.history.append({"word": word, "confidence": round(conf,1),
                                          "time": time.strftime("%H:%M:%S")})
        st.session_state.total_predictions += 1
        if conf >= thresh:
            st.session_state.streak += 1
            st.session_state.high_conf_count += 1
        else:
            st.session_state.streak = 0
        if autospeak or auto_add:
            st.session_state.aac_text = (st.session_state.aac_text + " " + word).strip()
        st.rerun()

    if st.session_state.prediction_made:
        conf = st.session_state.confidence
        word = st.session_state.prediction
        col  = "#64ffda" if conf >= 90 else ("#ffd166" if conf >= thresh else "#e63946")

        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-title">Prediction</div>
              <div class="metric-value" style="font-size:1.5rem">{word}</div>
              <div class="metric-sub">Top-1 class</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(gauge_svg(conf, col), unsafe_allow_html=True)

        if conf < thresh:
            st.warning(f"⚠️ Below threshold ({thresh}%). Please repeat the utterance.")
    else:
        st.info("Click **Predict** to start analysis.")

# ═══════════════════════════════════════════════════════════════════
# ROW 3 — XAI + Phoneme Heatmap + AAC Panel
# ═══════════════════════════════════════════════════════════════════
st.divider()
r3c1, r3c2 = st.columns([1.2, 1], gap="large")

with r3c1:
    st.markdown('<div class="section-label">🔍 Explainable AI (XAI) Analysis</div>', unsafe_allow_html=True)

    if st.session_state.prediction_made:
        word = st.session_state.prediction
        conf = st.session_state.confidence
        imp  = st.session_state.last_channel_imp

        lvl, lvl_col, body = xai_text(word, conf, imp)
        st.markdown(f"""
        <div class="xai-box">
          <div style="font-size:.92rem;font-weight:700;color:{lvl_col};margin-bottom:7px">{lvl}</div>
          <div style="color:#ccd6f6;font-size:.86rem;margin-bottom:5px">
            Prediction: <span style="color:#64ffda;font-weight:700">{word}</span>
          </div>
          <div style="color:#8892b0;font-size:.82rem;line-height:1.55">{body}</div>
        </div>""", unsafe_allow_html=True)

        # Channel importance bar
        if imp:
            vals = list(imp.values()); chs = list(imp.keys())
            mv   = max(vals)
            fig_imp, ax_imp = plt.subplots(figsize=(6, 1.9), facecolor="#080e1a")
            ax_dark(ax_imp)
            bcols = [lvl_col if v == mv else "#1e3a5f" for v in vals]
            ax_imp.bar(chs, vals, color=bcols, edgecolor="#080e1a", width=0.55)
            ax_imp.set_ylabel("Importance %", color="#8892b0", fontsize=7.5)
            ax_imp.set_title("Channel Feature Importance", color="#ccd6f6", fontsize=8.5)
            st.caption("Heuristic: mean |amplitude| per channel — not SHAP/Grad-CAM.")
            st.pyplot(fig_imp, use_container_width=True)
            plt.close(fig_imp)

        # Phoneme heatmap (illustrative in this UI — not the model's posterior unless wired from predict)
        st.markdown('<div class="section-label" style="margin-top:10px">🔬 Phoneme Probability Heatmap</div>', unsafe_allow_html=True)
        st.caption("Demo visualization: random simplex samples (not the network’s phoneme logits).")
        n_frames = 20
        rng_ph = np.random.default_rng((hash(word) % (2**30)) + int(st.session_state.get("emg_seed", 0)))
        hmap = rng_ph.dirichlet(np.ones(len(PHONEMES)), size=n_frames).T  # (phones, frames)
        dom = hash(word) % len(PHONEMES)
        hmap[dom] *= 2.5
        hmap = hmap / hmap.sum(axis=0, keepdims=True)

        fig_ph, ax_ph = plt.subplots(figsize=(6, 2.8), facecolor="#080e1a")
        ax_dark(ax_ph)
        im = ax_ph.imshow(hmap, aspect="auto", cmap="plasma", vmin=0, vmax=0.4, interpolation="bilinear")
        ax_ph.set_yticks(range(len(PHONEMES)))
        ax_ph.set_yticklabels(PHONEMES, fontsize=6)
        ax_ph.set_xlabel("Time frames →", color="#8892b0", fontsize=7.5)
        ax_ph.set_title("Predicted Phoneme Probabilities Over Time", color="#ccd6f6", fontsize=8.5)
        ax_ph.tick_params(colors="#8892b0", labelsize=7)
        plt.colorbar(im, ax=ax_ph, fraction=0.025, pad=0.02).ax.tick_params(colors="#8892b0", labelsize=6)
        fig_ph.tight_layout()
        st.pyplot(fig_ph, use_container_width=True)
        plt.close(fig_ph)

    else:
        st.markdown("""
        <div class="xai-box" style="text-align:center;padding:30px 16px">
          <div style="font-size:2.2rem">🧬</div>
          <div style="color:#6b7db3;font-size:.86rem;margin-top:10px">
            Run a prediction to see<br>XAI analysis &amp; phoneme heatmap
          </div>
        </div>""", unsafe_allow_html=True)

with r3c2:
    # ── AAC Panel ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">💬 AAC Communication Panel</div>', unsafe_allow_html=True)

    # Quick phrases
    for cat, phrases in QUICK_PHRASES.items():
        st.markdown(f'<span style="color:#6b7db3;font-size:.68rem;letter-spacing:1px">{cat.upper()}</span>', unsafe_allow_html=True)
        cols_q = st.columns(len(phrases))
        for ci, ph in enumerate(phrases):
            with cols_q[ci]:
                if st.button(ph, key=f"qp_{cat}_{ph}", use_container_width=True):
                    st.session_state.aac_text = (st.session_state.aac_text + " " + ph).strip()
                    st.rerun()

    st.markdown('<div class="aac-box">', unsafe_allow_html=True)
    if st.session_state.history and not st.session_state.aac_text:
        st.session_state.aac_text = " ".join(h["word"] for h in st.session_state.history)

    aac_text = st.text_area(
        "aac", value=st.session_state.aac_text,
        height=90, key="aac_display",
        label_visibility="collapsed",
        placeholder="Recognised words appear here…",
    )
    st.session_state.aac_text = aac_text

    pa1, pa2, pa3 = st.columns(3)
    with pa1:
        if st.button("🔊 Speak", use_container_width=True, type="primary"):
            if aac_text.strip():
                safe = aac_text.strip().replace("\\","\\\\").replace("'","\\'").replace('"','\\"').replace("\n"," ")
                st.components.v1.html(f"""<script>
var u=new SpeechSynthesisUtterance("{safe}");
u.rate=0.88;u.pitch=1.05;u.volume=1.0;
window.speechSynthesis.cancel();window.speechSynthesis.speak(u);
</script>""", height=0)
                st.toast("Speaking: " + aac_text.strip()[:40])
            else:
                st.warning("Nothing to speak.")
    with pa2:
        if st.button("➕ Add Word", use_container_width=True):
            if st.session_state.prediction_made:
                st.session_state.aac_text = (aac_text + " " + st.session_state.prediction).strip()
                st.rerun()
    with pa3:
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state.aac_text = ""
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # History pills
    if st.session_state.history:
        st.markdown("<br>", unsafe_allow_html=True)
        pills = "".join(
            f'<span class="hist-pill">{h["word"]} <small style="color:#6b7db3">{h["confidence"]}%</small> '
            f'<span style="color:#2962a8;margin-left:3px">{h["time"]}</span></span>'
            for h in reversed(st.session_state.history[-8:])
        )
        st.markdown(pills, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# ROW 4 — Confidence History + Word Frequency
# ═══════════════════════════════════════════════════════════════════
if st.session_state.history:
    st.divider()
    st.markdown('<div class="section-label">📈 Performance Analytics</div>', unsafe_allow_html=True)
    r4c1, r4c2 = st.columns([1.8, 1], gap="large")

    hist_df   = pd.DataFrame(st.session_state.history)
    hist_conf = hist_df["confidence"].tolist()
    hist_lbls = hist_df["word"].tolist()

    with r4c1:
        fig_h, ax_h = plt.subplots(figsize=(9, 2.4), facecolor="#080e1a")
        ax_dark(ax_h)
        x = range(len(hist_conf))
        pt_cols = ["#64ffda" if c>=90 else ("#ffd166" if c>=thresh else "#e63946") for c in hist_conf]
        ax_h.plot(x, hist_conf, color="#64ffda", linewidth=1.8, alpha=0.9, zorder=1)
        ax_h.scatter(x, hist_conf, c=pt_cols, s=44, zorder=2)
        ax_h.fill_between(x, hist_conf, alpha=0.1, color="#64ffda")
        ax_h.axhline(thresh, color="#e63946", linestyle="--", linewidth=1, alpha=0.6, label=f"Threshold {thresh}%")
        ax_h.axhline(90, color="#64ffda", linestyle=":", linewidth=0.8, alpha=0.4, label="High-conf (90%)")
        ax_h.set_xticks(range(len(hist_lbls)))
        ax_h.set_xticklabels(hist_lbls, color="#8892b0", fontsize=7.5, rotation=30)
        ax_h.set_ylim(40, 105)
        ax_h.set_ylabel("Confidence %", color="#8892b0", fontsize=7.5)
        ax_h.set_title("Prediction Confidence over Time", color="#ccd6f6", fontsize=9)
        ax_h.legend(fontsize=7, facecolor="#080e1a", edgecolor="#1e3a5f", labelcolor="#8892b0")
        st.pyplot(fig_h, use_container_width=True)
        plt.close(fig_h)

    with r4c2:
        word_counts = Counter(hist_lbls)
        wc_labels   = list(word_counts.keys())
        wc_vals     = list(word_counts.values())
        bar_pal     = plt.cm.plasma(np.linspace(0.2, 0.85, len(wc_labels)))

        fig_wf, ax_wf = plt.subplots(figsize=(5, 2.4), facecolor="#080e1a")
        ax_dark(ax_wf)
        ax_wf.barh(wc_labels, wc_vals, color=bar_pal, edgecolor="#080e1a", height=0.55)
        ax_wf.set_xlabel("Count", color="#8892b0", fontsize=7.5)
        ax_wf.set_title("Word Frequency Distribution", color="#ccd6f6", fontsize=9)
        ax_wf.tick_params(colors="#8892b0", labelsize=7.5)
        ax_wf.invert_yaxis()
        for i, v in enumerate(wc_vals):
            ax_wf.text(v+0.05, i, str(v), va="center", color="#8892b0", fontsize=7)
        fig_wf.tight_layout()
        st.pyplot(fig_wf, use_container_width=True)
        plt.close(fig_wf)

    # Accuracy summary metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f'<div class="metric-card"><div class="metric-title">Total Predictions</div><div class="metric-value">{total}</div><div class="metric-sub">this session</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-card"><div class="metric-title">High-confidence</div><div class="metric-value" style="color:#64ffda">{hi}</div><div class="metric-sub">above threshold</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="metric-card"><div class="metric-title">Accuracy Rate</div><div class="metric-value" style="color:#ffd166">{acc:.0f}%</div><div class="metric-sub">high-conf / total</div></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="metric-card"><div class="metric-title">Current Streak</div><div class="metric-value">🔥 {st.session_state.streak}</div><div class="metric-sub">consecutive high-conf</div></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center;color:#3d4f6e;font-size:.73rem;padding:6px 8px 8px;line-height:1.45">
  <strong style="color:#5a6d8f">Project Sonar</strong> — accessibility-minded silent-speech interface demo<br>
  <span style="font-size:.7rem">Streamlit · PyTorch · EMNLP 2020 silent-speech lineage · Session export for judging</span>
</div>""", unsafe_allow_html=True)
