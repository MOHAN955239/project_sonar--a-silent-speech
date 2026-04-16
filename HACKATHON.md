# Project Sonar — Hackathon pack

Use this file for **slides, judging Q&A, and rehearsal** for **Project Sonar** (`app.py`).

## One-line pitch

**Project Sonar turns silent-speech EMG signals into an interactive dashboard** — live-style visualization, neural prediction, explainability hints, and an **AAC-style communication panel** with browser text-to-speech — supporting accessibility and clinical-adjacent storytelling.

## Problem → solution

| Problem | What we show |
|--------|----------------|
| Silent speech is invisible to most people | Multi-channel EMG-style traces, spectrogram, RMS |
| Black-box models erode trust | XAI-style channel emphasis + narrative (heuristic) |
| Users may need to communicate urgently | Quick phrases + message buffer + **Speak** (TTS) |
| Judges need reproducibility | Session history, **CSV export** (and optional JSON) |

## 60-second demo script

1. **Open** the app (`python -m streamlit run app.py`). Point at the **EMG stack** and **spectrogram** — “this is the signal path we monitor.”
2. Click **Predict** — show **confidence gauge** and **prediction card** (demo mode = synthetic signal + mock or real weights if installed).
3. Scroll to **AAC panel** — tap a quick phrase, hit **Speak** so the browser reads it aloud.
4. Mention **Explainable AI** block and **session analytics** / export — “we log the session for review.”
5. Close with **impact**: accessibility, silent speech interfaces, future hardware EMG hookup.

---

## 2-minute demo video (recording script)

Use this for a **≤2:00** submission or pitch recording. Read at a calm pace (~230–260 spoken words total). **Rehearse once with a timer.**

### Before you hit record

- [ ] App running: `python -m streamlit run app.py` → **http://localhost:8501**
- [ ] Browser **full screen** (F11); zoom **100–110%** so text is readable on video
- [ ] **Dark room** + good **face lighting** if you appear on camera; otherwise **voiceover + screen only** is fine
- [ ] Close unrelated tabs; **Do Not Disturb** on
- [ ] Optional: collapse the **“For judges”** expander after first take if it steals vertical space

### Shot list (what’s on screen)

| Time | On screen | Your action |
|------|-----------|-------------|
| 0:00–0:15 | App header + judge strip | Scroll slightly so **Project Sonar** title + tagline visible |
| 0:15–0:45 | EMG stack + RMS + spectrogram | Point cursor at **multi-channel traces**, then **spectrogram** |
| 0:45–1:05 | Prediction row | Click **Predict**; pause on **gauge + word** |
| 1:05–1:35 | Scroll to XAI + AAC | Show **XAI** text; click a **quick phrase**, then **Speak** (TTS plays) |
| 1:35–2:00 | Sidebar or analytics | Open **Export JSON** / mention **session stats**; end on **footer** or title |

### Word-for-word narration (~2 minutes)

**0:00 – 0:20 | Hook**  
“Hi — this is **Project Sonar**: a dashboard for **silent speech**. People who can’t speak aloud still produce tiny muscle signals — EMG — but that signal is invisible and hard to interpret. We turn it into something **visual, interactive, and actionable**.”

**0:20 – 0:45 | Signal wall**  
“Here’s the **multi-channel EMG-style view** — you see energy over time per channel. Beside it, **RMS power** tells you which channels carry more of the signal. The **spectrogram** shows frequency content for channel one at a fixed sample rate used for consistent axis labels.”

**0:45 – 1:10 | Prediction**  
“I hit **Predict**. The model assigns a **word hypothesis** and a **confidence score** — so it’s not a black box: you see how sure the system is. In demo mode this is synthetic data; with **real PyTorch weights** loaded, this runs your trained **PyTorch** model.”

**1:10 – 1:40 | XAI + communication**  
“Below that, we surface **explainability-style context** — which channels dominated this window — and an **AAC-style panel**: quick phrases for urgent needs, a text buffer, and **Speak** so the browser reads the message out loud. That’s the accessibility angle — silent input, **audible output**.”

**1:40 – 2:00 | Impact + close**  
“We log **session analytics** and export **CSV or JSON** for judges or clinicians. Stack: **Streamlit** and **PyTorch**. **Project Sonar** — making silent speech **visible** and **usable**. Thanks.”

### If you run long

- Drop the sentence “In demo mode…” in block 3, or shorten block 2 by skipping RMS.

### Export / upload tips

- **Resolution:** 1920×1080 if possible; **30 fps** is enough  
- **Audio:** record **system audio** if TTS must be heard on the recording (OBS: capture desktop audio)  
- **Formats:** MP4 (H.264) for most hackathon portals  

---

## Run commands

**Windows (PowerShell)**

```powershell
.\run_hackathon_demo.ps1
```

**macOS / Linux**

```bash
chmod +x run_hackathon_demo.sh
./run_hackathon_demo.sh
```

**Manual**

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then open **http://localhost:8501**.

## Real PyTorch mode (optional, stronger technical score)

1. `pip install torch` (restart Streamlit after install).
2. Place pretrained weights at `pretrained_models/transduction_model.pt` (see README for layout).
3. Optional: `normalizers.pkl` next to `app.py` if you have training matched normalizers.

## Tech stack (for judges)

- **UI:** Streamlit  
- **Model:** PyTorch `Model` from `architecture.py` (when weights load)  
- **Signal processing:** NumPy / SciPy / Matplotlib; STFT labels use **689.06 Hz** in the app for consistent frequency axes  
- **Model:** optional PyTorch weights under `pretrained_models/`

## Honest limitations (if asked)

- **Demo mode** uses **synthetic** EMG unless you stream real data; mock predictions are random when no weights load.
- **Phoneme heatmap** in the UI is marked as **illustrative** unless you wire real phoneme logits from inference.
- **Channel “importance”** is mean absolute amplitude, not gradient attribution.

## Slide bullets (copy-paste)

- Silent speech + EMG → understandable dashboard  
- Prediction + confidence + session analytics  
- AAC + TTS for communication scenarios  
- Clear problem → dashboard → impact story  
- Extensible to real sensors and full decoding  

Good luck.
