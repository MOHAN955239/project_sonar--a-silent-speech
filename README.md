# Project Sonar

A **silent-speech interface** prototype: a **Streamlit** dashboard for EMG-style signal visualization, neural **Predict** with confidence, explainability-style panels, **AAC** quick phrases, and browser **text-to-speech** — oriented toward accessibility and communication use cases.

## Quick start

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

**Windows:** `.\run_hackathon_demo.ps1`  
**macOS / Linux:** `chmod +x run_hackathon_demo.sh && ./run_hackathon_demo.sh`

Then open **http://localhost:8501**.

## Docs

| File | Contents |
|------|----------|
| [HACKATHON.md](HACKATHON.md) | Demo scripts, 2-minute video narration, pitch bullets |
| [DEPLOY.md](DEPLOY.md) | Streamlit Cloud, Hugging Face Spaces, Render |

## Main app files

| Path | Role |
|------|------|
| `app.py` | Streamlit UI |
| `model_utils.py` | Model loading and inference helpers |
| `requirements.txt` | Dependencies |
| `Dockerfile` | Optional container deployment |
| `.streamlit/config.toml` | Streamlit theme |

Additional modules in this repository (for example `architecture.py`, `data_utils.py`, `read_emg.py`) support offline model and data workflows when you add your own datasets and checkpoints.

## License

See [LICENSE](LICENSE).
