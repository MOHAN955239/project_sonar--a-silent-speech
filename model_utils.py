"""
model_utils.py – Real + Mock hybrid model loader for Project Sonar
-------------------------------------------------------------------
• If a pretrained .pt checkpoint exists and torch is available, the real
  PyTorch Model from architecture.py is used for inference.
• If torch / the checkpoint is missing, a realistic mock prediction is
  returned so the Streamlit UI keeps working for demo purposes.
"""

import os
import sys
import pickle
import numpy as np

# ── Model paths (populated after Zenodo extract) ──────────────────────────────
PRETRAINED_TRANSDUCTION = os.path.join(os.path.dirname(__file__), "pretrained_models", "transduction_model.pt")
PRETRAINED_HIFIGAN      = os.path.join(os.path.dirname(__file__), "pretrained_models", "hifigan", "checkpoint")
NORMALIZERS_PATH        = os.path.join(os.path.dirname(__file__), "normalizers.pkl")

WORDS = ["HELP", "HELLO", "YES", "NO", "WATER", "THANK YOU", "STOP", "PLEASE"]

# ── Shared state set by absl FLAGS (called once from app) ─────────────────────
_flags_initialised = False

def _init_flags():
    global _flags_initialised
    if _flags_initialised:
        return
    try:
        from absl import flags
        FLAGS = flags.FLAGS
        # Mark flags as parsed so we can read defaults
        if not FLAGS.is_parsed():
            FLAGS(["model_utils"])  # fake argv[0]
    except Exception:
        pass
    _flags_initialised = True


# ── Torch availability guard ───────────────────────────────────────────────────
def _torch_available():
    try:
        import torch
        return True
    except ImportError:
        return False


# ── Real model loader ──────────────────────────────────────────────────────────
def load_model(model_path=None):
    """
    Returns a dict:
      {
        "real"  : bool,          # True  = real PyTorch weights loaded
        "model" : nn.Module | None,
        "device": str,
        "normalizers": (mfcc_norm, emg_norm) | None
      }
    """
    _init_flags()
    path = model_path or PRETRAINED_TRANSDUCTION

    _base = lambda **kw: {
        "real": False,
        "model": None,
        "device": "cpu",
        "normalizers": None,
        "demo_reason": None,
        "demo_detail": None,
        **kw,
    }

    if not _torch_available():
        return _base(
            demo_reason="no_torch",
            demo_detail="Install PyTorch, then restart Streamlit.",
        )

    if not os.path.exists(path):
        return _base(
            demo_reason="missing_weights",
            demo_detail=path,
        )

    try:
        import torch
        from architecture import Model
        from data_utils import phoneme_inventory

        device = "cuda" if torch.cuda.is_available() else "cpu"

        NUM_FEATURES  = 8
        NUM_SPEECH    = 80
        NUM_PHONEMES  = len(phoneme_inventory)

        model = Model(NUM_FEATURES, NUM_SPEECH, NUM_PHONEMES).to(device)
        try:
            state = torch.load(path, map_location=device, weights_only=True)
        except Exception:
            try:
                state = torch.load(path, map_location=device, weights_only=False)
            except TypeError:
                state = torch.load(path, map_location=device)
        if isinstance(state, dict):
            if "state_dict" in state:
                state = state["state_dict"]
            elif "model" in state and isinstance(state["model"], dict):
                state = state["model"]
        model.load_state_dict(state, strict=False)
        model.eval()

        normalizers = None
        if os.path.exists(NORMALIZERS_PATH):
            with open(NORMALIZERS_PATH, "rb") as f:
                normalizers = pickle.load(f)

        return {
            "real": True,
            "model": model,
            "device": device,
            "normalizers": normalizers,
            "demo_reason": None,
            "demo_detail": None,
        }
    except Exception as e:
        print(f"[model_utils] Could not load real model ({e}). Using mock.")
        return _base(demo_reason="load_error", demo_detail=str(e))


# ── Real inference ─────────────────────────────────────────────────────────────
def _real_predict(bundle, emg_raw: np.ndarray):
    """
    Runs the real model on a (T, 8) raw EMG numpy array.
    Returns (word_prediction: str, confidence: float)
    """
    import torch
    import torch.nn.functional as F
    from data_utils import get_emg_features

    model  = bundle["model"]
    device = bundle["device"]
    normalizers = bundle["normalizers"]

    # T_raw samples → compute features
    # emg_raw shape: (T_raw, 8)
    emg_features = get_emg_features(emg_raw)          # (T_feat, 8)

    if normalizers is not None:
        _, emg_norm = normalizers
        emg_features = emg_norm.normalize(emg_features)
        emg_features = 8 * np.tanh(emg_features / 8.0)

    # Prepare tensors: (1, T, feat)
    X     = torch.from_numpy(emg_features).unsqueeze(0).to(torch.float32).to(device)
    X_raw = torch.from_numpy(emg_raw).unsqueeze(0).to(torch.float32).to(device)
    sess  = torch.zeros(1, X.shape[1], dtype=torch.long).to(device)

    with torch.no_grad():
        out, phoneme_out = model(X, X_raw, sess)     # (1, T, 80), (1, T, 48)
        # Use phoneme logits → softmax → max for confidence proxy
        phone_probs = F.softmax(phoneme_out.squeeze(0), dim=-1)   # (T, 48)
        confidence  = float(phone_probs.max(dim=-1).values.mean().item()) * 100.0
        confidence  = min(max(confidence, 50.0), 99.9)

    # Without a full language model decode we map the dominant phoneme index to a word label
    dominant_phone_idx = int(phone_probs.mean(dim=0).argmax().item())
    word = WORDS[dominant_phone_idx % len(WORDS)]
    return word, confidence


# ── Mock inference (kept for demo when no weights/torch) ─────────────────────
def _mock_predict():
    import random, time
    time.sleep(0.4)  # simulate processing
    word       = random.choice(WORDS)
    confidence = round(np.random.uniform(72.0, 99.5), 1)
    return word, confidence


def _ensure_8ch_emg(emg_raw: np.ndarray) -> np.ndarray:
    """Match training: 8 EMG channels; pad with zeros or truncate as needed."""
    x = np.asarray(emg_raw, dtype=np.float32)
    if x.ndim == 1:
        return np.tile(x[:, None], (1, 8))
    if x.ndim != 2:
        raise ValueError("emg_raw must be 1D or 2D")
    c = x.shape[1]
    if c == 8:
        return x.astype(np.float32)
    if c < 8:
        pad = np.zeros((x.shape[0], 8 - c), dtype=np.float32)
        return np.concatenate([x.astype(np.float32), pad], axis=1)
    return x[:, :8].astype(np.float32)


# ── Public predict ─────────────────────────────────────────────────────────────
def predict(bundle, emg_raw: np.ndarray):
    """
    Args:
        bundle   – dict returned by load_model()
        emg_raw  – np.ndarray  (T,) or (T, C)   raw EMG (C channels; padded/truncated to 8 for the model)
    Returns:
        (word: str, confidence: float%)
    """
    if bundle.get("real") and bundle.get("model") is not None:
        try:
            return _real_predict(bundle, _ensure_8ch_emg(emg_raw))
        except Exception as e:
            print(f"[model_utils] Real inference failed ({e}). Falling back to mock.")
    return _mock_predict()


# ── EMG channel importance for XAI ───────────────────────────────────────────
def get_channel_importance(emg_raw: np.ndarray) -> dict:
    """
    Returns per-channel mean absolute amplitude – used as a simple
    proxy for feature importance in the XAI panel (not gradient-based).
    """
    if emg_raw.ndim == 1:
        emg_raw = emg_raw[:, None]
    importances = np.abs(emg_raw).mean(axis=0)
    total = float(importances.sum()) + 1e-9
    return {f"CH{i+1}": float(importances[i] / total * 100) for i in range(importances.shape[0])}
