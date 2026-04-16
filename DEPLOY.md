# Deploying Project Sonar

## Can I use GitHub + Vercel?

**Not really.** [Vercel](https://vercel.com) is built for **static sites**, **serverless functions**, and frameworks like Next.js. **Streamlit** is a **long-running Python web server** that keeps a process alive. Vercel does not host Streamlit apps in the normal, supported way.

Use one of the options below instead — they are **free tiers** and fit this stack.

---

## Option A — Streamlit Community Cloud (easiest for Streamlit)

**Best match:** connect your **GitHub** repo; Streamlit runs `app.py` for you.

1. Push the project to GitHub (see [Git prep](#git-prep) below).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. **New app** → pick **repository**, **branch**, **main file:** `app.py`.
4. Deploy. It will `pip install -r requirements.txt` automatically.

**Notes**

- First deploy runs in **Demo mode** unless you add weights via [Secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app#secrets) (large models are usually hosted elsewhere, not in the repo).
- If the build is heavy (e.g. **torch**), the first build can take several minutes.
- Python version: set in the Cloud UI if your build needs a specific version (e.g. 3.11).

---

## Option B — Hugging Face Spaces (free, good visibility)

Spaces support **Streamlit** as a “SDK.”

1. Create a **new Space** → choose **Streamlit**.
2. Push your `app.py`, `requirements.txt`, and assets (or connect GitHub).
3. See: [Spaces — Streamlit](https://huggingface.co/docs/hub/spaces-sdks-streamlit).

Good for **demo + portfolio**; same caveat about **large checkpoints** (use smaller demo or download weights at runtime from a URL if allowed by the hackathon).

---

## Option C — Render (Docker)

You already have a `Dockerfile`.

1. Push the repo to GitHub.
2. [Render](https://render.com) → **New** → **Web Service** → connect repo.
3. **Environment:** Docker; expose port **8501** (as in your Dockerfile).

Free tier may **sleep** when idle (cold start on next visit).

---

## Option D — Fly.io / Railway / Google Cloud Run

All can run a **container** from your `Dockerfile`. Slightly more setup (CLI, billing guardrails). Use if you outgrow Streamlit Cloud’s limits.

---

## Git prep (before GitHub)

1. **Do not commit** huge files (`pretrained_models.zip`, `.pt` checkpoints) — they break GitHub limits and slow deploys. The sample `.gitignore` in this repo excludes common cases.
2. Commit: `app.py`, `model_utils.py`, `requirements.txt`, `architecture.py`, `data_utils.py`, etc.
3. If the hackathon requires a **private** repo, Streamlit Community Cloud has limits on private apps — check current [Streamlit Cloud pricing/plans](https://streamlit.io/cloud).

---

## After deploy

- Open the public URL and confirm **Demo mode** works (synthetic EMG + mock or real inference if you configured weights).
- For **secrets** (API keys, rarely model URLs): use each platform’s **Secrets / Environment variables** UI — never commit secrets to Git.

---

## Quick comparison

| Platform              | Streamlit-native | Free tier | Good for        |
|-----------------------|------------------|-----------|-----------------|
| Streamlit Cloud       | Yes              | Yes       | Fastest path    |
| Hugging Face Spaces   | Yes              | Yes       | Demos + sharing |
| Render (Docker)       | Via container    | Yes*      | Full control    |
| Vercel                | No               | —         | Not for Streamlit |

\*Subject to provider limits; may sleep on idle.
