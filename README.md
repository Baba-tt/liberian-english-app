# Liberian English Transcriber (EN ↔ LIB) — Ad-Ready

This app transcribes speech and converts **English ↔ Liberian English**.
- Built with **Gradio** (UI) and **FastAPI** (server)
- **Ads-ready** via `templates/index.html` (Google AdSense)
- **Trainable**: fine-tune EN→LIB and LIB→EN models on your collected corrections

## Quick Start (Local)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# run
python server.py
# open http://localhost:5000
```

> Note: If audio decoding fails for certain formats, upload **.wav** files.

## Training
Collect data by editing outputs in the app and clicking **Save example**. Then:
```bash
python train.py --csv data/training_data.csv --epochs 5 --batch 8 --lr 2e-4 --fp16
```
Upload the trained folders to HF Hub or set env vars:
- `EN_TO_LIB_MODEL` → path or repo for EN→LIB
- `LIB_TO_EN_MODEL` → path or repo for LIB→EN

## Deploy (Render)
- Push this repo to GitHub
- Connect on Render → it uses `render.yaml`
- Live at: `https://<your-app>.onrender.com`

## Deploy (Heroku)
- Ensure `Procfile` present
- `heroku create your-app && git push heroku main`

## Ownership
```
Copyright (c) 2025 Baba Tamba
All rights reserved.
```
Replace with your name to assert ownership.

## Environment Tips
- CPU-only works; CUDA speeds up if available (`DEVICE=cuda`, `WHISPER_COMPUTE=float16`).
