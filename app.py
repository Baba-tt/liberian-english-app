import os
import json
import csv
from datetime import datetime
from typing import Optional, Tuple

import gradio as gr
from faster_whisper import WhisperModel
from transformers import pipeline

DATA_DIR = os.path.join("data")
TRAIN_CSV = os.path.join(DATA_DIR, "training_data.csv")
RULES_PATH = os.path.join("rules.json")

os.makedirs(DATA_DIR, exist_ok=True)

# === Config (env overrideable) ===
DEFAULT_ASR_SIZE = os.environ.get("ASR_SIZE", "small")  # tiny/base/small/medium/large-v2
DEFAULT_DEVICE = os.environ.get("DEVICE", "cpu")        # "cpu" or "cuda"
DEFAULT_COMPUTE = os.environ.get("WHISPER_COMPUTE", "int8")
EN_TO_LIB_MODEL = os.environ.get("EN_TO_LIB_MODEL", "outputs/latest_en2lib")
LIB_TO_EN_MODEL = os.environ.get("LIB_TO_EN_MODEL", "outputs/latest_lib2en")

# === Rules fallback ===
def load_rules(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"replacements": [], "directions": {}}

RULES = load_rules(RULES_PATH)

def apply_rules(text: str, direction: str) -> str:
    # direction-aware replacements first
    out = text or ""
    dir_rules = RULES.get("directions", {}).get(direction, [])
    for pair in dir_rules:
        src = pair.get("from", "")
        tgt = pair.get("to", "")
        if src:
            out = out.replace(src, tgt).replace(src.capitalize(), tgt.capitalize())
    # global replacements
    for pair in RULES.get("replacements", []):
        src = pair.get("from", "")
        tgt = pair.get("to", "")
        if src:
            out = out.replace(src, tgt).replace(src.capitalize(), tgt.capitalize())
    return " ".join(out.split())

# === Lazy-loaded models ===
ASR_MODEL: Optional[WhisperModel] = None
EN2LIB = None
LIB2EN = None

def load_asr(size: str, device: str, compute: str) -> WhisperModel:
    global ASR_MODEL
    ASR_MODEL = WhisperModel(size, device=device, compute_type=compute)
    return ASR_MODEL

def maybe_load_translator(model_id: str):
    if model_id and os.path.exists(model_id):
        return pipeline("text2text-generation", model=model_id, device_map="auto")
    if model_id and "/" in model_id:
        # Treat as HF repo
        return pipeline("text2text-generation", model=model_id, device_map="auto")
    return None

def load_translators(en2lib_id: str, lib2en_id: str):
    global EN2LIB, LIB2EN
    EN2LIB = maybe_load_translator(en2lib_id)
    LIB2EN = maybe_load_translator(lib2en_id)
    return EN2LIB, LIB2EN

# Preload at import time
load_asr(DEFAULT_ASR_SIZE, DEFAULT_DEVICE, DEFAULT_COMPUTE)
load_translators(EN_TO_LIB_MODEL, LIB_TO_EN_MODEL)

# === Core functions ===
def transcribe(audio_path: str) -> Tuple[str, str]:
    assert ASR_MODEL is not None, "ASR not loaded"
    segments, info = ASR_MODEL.transcribe(
        audio_path,
        task="transcribe",
        beam_size=5,
        vad_filter=True,
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return info.language or "", text

def en_to_lib(text: str) -> str:
    if EN2LIB is not None:
        out = EN2LIB(text, max_new_tokens=128, do_sample=False)[0]["generated_text"].strip()
        return out
    return apply_rules(text, "English → Liberian English")

def lib_to_en(text: str) -> str:
    if LIB2EN is not None:
        out = LIB2EN(text, max_new_tokens=128, do_sample=False)[0]["generated_text"].strip()
        return out
    return apply_rules(text, "Liberian English → English") or text

def save_example(src: str, model_out: str, corrected: str, notes: str = "", direction: str = "") -> str:
    newfile = not os.path.exists(TRAIN_CSV)
    with open(TRAIN_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if newfile:
            writer.writerow(["direction", "source_text", "model_output", "corrected_output", "notes", "created_at"])
        writer.writerow([direction, src, model_out, corrected, notes, datetime.utcnow().isoformat()])
    try:
        total = sum(1 for _ in open(TRAIN_CSV, "r", encoding="utf-8")) - 1
    except Exception:
        total = "?"
    return f"Saved. Total examples: {total}"

# === Gradio UI (exported as `demo` for mounting) ===
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🇱🇷 Liberian English ↔ English Transcriber
        Upload or record audio → auto-transcribe → convert either direction.
        """
    )

    with gr.Row():
        with gr.Column():
            audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio")
            direction = gr.Radio(["English → Liberian English", "Liberian English → English"],
                                 value="English → Liberian English", label="Conversion direction")
            btn_run = gr.Button("Transcribe → Convert", variant="primary")
            src_lang = gr.Textbox(label="Detected language", interactive=False)
            transcript = gr.Textbox(label="Transcript (raw)")
        with gr.Column():
            en2lib_id = gr.Textbox(value=EN_TO_LIB_MODEL, label="English→Liberian model ID (HF repo or local path)")
            lib2en_id = gr.Textbox(value=LIB_TO_EN_MODEL, label="Liberian→English model ID (HF repo or local path)")
            model_status = gr.Markdown(visible=False)
            output = gr.Textbox(label="Converted output (editable)")
            notes = gr.Textbox(label="Notes (optional)")
            btn_save = gr.Button("Save example")
            save_status = gr.Markdown()

    with gr.Accordion("Advanced settings", open=False):
        asr_size = gr.Dropdown(["tiny", "base", "small", "medium", "large-v2"], value=DEFAULT_ASR_SIZE, label="ASR model size")
        device = gr.Dropdown(["cpu", "cuda"], value=DEFAULT_DEVICE, label="Device")
        compute = gr.Dropdown(["int8", "int8_float16", "float16", "float32"], value=DEFAULT_COMPUTE, label="Whisper compute type")
        btn_reload_asr = gr.Button("Reload ASR")
        btn_reload_translators = gr.Button("Reload Translators")

    def _run(audio_path, dir_choice):
        if not audio_path:
            return "", "", ""
        lang, text = transcribe(audio_path)
        out = en_to_lib(text) if dir_choice == "English → Liberian English" else lib_to_en(text)
        return lang, text, out

    def _save(src, out, corr, n, dir_choice):
        if not (src or out or corr):
            return "Nothing to save."
        return save_example(src, out, corr or out, n or "", dir_choice)

    def _reload_asr(asr_sz, dev, comp):
        load_asr(asr_sz, dev, comp)
        return f"Reloaded ASR={asr_sz} on {dev} ({comp})", gr.update(visible=True)

    def _reload_translators(en2lib, lib2en):
        load_translators(en2lib.strip(), lib2en.strip())
        msg = f"Reloaded translators. EN→LIB={'custom' if en2lib.strip() else 'rules'}, LIB→EN={'custom' if lib2en.strip() else 'rules/identity'}"
        return msg, gr.update(visible=True)

    btn_run.click(_run, inputs=[audio, direction], outputs=[src_lang, transcript, output])
    btn_save.click(_save, inputs=[transcript, output, output, notes, direction], outputs=[save_status])
    btn_reload_asr.click(_reload_asr, inputs=[asr_size, device, compute], outputs=[model_status, model_status])
    btn_reload_translators.click(_reload_translators, inputs=[en2lib_id, lib2en_id], outputs=[model_status, model_status])

# Allow running directly, useful for local dev
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
