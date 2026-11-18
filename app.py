"""
AI Stem Separator — Gradio web app.

Upload a song and get its stems (vocals, drums, bass, other) separated
with Meta's Demucs (htdemucs model). Designed to run on Hugging Face Spaces.
"""

import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import gradio as gr

MODEL = "htdemucs"
FOUR_STEMS = ["vocals", "drums", "bass", "other"]
TWO_STEMS = ["vocals", "no_vocals"]


def separate(audio_path: str, mode: str):
    """Run Demucs on the uploaded file and return the stems plus a ZIP."""
    if not audio_path:
        raise gr.Error("Sube un archivo de audio primero / Please upload an audio file first.")

    out_dir = Path(tempfile.mkdtemp(prefix="stems_"))
    two_stems = mode.startswith("2")

    cmd = [sys.executable, "-m", "demucs", "-n", MODEL, "-o", str(out_dir)]
    if two_stems:
        cmd += ["--two-stems", "vocals"]
    cmd.append(audio_path)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise gr.Error(f"Demucs failed:\n{result.stderr[-1000:]}")

    track_name = Path(audio_path).stem
    stem_dir = out_dir / MODEL / track_name
    stems = TWO_STEMS if two_stems else FOUR_STEMS
    paths = [stem_dir / f"{s}.wav" for s in stems]

    zip_path = out_dir / f"{track_name}_stems.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for p in paths:
            if p.exists():
                zf.write(p, arcname=p.name)

    # Always return 4 audio slots; unused ones stay empty in 2-stem mode.
    audio_outputs = [str(p) if p.exists() else None for p in paths]
    audio_outputs += [None] * (4 - len(audio_outputs))
    return (*audio_outputs, str(zip_path))


with gr.Blocks(title="AI Stem Separator") as demo:
    gr.Markdown(
        "# 🎛️ AI Stem Separator\n"
        "Upload a song and split it into **vocals, drums, bass and other** using "
        "[Demucs](https://github.com/facebookresearch/demucs) (`htdemucs`).\n\n"
        "_Processing a 3–4 minute song can take a few minutes on CPU._"
    )

    with gr.Row():
        audio_in = gr.Audio(type="filepath", label="Song (mp3, wav, flac…)")
        mode = gr.Radio(
            ["4 stems (vocals, drums, bass, other)", "2 stems (vocals / instrumental)"],
            value="4 stems (vocals, drums, bass, other)",
            label="Mode",
        )

    run_btn = gr.Button("Separate", variant="primary")

    with gr.Row():
        out_1 = gr.Audio(label="Vocals")
        out_2 = gr.Audio(label="Drums / Instrumental")
    with gr.Row():
        out_3 = gr.Audio(label="Bass")
        out_4 = gr.Audio(label="Other")
    zip_out = gr.File(label="Download all stems (.zip)")

    run_btn.click(
        separate,
        inputs=[audio_in, mode],
        outputs=[out_1, out_2, out_3, out_4, zip_out],
    )

if __name__ == "__main__":
    demo.queue().launch()
