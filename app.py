"""
AI Stem Separator — Gradio web app.

Upload a song and get its stems (vocals, drums, bass, other) separated
with Meta's Demucs (htdemucs model). Designed to run on Hugging Face Spaces.
"""

# On Hugging Face ZeroGPU Spaces, `spaces.GPU` attaches a GPU while the
# decorated function runs. It must be imported before torch.
# Locally (no `spaces` package) it is a no-op.
try:
    import spaces

    gpu = spaces.GPU(duration=60)
except ImportError:
    def gpu(fn):
        return fn

import tempfile
import zipfile
from pathlib import Path

import gradio as gr
import torch
from demucs.separate import main as demucs_main

MODEL = "htdemucs"
FOUR_STEMS = ["vocals", "drums", "bass", "other"]
TWO_STEMS = ["vocals", "no_vocals"]


@gpu
def separate(audio_path: str, mode: str):
    """Run Demucs on the uploaded file and return the stems plus a ZIP."""
    if not audio_path:
        raise gr.Error("Please upload an audio file first.")

    out_dir = Path(tempfile.mkdtemp(prefix="stems_"))
    two_stems = mode.startswith("2")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = ["-n", MODEL, "-o", str(out_dir), "-d", device]
    if two_stems:
        args += ["--two-stems", "vocals"]
    args.append(audio_path)

    try:
        demucs_main(args)
    except SystemExit as e:
        if e.code not in (0, None):
            raise gr.Error(f"Demucs failed (exit code {e.code}).")
    except Exception as e:
        raise gr.Error(f"Demucs failed: {e}")

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
        "_Runs on a free shared GPU: a 3–4 minute song takes about a minute._"
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
