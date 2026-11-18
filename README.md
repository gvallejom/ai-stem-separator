# 🎛️ AI Stem Separator

Upload any song and split it into its individual stems (**vocals, drums, bass and other**) using deep learning.

Built on top of Meta's [Demucs](https://github.com/facebookresearch/demucs) (`htdemucs`, Hybrid Transformer Demucs), with two ways to use it:

- 🌐 **Web app** (Gradio): upload a song in the browser and listen to or download each stem.
- 📓 **Google Colab notebook**: run it on a free GPU, no installation needed.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gvallejom/ai-stem-separator/blob/main/stem_separator.ipynb)
<!-- Uncomment once the Space is live:
[![Hugging Face Space](https://img.shields.io/badge/🤗%20Demo-Hugging%20Face%20Spaces-yellow)](https://huggingface.co/spaces/gvallejom/ai-stem-separator)
-->

<!-- Add a screenshot or GIF of the app here -->
<!-- ![Demo](docs/demo.gif) -->

---

## ✨ Features

- **4-stem separation**: vocals, drums, bass, other
- **2-stem mode**: vocals / instrumental (karaoke-style)
- Listen to each stem directly in the browser
- Download all stems in a single `.zip`
- Supports mp3, wav, flac and other common formats

## 🧠 How it works

```
 song.mp3 ──► Demucs (htdemucs) ──► vocals.wav
                                ├─► drums.wav
                                ├─► bass.wav
                                └─► other.wav  ──► stems.zip
```

`htdemucs` is a hybrid model that processes audio in both the **waveform** and the **spectrogram** domains, with a Transformer connecting the two. That makes it one of the strongest open-source models for music source separation.

## 🚀 Run it locally

```bash
git clone https://github.com/gvallejom/ai-stem-separator.git
cd ai-stem-separator
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:7860` in your browser.

> The first run downloads the model weights (~80 MB). On CPU, a 3–4 minute song takes a few minutes; with a GPU it takes seconds.

## 📓 Run it in Google Colab

1. Open [`stem_separator.ipynb`](stem_separator.ipynb) in Colab (badge above).
2. *Runtime → Change runtime type → GPU*.
3. Run the cells, upload your song and download the stems.

## 🛠️ Tech stack

- **Python**
- **Demucs** (PyTorch): source separation model
- **Gradio**: web interface
- **Google Colab / Hugging Face Spaces**: free hosting and GPU

## 🗺️ Roadmap

- [ ] Choose between models (`htdemucs_ft`, `mdx_extra`)
- [ ] 6-stem mode (adds guitar and piano with `htdemucs_6s`)
- [ ] Export to MP3 to reduce download size

## 📄 License

MIT. Demucs is © Meta Platforms, Inc., also under the MIT license.
