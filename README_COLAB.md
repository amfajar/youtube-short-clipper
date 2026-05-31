# 🎬 YT-Short-Clipper - Google Colab Guide

This guide explains how to run **YT-Short-Clipper** on **Google Colab** to leverage free cloud GPUs for fast rendering and active-speaker tracking.

---

## 🚀 Setup Steps

1. **Open Google Colab**: Go to [colab.research.google.com](https://colab.research.google.com) and create a new notebook, or upload the `colab.ipynb` file in this repository.
2. **GPU Mode**: Select **Runtime > Change runtime type**, choose **T4 GPU** (or any other GPU accelerator), and click **Save**.
3. **Run Cell 1 & 2**: Install all the necessary binary dependencies and python packages automatically.
4. **Input Configuration**:
   - Paste your YouTube video URL.
   - Select your preferred AI provider (OpenRouter is default and supports free model rotation).
   - Paste your API Key.
5. **(Optional) Cookie upload**: If your video is age-restricted or private, export a `cookies.txt` using a browser extension (e.g. *Get cookies.txt LOCALLY*), and upload it inside **Cell 4**.
6. **Start Pipeline**: Run **Cell 5** to cut, track, and render the clips.
7. **Download**: Run **Cell 6** to download a compressed ZIP archive containing all the generated short clips.

---

## 🔑 AI Provider Guides

### 1. OpenRouter (Default & Recommended)
- **Why**: Supports free models with automatic rate limit rotation.
- **Link**: [openrouter.ai](https://openrouter.ai/)
- **Model name**: Set to `auto` to route across `"openrouter/free"` automatically, with automated secondary fail-safe rotation.

### 2. Google Gemini
- **Why**: Generous free tier quotas from Google AI Studio.
- **Link**: [aistudio.google.com](https://aistudio.google.com/)
- **Model name**: `gemini-2.5-flash` or `gemini-3.1-flash-lite`.

### 3. Groq
- **Why**: Ultra-fast execution times.
- **Link**: [console.groq.com](https://console.groq.com/)
- **Model name**: `mixtral-8x7b-32768`.

---

## 🛠️ Troubleshooting

- **Error: HTTP 403 Forbidden**: YouTube is rate-limiting the Google IP address. Solve this by exporting cookies from your browser using a cookies extension (while logged into YouTube) as `cookies.txt` and uploading it via Cell 4.
- **Whisper API Credits Error**: Captioning requires OpenAI/Groq API credits. If you are using a Gemini API key for highlight finding, supply a separate OpenAI API key in the Whisper uploader form settings to enable styled captions.
