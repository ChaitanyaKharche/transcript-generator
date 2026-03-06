# Transcript Generator

Local video/audio transcription tool using **Whisper V3 Turbo** (HF Inference API) with **LLM-powered post-processing** for clean, readable output.

## Architecture

```
Audio/Video Upload → ffmpeg conversion (16kHz mono WAV) → Whisper V3 Turbo (HF API)
                                                              ↓
                                                         Raw transcript
                                                              ↓
                                              LLM Polish: HF Llama 3.1 8B
                                                   ↓ (fallback)
                                              Groq Llama 3.1 8B Instant
                                                   ↓ (fallback)
                                              Regex (last resort)
                                                              ↓
                                                      Clean transcript
```

## Transcription Quality Metrics

Compared against a paid transcription service on a 5-min two-speaker conversation:

| Metric | Regex Only | + LLM Polish | Paid Service |
|---|---|---|---|
| Punctuation accuracy | ~60% | ~90% | ~95% |
| Speaker turn detection | None | Paragraph breaks | Dash-delimited |
| Technical term accuracy | Raw Whisper output | Same (Whisper limitation) | ~85% |
| Filler/backchannel capture | Dropped | Partial | Full (`[laughs]`, `Mm-hmm`) |
| Proper noun accuracy | Phonetic guesses | Same (no context) | ~80% |

**Key insight:** The LLM polish step closes ~70% of the gap between raw Whisper and paid services at zero additional cost. Remaining gaps (proper nouns, code terms, speaker diarization) are Whisper-level limitations.

## Setup

```bash
# 1. Clone and enter project
git clone <repo-url> && cd transcriptgenerator

# 2. Create venv and activate
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up .env
echo HF_TOKEN=hf_your_key_here > .env
echo GROQ_API_KEY=gsk_your_key_here >> .env
```

> **HF_TOKEN** (required): [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
> **GROQ_API_KEY** (optional, fallback): [console.groq.com](https://console.groq.com)

### Prerequisites

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/download.html) installed and in PATH

## Run

```bash
python app.py
```

Open [http://localhost:7860](http://localhost:7860)

## Supported Inputs

- **File upload:** MP3, MP4, WAV, M4A, WebM
- **URL paste:** YouTube, TikTok, Instagram Reels (requires `yt-dlp`, local deployment only)

## Stack

- **ASR:** `openai/whisper-large-v3-turbo` via HF Inference API
- **LLM polish:** `meta-llama/Llama-3.1-8B-Instruct` (HF) → `llama-3.1-8b-instant` (Groq fallback)
- **Backend:** FastAPI + Uvicorn
- **Audio processing:** pydub + ffmpeg