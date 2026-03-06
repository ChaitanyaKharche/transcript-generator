import os
import subprocess
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import uuid
from huggingface_hub import InferenceClient
from pydub import AudioSegment
import shutil
from dotenv import load_dotenv
import re
import logging

load_dotenv()

app = FastAPI()

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

# HF Inference API setup — model runs on HF servers, no local GPU needed
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not found in .env — add your Hugging Face API key")

MODEL = "openai/whisper-large-v3-turbo"
client = InferenceClient(token=HF_TOKEN)

# Optional Groq fallback — set GROQ_API_KEY in .env for fallback LLM polish
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# LLM model choices
HF_LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
GROQ_LLM_MODEL = "llama-3.1-8b-instant"

logger = logging.getLogger("transcriber")
logging.basicConfig(level=logging.INFO)

TEMP_DIR = Path("./temp_videos")
TEMP_DIR.mkdir(exist_ok=True)


class VideoURL(BaseModel):
  url: str

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Video Transcriber</title>
  <style>
    :root {
      --color-primary: #2180cb;
      --color-bg: #f5f5f5;
      --color-surface: #ffffff;
      --color-text: #1a1a1a;
      --color-error: #dc2626;
      --color-success: #16a34a;
      --radius: 8px;
      --shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }

    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: var(--color-bg);
      color: var(--color-text);
      padding: 20px;
      min-height: 100vh;
    }

    .container {
      max-width: 1000px;
      margin: 0 auto;
    }

    .header {
      margin-bottom: 30px;
    }

    h1 {
      font-size: 28px;
      margin-bottom: 8px;
    }

    .subtitle {
      color: #666;
      font-size: 14px;
    }

    .card {
      background: var(--color-surface);
      border-radius: var(--radius);
      padding: 24px;
      box-shadow: var(--shadow);
      margin-bottom: 20px;
    }

    .form-group {
      margin-bottom: 16px;
    }

    label {
      display: block;
      font-weight: 500;
      margin-bottom: 8px;
      font-size: 14px;
    }

    input[type="text"] {
      width: 100%;
      padding: 12px;
      border: 1px solid #ddd;
      border-radius: var(--radius);
      font-size: 14px;
      font-family: inherit;
    }

    input[type="text"]:focus {
      outline: none;
      border-color: var(--color-primary);
      box-shadow: 0 0 0 3px rgba(33, 128, 203, 0.1);
    }

    .button-group {
      display: flex;
      gap: 12px;
      margin-top: 20px;
    }

    button {
      padding: 12px 20px;
      border: none;
      border-radius: var(--radius);
      font-size: 14px;
      font-weight: 500;
      cursor: pointer;
      transition: all 150ms ease;
    }

    .btn-primary {
      background: var(--color-primary);
      color: white;
    }

    .btn-primary:hover:not(:disabled) {
      background: #1e6fa8;
    }

    .btn-primary:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }

    .btn-secondary {
      background: #f0f0f0;
      color: var(--color-text);
    }

    .btn-secondary:hover:not(:disabled) {
      background: #e0e0e0;
    }

    .status {
      padding: 12px;
      border-radius: var(--radius);
      font-size: 14px;
      margin-top: 16px;
      display: none;
    }

    .status.show {
      display: block;
    }

    .status.loading {
      background: #e3f2fd;
      color: #1565c0;
      border: 1px solid #90caf9;
    }

    .status.success {
      background: #f1f8e9;
      color: #558b2f;
      border: 1px solid #c5e1a5;
    }

    .status.error {
      background: #ffebee;
      color: #c62828;
      border: 1px solid #ef9a9a;
    }

    .transcript-section {
      display: none;
    }

    .transcript-section.show {
      display: block;
    }

    .transcript-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
    }

    .transcript-header h2 {
      font-size: 18px;
    }

    .copy-btn {
      padding: 8px 16px;
      background: #f0f0f0;
      color: var(--color-text);
      border: 1px solid #ddd;
      font-size: 13px;
    }

    .copy-btn:hover {
      background: #e0e0e0;
    }

    .transcript-box {
      background: #f9f9f9;
      border: 1px solid #ddd;
      border-radius: var(--radius);
      padding: 16px;
      font-family: 'Courier New', monospace;
      font-size: 14px;
      line-height: 1.6;
      white-space: pre-wrap;
      word-wrap: break-word;
      max-height: 500px;
      overflow-y: auto;
    }

    .spinner {
      display: inline-block;
      width: 16px;
      height: 16px;
      border: 2px solid #f3f3f3;
      border-top: 2px solid var(--color-primary);
      border-radius: 50%;
      animation: spin 1s linear infinite;
      margin-right: 8px;
      vertical-align: middle;
    }

    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }

    .url-examples {
      background: #f9f9f9;
      border-left: 3px solid var(--color-primary);
      padding: 12px;
      margin-top: 16px;
      border-radius: 4px;
      font-size: 13px;
    }

    .url-examples strong {
      display: block;
      margin-bottom: 8px;
    }

    .url-examples code {
      display: block;
      background: white;
      padding: 8px;
      margin-top: 4px;
      border-radius: 4px;
      font-family: 'Courier New', monospace;
      color: #666;
      overflow-x: auto;
    }

    .input-method-tabs {
      display: flex;
      gap: 12px;
      margin-bottom: 16px;
      border-bottom: 2px solid #ddd;
    }

    .tab-btn {
      padding: 12px 16px;
      background: transparent;
      border: none;
      border-bottom: 3px solid transparent;
      color: var(--color-text);
      font-weight: 500;
      cursor: pointer;
      transition: all 150ms ease;
      margin-bottom: -2px;
    }

    .tab-btn:hover {
      color: var(--color-primary);
    }

    .tab-btn.active {
      color: var(--color-primary);
      border-bottom-color: var(--color-primary);
    }

    .tab-content {
      display: none;
    }

    .tab-content.active {
      display: block;
    }

    input[type="file"] {
      display: block;
      width: 100%;
      padding: 12px;
      border: 2px dashed #ddd;
      border-radius: var(--radius);
      font-size: 14px;
      cursor: pointer;
      transition: all 150ms ease;
    }

    input[type="file"]:hover {
      border-color: var(--color-primary);
      background: rgba(33, 128, 203, 0.03);
    }

    input[type="file"]:focus {
      outline: none;
      border-color: var(--color-primary);
      box-shadow: 0 0 0 3px rgba(33, 128, 203, 0.1);
    }

    .file-info {
      margin-top: 8px;
      font-size: 12px;
      color: #999;
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>🎥 Video Transcriber</h1>
      <p class="subtitle">Extract transcripts from TikTok, Instagram Reels, YouTube, and more using Whisper V3 Turbo</p>
    </div>

    <div class="card">
      <form id="transcribeForm">
        <div class="form-group">
          <label>Choose Input Method</label>
          <div class="input-method-tabs">
            <button type="button" class="tab-btn active" data-method="upload">
              📁 Upload File
            </button>
            <button type="button" class="tab-btn" data-method="url">
              🔗 Paste URL
            </button>
          </div>
        </div>

        <div id="uploadTab" class="tab-content active">
          <div class="form-group">
            <label for="videoFile">Select Audio or Video File</label>
            <input 
              type="file" 
              id="videoFile" 
              accept="audio/*,video/*"
            >
            <div class="file-info">
              <small>Supported: MP3, MP4, WAV, M4A, WebM, etc. (Max 500MB)</small>
            </div>
          </div>
        </div>

        <div id="urlTab" class="tab-content">
          <div class="form-group">
            <label for="videoUrl">Video URL</label>
            <input 
              type="text" 
              id="videoUrl" 
              placeholder="Paste TikTok, Instagram, or YouTube URL here"
            >
            <div class="url-examples">
              <strong>Supported:</strong>
              <code>https://www.youtube.com/watch?v=dQw4w9WgXcQ</code>
              <code>https://www.tiktok.com/@creator/video/123456</code>
              <code>https://www.instagram.com/reel/ABC123/</code>
              <small style="color: #999; margin-top: 8px; display: block;">⚠️ URL mode requires local deployment (not working on free HF Spaces)</small>
            </div>
          </div>
        </div>

        <div class="button-group">
          <button type="submit" class="btn-primary" id="transcribeBtn">
            Transcribe
          </button>
          <button type="button" class="btn-secondary" id="clearBtn">
            Clear
          </button>
        </div>

        <div class="status" id="status"></div>
      </form>
     
    </div>

    <div class="transcript-section" id="transcriptSection">
      <div class="card">
        <div class="transcript-header">
          <h2>Transcript</h2>
          <button class="copy-btn" id="copyBtn">Copy to Clipboard</button>
        </div>
        <div class="transcript-box" id="transcriptBox"></div>
      </div>
    </div>
  </div>

  <script>
    const MODEL = 'openai/whisper-large-v3-turbo';
    const API_URL = window.location.origin;

    const form = document.getElementById('transcribeForm');
    const urlInput = document.getElementById('videoUrl');
    const fileInput = document.getElementById('videoFile');
    const transcribeBtn = document.getElementById('transcribeBtn');
    const clearBtn = document.getElementById('clearBtn');
    const status = document.getElementById('status');
    const transcriptSection = document.getElementById('transcriptSection');
    const transcriptBox = document.getElementById('transcriptBox');
    const copyBtn = document.getElementById('copyBtn');
    let currentMethod = 'upload';

    function showStatus(message, type = 'loading') {
      status.textContent = type === 'loading' ? '' : message;
      status.className = `status show ${type}`;
      
      if (type === 'loading') {
        status.innerHTML = `<span class="spinner"></span>${message}`;
      }
    }

    function hideStatus() {
      status.classList.remove('show');
    }

    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.preventDefault();
        const method = btn.dataset.method;
        
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
        if (method === 'upload') {
          document.getElementById('uploadTab').classList.add('active');
        } else {
            document.getElementById('urlTab').classList.add('active');
        }
        
        currentMethod = method;
      });
    });

    async function transcribeFile(file) {
      showStatus('Reading and transcribing file...', 'loading');

      try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_URL}/api/transcribe-file`, {
          method: 'POST',
          body: formData
        });

        if (!response.ok) {
          const error = await response.json();
          throw new Error(error.detail || `Error: ${response.status}`);
        }

        const data = await response.json();
        return data.transcript;
      } catch (error) {
        throw new Error(`Transcription failed: ${error.message}`);
      }
    }

    async function transcribeVideo(videoUrl) {
      showStatus('Downloading and transcribing video...', 'loading');

      try {
        const response = await fetch(`${API_URL}/api/transcribe`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ url: videoUrl })
        });

        if (!response.ok) {
          const error = await response.json();
          throw new Error(error.detail || `Error: ${response.status}`);
        }

        const data = await response.json();
        return data.transcript;
      } catch (error) {
        throw new Error(`Transcription failed: ${error.message}`);
      }
    }

    function displayTranscript(text) {
      transcriptBox.textContent = text;
      transcriptSection.classList.add('show');
      showStatus('Transcription complete!', 'success');
    }

    copyBtn.addEventListener('click', () => {
      navigator.clipboard.writeText(transcriptBox.textContent)
        .then(() => {
          const originalText = copyBtn.textContent;
          copyBtn.textContent = '✓ Copied!';
          setTimeout(() => {
            copyBtn.textContent = originalText;
          }, 2000);
        })
        .catch(() => showStatus('Failed to copy', 'error'));
    });

    clearBtn.addEventListener('click', () => {
      urlInput.value = '';
      fileInput.value = '';
      transcriptSection.classList.remove('show');
      hideStatus();
    });

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      
      transcribeBtn.disabled = true;

      try {
        let transcript;
        if (currentMethod === 'upload') {
          if (!fileInput.files.length) {
            showStatus('Please select a file', 'error');
            transcribeBtn.disabled = false;
            return;
          }
          transcript = await transcribeFile(fileInput.files[0]);
        } else {
          const url = urlInput.value.trim();
          if (!url) {
            showStatus('Please enter a URL', 'error');
            transcribeBtn.disabled = false;
            return;
          }
          transcript = await transcribeVideo(url);
        }
        
        displayTranscript(transcript);
      } catch (error) {
        showStatus(error.message, 'error');
      } finally {
        transcribeBtn.disabled = false;
      }
    });
  </script>
</body>
</html>"""

POLISH_SYSTEM_PROMPT = """You are a transcript post-processor. You receive raw ASR (Whisper) output and clean it up.

Rules:
1. Fix punctuation, capitalization, and sentence boundaries
2. Mark speaker turns with line breaks when you detect a change in speaker
3. Add filler markers like "Uh," "Mm-hmm," "[laughs]" where the raw text suggests them
4. Fix obvious phonetic misrecognitions of technical terms (e.g., "searchogs" → "search_args", "loadLM" → "load_llm")
5. Do NOT add, remove, or change the meaning of any words — only fix formatting
6. Do NOT summarize — output the FULL cleaned transcript
7. Keep it natural and conversational — this is a real conversation, not a formal document

Output ONLY the cleaned transcript. No preamble, no explanations."""


def clean_transcript_regex(text: str) -> str:
    """Regex fallback — basic capitalization and spacing fixes."""
    def capitalize_sentence(match):
        return match.group(0).upper()
    
    text = re.sub(r'([a-z])', capitalize_sentence, text, 1)
    text = re.sub(r'([.!?]\s*)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
    text = re.sub(r'\b(i)\b', 'I', text)
    text = re.sub(r'\s([,.?!:])', r'\1', text)
    text = text.replace('. ', '.\n\n')
    return text.strip()


def polish_with_hf(raw_text: str) -> str:
    """Try HF Inference API chat completion for transcript cleanup."""
    response = client.chat_completion(
        model=HF_LLM_MODEL,
        messages=[
            {"role": "system", "content": POLISH_SYSTEM_PROMPT},
            {"role": "user", "content": f"Clean up this raw transcript:\n\n{raw_text}"},
        ],
        max_tokens=4096,
        temperature=0.1,
    )
    result = response.choices[0].message.content
    if not result or len(result.strip()) < len(raw_text) * 0.5:
        raise ValueError("HF LLM returned truncated or empty result")
    return result.strip()


def polish_with_groq(raw_text: str) -> str:
    """Fallback to Groq REST API for transcript cleanup."""
    import requests as req
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set")
    
    resp = req.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": GROQ_LLM_MODEL,
            "messages": [
                {"role": "system", "content": POLISH_SYSTEM_PROMPT},
                {"role": "user", "content": f"Clean up this raw transcript:\n\n{raw_text}"},
            ],
            "max_tokens": 4096,
            "temperature": 0.1,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    result = data["choices"][0]["message"]["content"]
    if not result or len(result.strip()) < len(raw_text) * 0.5:
        raise ValueError("Groq LLM returned truncated or empty result")
    return result.strip()


def clean_transcript(raw_text: str) -> str:
    """
    LLM-powered transcript polish pipeline.
    HF Inference → Groq fallback → regex last resort.
    """
    # Try HF first (already authenticated)
    try:
        polished = polish_with_hf(raw_text)
        logger.info("Transcript polished via HF LLM")
        return polished
    except Exception as e:
        logger.warning(f"HF LLM polish failed: {e}")

    # Groq fallback
    try:
        polished = polish_with_groq(raw_text)
        logger.info("Transcript polished via Groq LLM")
        return polished
    except Exception as e:
        logger.warning(f"Groq LLM polish failed: {e}")

    # Regex last resort
    logger.warning("All LLM polishers failed — falling back to regex")
    return clean_transcript_regex(raw_text)


def download_video_audio(video_url: str, session_id: str) -> str:
  """Download video, extract audio, and force format to 16kHz Mono WAV."""
  try:
    output_path = TEMP_DIR / f"{session_id}.mp3"
    
    cmd = [
      "yt-dlp",
      "--socket-timeout", "30",
      "-x",
      "--audio-format", "mp3",
      "--audio-quality", "192",
      "-o", str(output_path),
      "-q",
      video_url
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    
    if result.returncode != 0:
      error_msg = result.stderr
      if "No address associated with hostname" in error_msg or "Failed to resolve" in error_msg:
        raise Exception("Network error: Cannot reach video server. This mode requires local deployment.")
      raise Exception(f"Download error: {error_msg[:200]}")
    
    if not output_path.exists():
      raise Exception("Audio extraction failed")
    
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
      raise Exception("ffmpeg not found in PATH.")
    
    wav_path = TEMP_DIR / f"{session_id}.wav"
    
    # Load MP3
    audio = AudioSegment.from_mp3(str(output_path))
    
    # CRITICAL FIX: Force 16kHz sample rate and Mono channel for Whisper
    audio = audio.set_frame_rate(16000).set_channels(1)
    
    audio.export(str(wav_path), format="wav")
    output_path.unlink()
    
    return str(wav_path)
  
  except Exception as e:
    raise Exception(f"Download failed: {str(e)}")

# Transcription via HF Inference API — no local GPU required
def transcribe_whisper(audio_path: str) -> str:
    """Transcribe audio using HF Inference API (Whisper on HF servers)."""
    try:
        # Pass the file path directly — huggingface_hub reads the file
        # and sets Content-Type from the extension automatically
        result = client.automatic_speech_recognition(
            audio=audio_path,
            model=MODEL,
        )

        # result can be a dict with 'text' key or a string directly
        if isinstance(result, dict):
            return result.get("text", "")
        return str(result)

    except Exception as e:
        error_message = str(e)
        if "rate limit" in error_message.lower():
            error_message = "HF API rate limited — wait a moment and retry."
        elif "401" in error_message or "unauthorized" in error_message.lower():
            error_message = "Invalid HF_TOKEN — check your .env file."
        raise Exception(f"Transcription failed: {error_message}")

@app.get("/")
async def serve_html():
  return HTMLResponse(HTML_CONTENT)

@app.post("/api/transcribe")
async def transcribe_endpoint(video_data: VideoURL):
  """Transcribe from video URL"""
  session_id = str(uuid.uuid4())[:8]
  
  try:
    audio_path = download_video_audio(video_data.url, session_id)
    raw_transcript = transcribe_whisper(audio_path)
    final_transcript = clean_transcript(raw_transcript)
    Path(audio_path).unlink(missing_ok=True)
    
    return {
      "status": "success",
      "transcript": final_transcript,
      "session_id": session_id
    }
  
  except Exception as e:
    for ext in [".wav", ".mp3"]:
      Path(TEMP_DIR / f"{session_id}{ext}").unlink(missing_ok=True)
    # Pass full error detail to client
    raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/transcribe-file")
async def transcribe_file_endpoint(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())[:8]
    filename = file.filename or f"{session_id}_upload"

    try:
        file_path = TEMP_DIR / f"{session_id}_{filename}"
        
        # Save uploaded file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        file_ext = Path(filename).suffix.lower()
        audio_to_send_path = file_path
        
        # --- Audio Pre-processing ---
        if file_ext not in ['.wav', '.mp3', '.m4a', '.webm']:
            try:
                # Load the uploaded file
                audio = AudioSegment.from_file(str(file_path), format=file_ext.strip('.'))
            except Exception as e:
                raise Exception(f"Audio loading failure for {file_ext}: {str(e)}")

            # CRITICAL: Resample and convert to Whisper-compatible format (16kHz, Mono)
            audio = audio.set_frame_rate(16000).set_channels(1)
            
            # Export to WAV
            wav_path = TEMP_DIR / f"{session_id}.wav"
            audio.export(str(wav_path), format="wav")
            
            file_path.unlink() # Delete original file
            audio_to_send_path = wav_path
        
        # 1. Transcribe the audio locally
        raw_transcript = transcribe_whisper(str(audio_to_send_path))
        
        # 2. Apply rule-based post-processing (Capitalization and Segmentation)
        final_transcript = clean_transcript(raw_transcript)
        
        Path(audio_to_send_path).unlink(missing_ok=True) # Delete temp file

        return {
            "status": "success",
            "transcript": final_transcript, # Return the cleaned version
            "session_id": session_id
        }
    
    except Exception as e:
        # Clean up all possible temporary files on error
        for ext in [".wav", ".mp3", ".mp4", ".m4a", ".webm"]:
            Path(TEMP_DIR / f"{session_id}{ext}").unlink(missing_ok=True)
            Path(TEMP_DIR / f"{session_id}_{filename}").unlink(missing_ok=True)
        # Pass the specific error for frontend display
        raise HTTPException(status_code=400, detail=f"File transcription error: {str(e)}")

@app.get("/health")
async def health():
  return {"status": "ok"}

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="localhost", port=7860)