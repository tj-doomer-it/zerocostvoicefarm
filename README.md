# zerocostvoicefarm

Standalone, zero-cloud-cost CLI and Python package for local multilingual video dubbing and voice cloning on Apple Silicon and CUDA hardware.

Built to replace recurring subscriptions to proprietary dubbing services (such as ElevenLabs Dubbing Studio), **`zerocostvoicefarm`** extracts audio from source video or voice clips, performs timestamped speech-to-text with OpenAI Whisper, translates dialogue across 17 languages, clones the speaker's vocal timbre using Coqui XTTS-v2, and masters the final dialogue back into the video with dynamic audio ducking and lossless stream remuxing.

---

## Architecture & Workflow

```
Source Video (.m4v / .mp4)
       │
       ▼
[1. Transcriber] ──► OpenAI Whisper (CPU / MPS / CUDA)
       │              └─ Timestamped dialogue segments (.json, .srt, .vtt)
       ▼
[2. Translation]  ──► Zero-cost Neural Machine Translation (GTX / Local JSON)
       │              └─ Calculates cut window bounds (max_duration)
       ▼
[3. Voice Cloner] ──► Coqui XTTS-v2 (Cloned from reference .wav)
       │              └─ Dynamic tempo adjustment (FFmpeg atempo up to 1.5x)
       ▼
[4. Audio Master] ──► Background Audio Ducking (18% level, 200ms in / 300ms out)
       │              ├─ Polyphase Resampling (24 kHz -> 48 kHz standard)
       │              ├─ Stereo placement & True-Peak normalization (-1.0 dB)
       │              └─ Lossless video remux (-c:v copy)
       ▼
Dubbed Master Video (.m4v / .mp4)
```

---

## Key Features & Engineering Design

- **100% Zero Cloud Cost:** Runs entirely on local compute. No ElevenLabs, OpenAI, or Google Cloud API subscriptions required.
- **Apple Silicon Hardware Acceleration:** Automatically routes OpenAI Whisper to CPU/MPS and Coqui XTTS-v2 to CPU. On Apple M-series chips, CPU execution leverages vectorized Apple Accelerate kernels for stable, fast (~0.2-0.5x real-time) synthesis without GPU MPS kernel crashes.
- **PyTorch 2.6+ Compatibility:** Ships with built-in runtime patches for PyTorch 2.6+ `weights_only=False` checkpoint unpickling and macOS torchaudio soundfile backend bindings.
- **Dynamic Cut Pacing (FFmpeg atempo):** Automatically calculates segment boundaries (`max_duration = next_start - current_start`). If translated foreign dialogue exceeds its cut window, it accelerates playback cleanly (up to 1.5x) without pitch distortion.
- **Broadcast Audio Mastering:** Ducks original background audio down to 18% during dialogue, places mono voice across stereo, normalizes True Peak to -1.0 dB, and encodes 256 kbps AAC tracks.
- **Lossless Video Remuxing:** Replaces original audio while copying video bitstreams verbatim (`-c:v copy`), preserving 100% video quality with near-instant export.

---

## Supported Foreign Languages

Supports all 17 official Coqui XTTS-v2 languages:

| Code | Language | Code | Language |
| :--- | :--- | :--- | :--- |
| `en` | English | `es` | Spanish |
| `fr` | French | `de` | German |
| `it` | Italian | `pt` | Portuguese |
| `pl` | Polish | `tr` | Turkish |
| `ru` | Russian | `nl` | Dutch |
| `cs` | Czech | `ar` | Arabic |
| `zh-cn` | Chinese (Simplified) | `ja` | Japanese |
| `hu` | Hungarian | `ko` | Korean |
| `hi` | Hindi | | |

---

## Prerequisites & Installation

### 1. System Requirements

- **FFmpeg:** Installed and available on your `PATH`.
  ```bash
  brew install ffmpeg
  ```
- **Python:** Python 3.10 or 3.11 recommended.

### 2. Install Package

Clone the repository and install in editable mode:

```bash
git clone https://github.com/tj-doomer-it/zerocostvoicefarm.git ~/zerocostvoicefarm
cd ~/zerocostvoicefarm
pip install -e .
```

This registers the `voicefarm` executable in your active environment.

---

## CLI Usage Guide

The unified `voicefarm` CLI exposes four commands:

### 1. Inspect Hardware & Supported Languages (`presets`)

Detects host CPU architecture, RAM, PyTorch accelerator availability, and XTTS model cache status:

```bash
voicefarm presets
```

### 2. Zero-Cost Voice Cloning (`clone`)

Clone vocal timbre from a short reference WAV (10-30 seconds of clean speech) and render a custom phrase:

```bash
voicefarm clone \
  --ref ./data/voice_reference.wav \
  --text "Bonjour, je teste le clonage vocal gratuit sur Apple Silicon." \
  --lang fr \
  --out ./output/french_sample.wav
```

### 3. Speech-to-Text Transcription (`transcribe`)

Extract audio from a video or audio file, transcribe with Whisper, and output structured JSON, SRT, VTT, and TXT:

```bash
voicefarm transcribe /path/to/video.m4v --model small --out-dir ./transcripts
```

Outputs:
- `./transcripts/<name>_segments.json`: Structured segment boundaries with timestamps.
- `./transcripts/<name>.srt`: SubRip subtitle track.
- `./transcripts/<name>.vtt`: WebVTT subtitle track.
- `./transcripts/<name>.txt`: Plaintext transcript.

### 4. End-to-End Multilingual Video Dubbing (`dub`)

Runs the full pipeline (transcribe -> translate -> clone -> duck audio -> remux video):

```bash
voicefarm dub /path/to/video.m4v \
  --target-lang fr \
  --ref ./data/voice_reference.wav \
  --out ./output/video_french_dubbed.m4v
```

#### Optional Flags:
- `--translations <path.json>`: Bypass auto-translation by supplying a pre-translated JSON segment file.
- `--model <small|medium|base>`: Choose Whisper model size (default: `small`).
- `--duck-level <float>`: Background audio volume multiplier during speech (default: `0.18`).
- `--work-dir <path>`: Specify custom workspace for caching audio stems.
- `--keep-temp`: Keep intermediate segment files for manual inspection.

---

## Python API Usage

You can also import and compose `zerocostvoicefarm` components directly in Python:

```python
from zerocostvoicefarm import (
    Transcriber,
    TranslationMapper,
    VoiceCloner,
    AudioMasterer,
)

# 1. Transcribe
transcriber = Transcriber(model_name="small")
result = transcriber.transcribe("input_video.m4v")

# 2. Map timing and translate
translated_segs = TranslationMapper.translate_segments(
    result["segments"],
    target_lang="es",
)

# 3. Clone and tempo-fit
cloner = VoiceCloner()
segs_with_audio = cloner.synthesize_segments(
    translated_segs,
    speaker_wav="voice_reference.wav",
    language="es",
    cache_dir="/tmp/dub_cache",
)

# 4. Master and remux
masterer = AudioMasterer(duck_level=0.18)
masterer.run_mastering_pipeline(
    source_media_path="input_video.m4v",
    segments_with_audio=segs_with_audio,
    output_path="video_spanish_dubbed.m4v",
    work_dir="/tmp/dub_cache",
)
```

---

## License

MIT License.
