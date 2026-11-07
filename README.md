# 🎙️ zerocostvoicefarm

> **The headless, zero-cost video dubbing engine for AI agents and creator pipelines.**  
> Autonomous multilingual video localization: Whisper + XTTS-v2 + dynamic cut pacing + broadcast audio ducking. Runs 100% locally on Apple Silicon and CUDA with native MCP server support. Zero subscriptions. Zero cloud bills.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Compute: Local](https://img.shields.io/badge/Compute-Apple%20Silicon%20%2F%20CUDA-orange.svg)]()
[![Cost: $0](https://img.shields.io/badge/Cloud%20Cost-%240.00-brightgreen.svg)]()
[![MCP: Enabled](https://img.shields.io/badge/MCP-Native%20Agent%20Server-purple.svg)]()

---

## ☕ Why I Built This

I make photography vlogs for YouTube ([*Surf the Chaos*](https://www.youtube.com)), walking around Melbourne shooting street photos on 35mm film with vintage cameras (Leicas, Canon rangefinders, Olympus Mju).

When I looked into dubbing my videos into Japanese, French, Spanish, and German for international viewers, commercial dubbing tools (like ElevenLabs Dubbing Studio) wanted recurring subscriptions, credit meters, and hundreds of dollars to process a handful of 10-minute videos.

That seemed crazy when modern laptops have massive compute sitting idle.

Most open-source voice tools focus on desktop GUI dictation or spitting out disconnected `.wav` files. But for creators and autonomous agents, raw audio isn't enough - you need the **entire video localization loop**: timing the cuts, translating the dialogue, pacing foreign syllables so they don't overrun scene boundaries, and preserving the background room tone and music.

So I built **`zerocostvoicefarm`**: a headless, zero-cost engine that runs 100% locally on Apple Silicon and CUDA. You give it a video and a 10-second reference clip of your voice, and it delivers a ready-to-upload master video without human GUI intervention.

---

## 🏆 The Three Core Pillars

### 1. 🎬 Autonomous Video Dubbing
Not just a text-to-speech generator or dictation app. `zerocostvoicefarm` executes the entire localization pipeline end-to-end:
```
Raw Video In ──► Whisper Transcribe ──► Translate & Window Map ──► XTTS-v2 Clone ──► atempo Pacing ──► Audio Ducking ──► Lossless Remux ──► Master Video Out
```
One command in your terminal, or one tool call from your coding agent, outputs a finished video with multilingual audio tracks ready for YouTube.

### 2. ⏱️ Video Cut-Window Pacing
Different languages have fundamentally different syllable densities. Spanish and Japanese dialogue often run 25-30% longer than equivalent English sentences. A dumb TTS model overlaps adjacent cuts or throws video pacing completely out of sync.

`zerocostvoicefarm` dynamically calculates the maximum allowable window between cuts (`max_duration = next_start - current_start`). If translated dialogue exceeds that slot, it accelerates playback cleanly using FFmpeg's phase-vocoder `atempo` filter (up to 1.5x) without pitch distortion.

### 3. 🎧 Background Ambience Preservation
Commercial tools often wipe the original audio track, replacing natural video ambience with dead silence and isolated vocals.

`zerocostvoicefarm` extracts and preserves your background music, street Foley, shutter clicks, and room tone. It applies a dual-ramp audio ducking envelope that smoothly attenuates original audio to 18% volume during dialogue (200ms ease-in) and ramps it back up during natural pauses (300ms ease-out), normalized to -1.0 dB True Peak.

*If you don't want this, an isolated voice track (`--isolated-voice` or `--duck-level 0.0`) will give you crystal-clear translated audio with zero background bleed.*

---

## 🤖 Native MCP Server for AI Agents

`zerocostvoicefarm` ships with a built-in **Model Context Protocol (MCP)** server. Coding agents like **Claude Code**, **Cursor**, **Cline**, and **Antigravity** can invoke video dubbing and voice cloning directly as native tools.

### 1. Add to your Agent Config

Add `voicefarm` to your agent's MCP configuration (`~/.claude/settings.json`, `claude_desktop_config.json`, or Cursor MCP settings):

```json
{
  "mcpServers": {
    "voicefarm": {
      "command": "voicefarm",
      "args": ["mcp"]
    }
  }
}
```

### 2. Available Agent Tools

| Tool Name | Parameters | Description |
| :--- | :--- | :--- |
| `voicefarm_dub` | `video_path`, `target_lang`, `speaker_ref`, `output_path`, `translations`, `model`, `duck_level`, `isolated_voice` | Runs the full autonomous dubbing and video remuxing pipeline. |
| `voicefarm_clone` | `text`, `speaker_ref`, `language`, `output_wav` | Clones vocal identity from reference audio and generates foreign speech. |
| `voicefarm_transcribe` | `media_path`, `model`, `out_dir` | Transcribes video/audio with Whisper and writes JSON, SRT, VTT, and TXT files. |
| `voicefarm_presets` | None | Returns supported languages and diagnostic local hardware acceleration status. |

---

## ⚡ Architecture Flow

```
                Source Video (.m4v / .mp4)
                           │
                           ▼
 ┌──────────────────────────────────────────────────┐
 │ 1. Whisper Transcription (CPU / MPS / CUDA)     │
 │    Extracts dialogue segments with timestamps   │
 └─────────────────────────┬────────────────────────┘
                           │
                           ▼
 ┌──────────────────────────────────────────────────┐
 │ 2. Translation & Cut-Window Mapping             │
 │    Zero-cost NMT + calculates cut constraints   │
 └─────────────────────────┬────────────────────────┘
                           │
                           ▼
 ┌──────────────────────────────────────────────────┐
 │ 3. XTTS-v2 Voice Cloning & Tempo Fitting        │
 │    Clones vocal timbre from voice_reference.wav │
 │    Paces audio via FFmpeg atempo (up to 1.5x)   │
 └─────────────────────────┬────────────────────────┘
                           │
                           ▼
 ┌──────────────────────────────────────────────────┐
 │ 4. Audio Ducking, Mastering & Video Remux       │
 │    Ducks background to 18% (200ms in / 300ms out)│
 │    Polyphase resampling: 24 kHz -> 48 kHz       │
 │    Lossless video bitstream remux (-c:v copy)   │
 └─────────────────────────┬────────────────────────┘
                           │
                           ▼
             Dubbed Master Video (.m4v / .mp4)
```

---

## 🛠️ Key Engineering Features

- **Apple Silicon Vectorized CPU:** Runs XTTS-v2 on Apple Silicon M-series CPU using Apple Accelerate vectorized kernels (~0.4-0.5x real-time speed, rock-solid stability). Bypasses PyTorch MPS kernel limitations that cause GPU tensor NaNs in Coqui TTS.
- **PyTorch 2.6+ Checkpoint Patch:** Built-in monkey-patches for PyTorch 2.6+ unpickling (`weights_only=False`) and macOS torchaudio soundfile backend bindings.
- **Lossless Video Remuxing:** Copies the original video bitstream verbatim (`-c:v copy`), allowing 4K video exports in under a second without re-encoding loss.

---

## 🌍 Supported Foreign Languages (17 Languages)

| Code | Language | Code | Language | Code | Language |
| :---: | :--- | :---: | :--- | :---: | :--- |
| `ja` | 🇯🇵 Japanese | `fr` | 🇫🇷 French | `es` | 🇪🇸 Spanish |
| `de` | 🇩🇪 German | `it` | 🇮🇹 Italian | `ko` | 🇰🇷 Korean |
| `pt` | 🇵🇹 Portuguese | `pl` | 🇵🇱 Polish | `tr` | 🇹🇷 Turkish |
| `ru` | 🇷🇺 Russian | `nl` | 🇳🇱 Dutch | `cs` | 🇨🇿 Czech |
| `ar` | 🇸🇦 Arabic | `zh-cn`| 🇨🇳 Chinese (Simp.) | `hu` | 🇭🇺 Hungarian |
| `hi` | 🇮🇳 Hindi | `en` | 🇬🇧 English | | |

---

## 🚀 Quickstart

### 1. Prerequisites

- macOS (Apple Silicon recommended) or Linux with CUDA.
- **FFmpeg:** `brew install ffmpeg`
- **Python:** 3.10 or 3.11.

### 2. Install Package

```bash
git clone https://github.com/tj-doomer-it/zerocostvoicefarm.git ~/zerocostvoicefarm
cd ~/zerocostvoicefarm
pip install -e .
```

Verify your hardware compute routing:

```bash
voicefarm presets
```

---

## 💻 CLI Usage

### Command 1: End-to-End Video Dubbing (`dub`)

Takes your video and a 10-30 second WAV sample of your voice, transcribes it, translates it, clones your voice in the target language, ducks the background audio, and muxes everything into a new video:

```bash
voicefarm dub ./my_vlog.m4v \
  --target-lang ja \
  --ref ./voice_reference.wav \
  --out ./my_vlog_japanese.m4v
```

#### Handy Options:
- `--translations ./translated_script.json`: Supply pre-translated lines (e.g. reviewed by a native speaker or translated by an LLM) to bypass auto-translation.
- `--model small`: Whisper model size (`tiny`, `base`, `small`, `medium`, `large`).
- `--duck-level 0.18`: Background audio volume during dialogue (0.18 = 18%).
- `--isolated-voice`: Completely mutes background audio to output an isolated translated voice track.
- `--keep-temp`: Keeps intermediate audio stems for inspection or manual DAW mixing.

---

### Command 2: Direct Voice Cloning (`clone`)

Generate a quick speech sample in your cloned voice without running a whole video:

```bash
voicefarm clone \
  --ref ./voice_reference.wav \
  --text "よし、春がやってきて、とても晴れて気持ちがいいので、街に出てきました。" \
  --lang ja \
  --out ./japanese_sample.wav
```

---

### Command 3: Whisper Transcription & Subtitles (`transcribe`)

Transcribes any video or audio file and writes timestamped JSON segments, SRT subtitles, WebVTT files, and plaintext transcripts:

```bash
voicefarm transcribe ./my_vlog.m4v --model small --out-dir ./subtitles
```

---

### Command 4: Model Context Protocol Server (`mcp`)

Start the stdio MCP server for agent integration:

```bash
voicefarm mcp
```

---

### Command 5: Inspect System & Presets (`presets`)

Prints an instant diagnostic report of your local hardware, detected compute targets, and FFmpeg version:

```bash
voicefarm presets
```

---

## 🐍 Python Library API

Use `zerocostvoicefarm` directly in your own automated scripts and video tools:

```python
from zerocostvoicefarm import (
    Transcriber,
    TranslationMapper,
    VoiceCloner,
    AudioMasterer,
)

# 1. Transcribe dialogue
transcriber = Transcriber(model_name="small")
transcript = transcriber.transcribe("street_photos.m4v")

# 2. Map timing windows & translate to French
translated = TranslationMapper.translate_segments(
    transcript["segments"],
    target_lang="fr",
)

# 3. Clone voice with dynamic tempo fitting
cloner = VoiceCloner()
segs_with_audio = cloner.synthesize_segments(
    translated,
    speaker_wav="voice_ref.wav",
    language="fr",
    cache_dir="/tmp/dub_cache",
)

# 4. Master audio track and remux video
masterer = AudioMasterer(duck_level=0.18)
masterer.run_mastering_pipeline(
    source_media_path="street_photos.m4v",
    segments_with_audio=segs_with_audio,
    output_path="street_photos_french.m4v",
    work_dir="/tmp/dub_cache",
)
```

---

## 📜 License

MIT License. Built for creators, photographers, and independent filmmakers.
