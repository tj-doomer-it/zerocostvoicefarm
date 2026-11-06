# 🎙️ zerocostvoicefarm

> **Zero-cloud-cost multilingual video dubbing and voice cloning.**  
> Transcribe, translate, clone your own voice with XTTS-v2, duck background audio, and remux video locally on Apple Silicon and CUDA. Zero subscriptions. Zero cloud bills.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hardware: Apple Silicon](https://img.shields.io/badge/Compute-Apple%20Silicon%20M--Series-orange.svg)]()
[![Cost: $0](https://img.shields.io/badge/Cloud%20Cost-%240.00-brightgreen.svg)]()

---

## ☕ Why I Built This

I make photography vlogs for YouTube ([*Surf the Chaos*](https://www.youtube.com)), walking around Melbourne shooting street photos on 35mm film with vintage cameras (Leicas, Canon rangefinders, Olympus Mju).

When I wanted to dub my videos into Japanese, French, Spanish, and German for people overseas, I looked at commercial tools like ElevenLabs Dubbing Studio. They wanted recurring monthly subscriptions, credit meters, and hundreds of dollars to process 10-minute videos.

That seemed crazy when modern Macs have plenty of compute sitting idle.

So I built **`zerocostvoicefarm`**: a standalone, zero-cost CLI that runs 100% locally on your machine. Give it a video and a 10-second reference clip of your voice, and it handles the entire chain:
1. Listens to the original video and extracts timestamped lines via **OpenAI Whisper**.
2. Translates the dialogue to the target language and calculates cut window constraints.
3. Clones your vocal identity into fluent foreign speech using **Coqui XTTS-v2**.
4. Dynamically speeds up foreign lines with FFmpeg's `atempo` if the translation runs longer than the video cut.
5. Smoothly ducks original background audio (music, city ambience, shutter clicks) to 18% during dialogue.
6. Masters the audio at broadcast True Peak (-1.0 dB) and remuxes the video losslessly.

You get a finished video ready to upload to YouTube with multi-language audio tracks, and it costs exactly zero dollars.

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
 │ 2. Translation & Window Mapping                 │
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

- **Built for Apple Silicon:** Runs XTTS-v2 on Apple Silicon M-series CPU using vectorized Apple Accelerate kernels (~0.4-0.5x real-time speed, rock-solid stability). Bypasses PyTorch MPS kernel limitations that cause GPU tensor NaNs in Coqui TTS.
- **PyTorch 2.6+ Checkpoint Patch:** Ships with automatic monkey-patches for PyTorch 2.6+ unpickling (`weights_only=False`) and macOS torchaudio soundfile backend bindings.
- **Dynamic Cut Pacing (`atempo`):** Different languages have different syllable counts (Spanish and Japanese dialogue often run 25% longer than English). If a translated phrase exceeds its cut window (`max_duration = next_start - current_start`), it automatically accelerates playback cleanly without pitch shift.
- **Preserves Video Ambience:** Doesn't discard original audio. Background music, street noise, and room tone are smoothly ducked down during speech and ramped back up during silent pauses.
- **Lossless Video Remuxing:** Copies the original video stream verbatim (`-c:v copy`), meaning 4K video exports in under a second without quality degradation.

---

## 🌍 Supported Foreign Languages (17 Languages)

Supports all 17 official Coqui XTTS-v2 target languages:

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

### 1. Requirements

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

The flagship command. Takes your video and a 10-30 second WAV sample of your voice, transcribes it, translates it, clones your voice in the target language, ducks the background audio, and muxes everything into a new video:

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
- `--keep-temp`: Keeps intermediate audio stems for inspection or manual DAW mixing.

---

### Command 2: Direct Voice Cloning (`clone`)

Want to generate a quick speech sample in your cloned voice without running a whole video?

```bash
voicefarm clone \
  --ref ./voice_reference.wav \
  --text "よし、春がやってきて、とても晴れて気持ちがいいので、街に出てきました。" \
  --lang ja \
  --out ./japanese_sample.wav
```

---

### Command 3: Whisper Transcription & Subtitle Export (`transcribe`)

Transcribes any video or audio file and writes timestamped JSON segments, SRT subtitles, WebVTT files, and plaintext transcripts:

```bash
voicefarm transcribe ./my_vlog.m4v --model small --out-dir ./subtitles
```

Outputs:
- `./subtitles/my_vlog_segments.json` (Structured timestamps and segment IDs)
- `./subtitles/my_vlog.srt` (SubRip format with millisecond commas)
- `./subtitles/my_vlog.vtt` (WebVTT format for web players)
- `./subtitles/my_vlog.txt` (Full dialogue transcript)

---

### Command 4: Inspect System & Presets (`presets`)

Prints an instant diagnostic report of your local hardware, detected compute targets, and FFmpeg version:

```bash
voicefarm presets
```

---

## 🐍 Python Library API

You can also use `zerocostvoicefarm` as a standard Python library in your own creative tools:

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
