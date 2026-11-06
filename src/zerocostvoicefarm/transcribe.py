"""Whisper Audio Extraction & Timestamped Transcription for zerocostvoicefarm.

Module Architecture:
--------------------
This module handles Step 1 of the voice dubbing pipeline: extracting audio from
source video/audio containers, executing local speech-to-text with OpenAI Whisper,
and formatting timestamped dialogue segments into standard subtitle formats
(.srt, .vtt) and structured JSON.

Design Decisions:
-----------------
1. Dedicated 16 kHz Mono Pre-extraction:
   While Whisper can decode various formats via internal FFmpeg subprocess calls,
   pre-extracting clean 16 kHz mono PCM WAV via FFmpeg guarantees consistent
   channel downmixing and sample-rate resampling before feeding the PyTorch model.
2. Structured Segment Representation:
   Whisper returns floating-point start/end offsets. We normalize these into
   1-indexed segments with explicit duration fields, which downstream timing
   and tempo-adjustment modules rely upon.
3. fp16 Guard:
   Whisper crashes or throws warnings on CPU/MPS when fp16=True is used. We enforce
   fp16=False whenever running on CPU or Apple Silicon MPS.
"""

import os
import json
import subprocess
from typing import Dict, List, Optional, Any

from zerocostvoicefarm.config import (
    DEFAULT_WHISPER_MODEL,
    HardwareConfig,
    get_ffmpeg_path,
    apply_runtime_patches,
)


def format_timestamp(seconds: float, decimal_sep: str = ",") -> str:
    """Format floating point seconds into HH:MM:SS,mmm (SRT) or HH:MM:SS.mmm (VTT)."""
    total_seconds = max(0.0, float(seconds))
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    secs = int(total_seconds % 60)
    millis = int(round((total_seconds - int(total_seconds)) * 1000))
    if millis >= 1000:
        millis = 999
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal_sep}{millis:03d}"


class Transcriber:
    """High-level interface for local speech-to-text with Whisper."""

    def __init__(self, model_name: str = DEFAULT_WHISPER_MODEL, device: Optional[str] = None):
        apply_runtime_patches()
        self.model_name = model_name
        self.device = device or HardwareConfig.probe().whisper_device
        self._model = None

    def load_model(self):
        """Lazy-load the Whisper model checkpoint to save memory until needed."""
        if self._model is None:
            import whisper
            self._model = whisper.load_model(self.model_name, device=self.device)
        return self._model

    def extract_audio(self, media_path: str, output_wav: str) -> str:
        """Extract a 16 kHz mono WAV file from any video or audio format."""
        ffmpeg_bin = get_ffmpeg_path()
        if not ffmpeg_bin:
            raise RuntimeError("FFmpeg executable not found. Install via 'brew install ffmpeg' or system package manager.")

        os.makedirs(os.path.dirname(os.path.abspath(output_wav)), exist_ok=True)
        cmd = [
            ffmpeg_bin, "-y",
            "-i", media_path,
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            output_wav,
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg audio extraction failed: {res.stderr}")
        return output_wav

    def transcribe(
        self,
        media_path: str,
        initial_prompt: Optional[str] = None,
        verbose: bool = False,
        work_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Transcribe video or audio media and return structured segments.

        Returns a dictionary:
            - text: Full concatenated transcript.
            - language: Detected source language code.
            - segments: List of normalized segment dictionaries:
                {id: int, start: float, end: float, duration: float, text: str}
        """
        model = self.load_model()

        # Pre-extract to 16kHz mono wav in work_dir if media is not already 16k wav
        target_audio = media_path
        temp_extracted = None

        if work_dir:
            temp_extracted = os.path.join(work_dir, "extracted_whisper_16k.wav")
            try:
                target_audio = self.extract_audio(media_path, temp_extracted)
            except Exception:
                # If extraction fails (e.g. ffmpeg quirk), fallback to passing media directly to Whisper
                target_audio = media_path

        # Determine fp16 suitability: CPU and MPS must use fp16=False
        use_fp16 = self.device.startswith("cuda")

        kwargs = {
            "verbose": verbose,
            "fp16": use_fp16,
        }
        if initial_prompt:
            kwargs["initial_prompt"] = initial_prompt

        raw_result = model.transcribe(target_audio, **kwargs)

        # Normalize segments
        normalized_segments: List[Dict[str, Any]] = []
        for i, s in enumerate(raw_result.get("segments", [])):
            start_s = round(float(s["start"]), 3)
            end_s = round(float(s["end"]), 3)
            duration_s = max(0.05, round(end_s - start_s, 3))
            text = s.get("text", "").strip()
            if not text:
                continue

            normalized_segments.append({
                "id": i + 1,
                "start": start_s,
                "end": end_s,
                "duration": duration_s,
                "text": text,
            })

        return {
            "text": raw_result.get("text", "").strip(),
            "language": raw_result.get("language", "en"),
            "segments": normalized_segments,
        }

    @staticmethod
    def export_transcripts(
        transcription_result: Dict[str, Any],
        output_dir: str,
        base_name: str,
    ) -> Dict[str, str]:
        """Write .json, .srt, .vtt, and .txt files to output_dir."""
        os.makedirs(output_dir, exist_ok=True)
        segments = transcription_result.get("segments", [])
        full_text = transcription_result.get("text", "")

        out_paths: Dict[str, str] = {}

        # 1. Structured JSON
        json_path = os.path.join(output_dir, f"{base_name}_segments.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        out_paths["json"] = json_path

        # 2. SRT Format
        srt_path = os.path.join(output_dir, f"{base_name}.srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            for seg in segments:
                t_start = format_timestamp(seg["start"], decimal_sep=",")
                t_end = format_timestamp(seg["end"], decimal_sep=",")
                f.write(f"{seg['id']}\n{t_start} --> {t_end}\n{seg['text']}\n\n")
        out_paths["srt"] = srt_path

        # 3. WebVTT Format
        vtt_path = os.path.join(output_dir, f"{base_name}.vtt")
        with open(vtt_path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for seg in segments:
                t_start = format_timestamp(seg["start"], decimal_sep=".")
                t_end = format_timestamp(seg["end"], decimal_sep=".")
                f.write(f"{seg['id']}\n{t_start} --> {t_end}\n{seg['text']}\n\n")
        out_paths["vtt"] = vtt_path

        # 4. Plain text
        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_text + "\n")
        out_paths["txt"] = txt_path

        return out_paths
