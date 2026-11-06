"""Coqui XTTS-v2 Voice Cloning & Tempo-Fitting Audio Synthesis.

Module Architecture:
--------------------
This module handles Step 3 of the pipeline: cloning vocal identity from a short
reference audio file (e.g. voice_reference.wav) and synthesizing translated dialogue
into foreign languages with Coqui XTTS-v2.

Key Mechanics & Constraints:
----------------------------
1. Apple Silicon CPU Target:
   As established in config.py, XTTS execution on macOS runs on CPU. Apple Silicon
   M-series CPUs deliver exceptionally fast neural inference without the PyTorch
   MPS kernel failures that occur on Apple GPUs.
2. Dynamic Tempo Fitting via FFmpeg atempo:
   Languages differ in syllable density (e.g., Spanish and Japanese dialogue often
   takes 20-30% longer to pronounce than equivalent English phrases).
   If a synthesized segment exceeds its allotted cut window (max_duration), this module
   dynamically speeds up the audio via FFmpeg's phase-vocoder 'atempo' filter (up to 1.5x)
   so the speech fits seamlessly into the video edit without clipping or overlapping
   the subsequent scene.
3. Persistent Segment Caching:
   Each segment is saved to a cache directory. If a dubbing run is restarted or re-mastered,
   already synthesized audio clips are reused immediately.
"""

import os
import shutil
import subprocess
from typing import Dict, List, Optional, Any, Callable

import soundfile as sf

from zerocostvoicefarm.config import (
    DEFAULT_TTS_MODEL,
    HardwareConfig,
    get_ffmpeg_path,
    apply_runtime_patches,
)


class VoiceCloner:
    """Orchestrates Coqui XTTS-v2 cloning and tempo-fitting alignment."""

    def __init__(self, model_name: str = DEFAULT_TTS_MODEL, device: Optional[str] = None):
        apply_runtime_patches()
        self.model_name = model_name
        self.device = device or HardwareConfig.probe().xtts_device
        self._tts = None

    def load_model(self):
        """Lazy-load the XTTS-v2 model checkpoint."""
        if self._tts is None:
            from TTS.api import TTS
            use_gpu = self.device == "cuda"
            self._tts = TTS(self.model_name, gpu=use_gpu)
        return self._tts

    def clone_single(
        self,
        text: str,
        speaker_wav: str,
        language: str,
        output_wav: str,
    ) -> str:
        """Clone voice for a single text utterance and save to output_wav."""
        if not os.path.isfile(speaker_wav):
            raise FileNotFoundError(f"Speaker reference audio not found: {speaker_wav}")

        tts = self.load_model()
        os.makedirs(os.path.dirname(os.path.abspath(output_wav)), exist_ok=True)

        tts.tts_to_file(
            text=text.strip(),
            speaker_wav=speaker_wav,
            language=language.lower(),
            file_path=output_wav,
        )
        return output_wav

    def fit_audio_tempo(
        self,
        input_wav: str,
        output_wav: str,
        max_duration: float,
        max_speed_factor: float = 1.5,
    ) -> float:
        """Adjust playback tempo using FFmpeg atempo if audio exceeds max_duration."""
        ffmpeg_bin = get_ffmpeg_path()
        if not ffmpeg_bin:
            raise RuntimeError("FFmpeg required for tempo adjustment.")

        info = sf.info(input_wav)
        actual_dur = info.duration

        # If audio fits within window, simply copy
        if actual_dur <= max_duration or max_duration <= 0.5:
            shutil.copyfile(input_wav, output_wav)
            return actual_dur

        # Calculate required speedup factor, bounded by max_speed_factor
        speed_factor = min(actual_dur / max_duration, max_speed_factor)
        cmd = [
            ffmpeg_bin, "-y",
            "-i", input_wav,
            "-filter:a", f"atempo={speed_factor:.3f}",
            output_wav,
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            # Fall back to un-stretched audio if atempo fails
            shutil.copyfile(input_wav, output_wav)
            return actual_dur

        return sf.info(output_wav).duration

    def synthesize_segments(
        self,
        segments: List[Dict[str, Any]],
        speaker_wav: str,
        language: str,
        cache_dir: str,
        text_key: str = "target_text",
        max_speed_factor: float = 1.5,
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Synthesize all dialogue segments with caching and tempo fitting."""
        if not os.path.isfile(speaker_wav):
            raise FileNotFoundError(f"Speaker reference file not found: {speaker_wav}")

        tts = self.load_model()
        lang_cache = os.path.join(cache_dir, language.lower())
        os.makedirs(lang_cache, exist_ok=True)

        total_segments = len(segments)
        processed: List[Dict[str, Any]] = []

        for idx, seg in enumerate(segments):
            seg_id = seg.get("id", idx + 1)
            start_time = float(seg["start"])
            end_time = float(seg.get("end", start_time + 1.0))
            max_duration = float(seg.get("max_duration", end_time - start_time + 1.5))

            text = seg.get(text_key) or seg.get(language) or seg.get("text", "")
            text = text.strip()

            raw_out = os.path.join(lang_cache, f"seg_{seg_id:03d}_raw.wav")
            fitted_out = os.path.join(lang_cache, f"seg_{seg_id:03d}.wav")

            # 1. Synthesize if raw or fitted output does not exist
            if not os.path.isfile(fitted_out):
                if not os.path.isfile(raw_out):
                    if text:
                        tts.tts_to_file(
                            text=text,
                            speaker_wav=speaker_wav,
                            language=language.lower(),
                            file_path=raw_out,
                        )
                    else:
                        # Empty text: generate a silent wav matching duration
                        import numpy as np
                        silence = np.zeros((int(48000 * max(0.5, end_time - start_time)),), dtype=np.float32)
                        sf.write(raw_out, silence, 48000)

                # 2. Check duration and adjust tempo
                final_dur = self.fit_audio_tempo(
                    raw_out,
                    fitted_out,
                    max_duration=max_duration,
                    max_speed_factor=max_speed_factor,
                )
            else:
                final_dur = sf.info(fitted_out).duration

            updated_seg = dict(seg)
            updated_seg["audio_path"] = fitted_out
            updated_seg["fitted_duration"] = final_dur
            processed.append(updated_seg)

            if progress_callback:
                progress_callback(idx + 1, total_segments, updated_seg)

        return processed
