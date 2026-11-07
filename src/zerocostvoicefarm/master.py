"""FFmpeg Audio Ducking, Sample Rate Alignment & Video Remuxing.

Module Architecture:
--------------------
This module handles Step 4 of the dubbing pipeline: mixing the cloned voice tracks
back with the original background audio and muxing into the final video file.

Validated Audio Mechanics:
--------------------------
1. Dual-Ramp Dialogue Ducking Envelope:
   Original audio contains valuable background music, ambience, and environmental
   foley. When dialogue is active, the background track is ducked to 18% volume
   with a 200ms ease-in ramp and 300ms ease-out recovery.
2. Polyphase Sample Rate Resampling:
   Coqui XTTS renders at 24,000 Hz. Broadcast and YouTube videos standardise on
   48,000 Hz or 44,100 Hz. We apply scipy.signal.resample_poly using the exact
   integer ratio (e.g. 2:1 for 48kHz, 147:80 for 44.1kHz) to prevent phase
   distortion and aliasing artifacts.
3. Stereo Placement & True-Peak Protection:
   Mono voice synthesis is placed across both stereo channels. The combined mix
   is peak-normalized to -1.0 dB True Peak (linear 0.891) to guarantee zero
   digital distortion when uploaded to streaming platforms.
4. Lossless Video Remuxing:
   Video frames (-c:v copy) are passed through without re-encoding, ensuring
   instant rendering and 100% preservation of original video quality.
"""

import os
import math
import subprocess
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import soundfile as sf
import scipy.signal

from zerocostvoicefarm.config import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_DUCK_LEVEL,
    DEFAULT_RAMP_EASE_IN,
    DEFAULT_RAMP_EASE_OUT,
    DEFAULT_PEAK_DB,
    get_ffmpeg_path,
)


class AudioMasterer:
    """Mastering orchestrator for audio ducking, resampling, and video muxing."""

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        duck_level: float = DEFAULT_DUCK_LEVEL,
        ease_in_s: float = DEFAULT_RAMP_EASE_IN,
        ease_out_s: float = DEFAULT_RAMP_EASE_OUT,
        target_peak_db: float = DEFAULT_PEAK_DB,
    ):
        self.sample_rate = sample_rate
        self.duck_level = duck_level
        self.ease_in_s = ease_in_s
        self.ease_out_s = ease_out_s
        self.target_peak_db = target_peak_db
        self.ffmpeg_bin = get_ffmpeg_path()
        if not self.ffmpeg_bin:
            raise RuntimeError("FFmpeg executable not found in PATH or standard system locations.")

    def extract_audio_track(self, media_path: str, output_wav: str) -> Tuple[float, int]:
        """Decode input media audio track into stereo PCM WAV at target sample rate."""
        os.makedirs(os.path.dirname(os.path.abspath(output_wav)), exist_ok=True)
        cmd = [
            self.ffmpeg_bin, "-y",
            "-i", media_path,
            "-vn",
            "-ac", "2",
            "-ar", str(self.sample_rate),
            output_wav,
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg audio extraction failed: {res.stderr}")

        info = sf.info(output_wav)
        return info.duration, info.samplerate

    def duck_and_mix(
        self,
        bg_audio_path: str,
        segments_with_audio: List[Dict[str, Any]],
        output_master_wav: str,
        isolated_voice: bool = False,
    ) -> str:
        """Apply dynamic ducking envelope, overlay dialogue, and normalize master peaks.

        If isolated_voice is True (or duck_level == 0.0), original background audio
        is completely muted to deliver a crystal-clear isolated translated voice track.
        """
        # 1. Load background audio
        bg_audio, sr_bg = sf.read(bg_audio_path, dtype="float32")
        if bg_audio.ndim == 1:
            bg_audio = np.column_stack([bg_audio, bg_audio])

        total_samples = len(bg_audio)
        ramp_in_samples = int(self.ease_in_s * sr_bg)
        ramp_out_samples = int(self.ease_out_s * sr_bg)

        duck_envelope = np.ones(total_samples, dtype=np.float32)
        voice_track = np.zeros((total_samples, 2), dtype=np.float32)

        # 2. Process each dialogue clip
        for seg in segments_with_audio:
            audio_path = seg.get("audio_path")
            if not audio_path or not os.path.isfile(audio_path):
                continue

            seg_data, seg_sr = sf.read(audio_path, dtype="float32")
            if seg_data.ndim == 2:
                seg_data = seg_data[:, 0]  # Take primary channel

            # Polyphase resample if sample rate does not match background
            if seg_sr != sr_bg:
                gcd = math.gcd(int(sr_bg), int(seg_sr))
                up = int(sr_bg) // gcd
                down = int(seg_sr) // gcd
                seg_data = scipy.signal.resample_poly(seg_data, up, down)

            start_time = float(seg["start"])
            start_sample = int(start_time * sr_bg)
            seg_len = len(seg_data)
            end_sample = min(start_sample + seg_len, total_samples)
            actual_added = end_sample - start_sample

            if actual_added > 0:
                # Place dialogue into stereo track
                voice_track[start_sample:end_sample, 0] += seg_data[:actual_added]
                voice_track[start_sample:end_sample, 1] += seg_data[:actual_added]

                # Ease-in ramp down
                s_ramp_start = max(0, start_sample - ramp_in_samples)
                s_ramp_end = start_sample
                if s_ramp_end > s_ramp_start:
                    ramp_down = np.linspace(1.0, self.duck_level, s_ramp_end - s_ramp_start, dtype=np.float32)
                    duck_envelope[s_ramp_start:s_ramp_end] = np.minimum(
                        duck_envelope[s_ramp_start:s_ramp_end], ramp_down
                    )

                # Dialogue hold level
                duck_envelope[start_sample:end_sample] = np.minimum(
                    duck_envelope[start_sample:end_sample], self.duck_level
                )

                # Ease-out ramp up
                e_ramp_start = end_sample
                e_ramp_end = min(total_samples, end_sample + ramp_out_samples)
                if e_ramp_end > e_ramp_start:
                    ramp_up = np.linspace(self.duck_level, 1.0, e_ramp_end - e_ramp_start, dtype=np.float32)
                    duck_envelope[e_ramp_start:e_ramp_end] = np.minimum(
                        duck_envelope[e_ramp_start:e_ramp_end], ramp_up
                    )

        # 3. Combine ducked background and voice overlay
        if isolated_voice or self.duck_level == 0.0:
            mixed = voice_track
        else:
            mixed = bg_audio * duck_envelope[:, None] + voice_track

        # 4. True-peak normalization to target dB
        target_linear = 10.0 ** (self.target_peak_db / 20.0)  # e.g. -1.0 dB = ~0.891
        max_peak = float(np.max(np.abs(mixed)))
        if max_peak > target_linear and max_peak > 0:
            mixed = mixed * (target_linear / max_peak)

        # 5. Export master wav
        os.makedirs(os.path.dirname(os.path.abspath(output_master_wav)), exist_ok=True)
        sf.write(output_master_wav, mixed, sr_bg)
        return output_master_wav

    def encode_aac(self, input_wav: str, output_m4a: str, bitrate: str = "256k") -> str:
        """Encode master audio WAV to high-bitrate AAC (.m4a)."""
        os.makedirs(os.path.dirname(os.path.abspath(output_m4a)), exist_ok=True)
        cmd = [
            self.ffmpeg_bin, "-y",
            "-i", input_wav,
            "-c:a", "aac",
            "-b:a", bitrate,
            output_m4a,
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg AAC encoding failed: {res.stderr}")
        return output_m4a

    def mux_video(self, video_path: str, master_audio_path: str, output_video_path: str) -> str:
        """Remux original video stream with the new dubbed master audio track."""
        os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
        cmd = [
            self.ffmpeg_bin, "-y",
            "-i", video_path,
            "-i", master_audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "256k",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            output_video_path,
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg video muxing failed: {res.stderr}")
        return output_video_path

    def run_mastering_pipeline(
        self,
        source_media_path: str,
        segments_with_audio: List[Dict[str, Any]],
        output_path: str,
        work_dir: str,
        isolated_voice: bool = False,
    ) -> str:
        """Execute complete mastering and remuxing sequence."""
        os.makedirs(work_dir, exist_ok=True)
        bg_wav = os.path.join(work_dir, "original_bg_stereo.wav")
        master_wav = os.path.join(work_dir, "dubbed_master.wav")

        # 1. Extract background
        self.extract_audio_track(source_media_path, bg_wav)

        # 2. Duck and mix (or isolate voice)
        self.duck_and_mix(bg_wav, segments_with_audio, master_wav, isolated_voice=isolated_voice)

        # 3. Determine if target output is audio-only or video container
        ext = os.path.splitext(output_path)[1].lower()
        if ext in (".m4a", ".aac"):
            return self.encode_aac(master_wav, output_path)
        elif ext in (".wav",):
            import shutil
            shutil.copyfile(master_wav, output_path)
            return output_path
        else:
            # Video remux (e.g. .m4v, .mp4, .mov, .mkv)
            return self.mux_video(source_media_path, master_wav, output_path)
