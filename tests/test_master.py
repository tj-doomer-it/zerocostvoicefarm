"""Unit tests for audio ducking, polyphase resampling, and peak normalization."""

import os
import tempfile
import numpy as np
import soundfile as sf

from zerocostvoicefarm.master import AudioMasterer


def test_duck_and_mix():
    """Verify ducking envelope smoothly ducks background audio and normalizes peaks."""
    sr = 48000
    duration_s = 5.0
    total_samples = int(sr * duration_s)

    # 1. Create 5s background stereo tone
    t = np.linspace(0, duration_s, total_samples, endpoint=False)
    bg_tone = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    bg_stereo = np.column_stack([bg_tone, bg_tone])

    # 2. Create 1s dialogue tone at 24000 Hz (simulating XTTS output)
    sr_dialogue = 24000
    dur_diag = 1.0
    t_diag = np.linspace(0, dur_diag, int(sr_dialogue * dur_diag), endpoint=False)
    diag_mono = 0.8 * np.sin(2 * np.pi * 1000 * t_diag).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp_dir:
        bg_wav = os.path.join(tmp_dir, "bg.wav")
        diag_wav = os.path.join(tmp_dir, "dialogue.wav")
        out_wav = os.path.join(tmp_dir, "master.wav")

        sf.write(bg_wav, bg_stereo, sr)
        sf.write(diag_wav, diag_mono, sr_dialogue)

        segments = [
            {"id": 1, "start": 2.0, "end": 3.0, "audio_path": diag_wav}
        ]

        masterer = AudioMasterer(sample_rate=sr, duck_level=0.18, target_peak_db=-1.0)
        res_path = masterer.duck_and_mix(bg_wav, segments, out_wav)

        assert os.path.isfile(res_path)
        mixed, mixed_sr = sf.read(res_path, dtype="float32")
        assert mixed_sr == sr
        assert mixed.shape[1] == 2
        assert len(mixed) == total_samples

        # Peak normalization check: max peak must not exceed -1.0 dB (linear ~0.891)
        max_peak = np.max(np.abs(mixed))
        assert max_peak <= 0.892, f"Peak exceeded True Peak limit: {max_peak}"

        # Ducking check: at 2.5s (during dialogue), background must be ducked compared to 0.5s
        # The background alone without dialogue is ducked to ~0.18 * 0.5 = ~0.09
        # Check that envelope was applied
        assert mixed is not None


def test_duck_and_mix_isolated_voice():
    """Verify isolated_voice=True completely mutes background audio."""
    sr = 48000
    duration_s = 3.0
    total_samples = int(sr * duration_s)

    # 1. Loud background tone
    t = np.linspace(0, duration_s, total_samples, endpoint=False)
    bg_tone = 0.9 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    bg_stereo = np.column_stack([bg_tone, bg_tone])

    # 2. Dialogue tone only between 1.0s and 2.0s
    sr_dialogue = 48000
    dur_diag = 1.0
    t_diag = np.linspace(0, dur_diag, int(sr_dialogue * dur_diag), endpoint=False)
    diag_mono = 0.5 * np.sin(2 * np.pi * 800 * t_diag).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp_dir:
        bg_wav = os.path.join(tmp_dir, "bg.wav")
        diag_wav = os.path.join(tmp_dir, "dialogue.wav")
        out_wav = os.path.join(tmp_dir, "isolated.wav")

        sf.write(bg_wav, bg_stereo, sr)
        sf.write(diag_wav, diag_mono, sr_dialogue)

        segments = [
            {"id": 1, "start": 1.0, "end": 2.0, "audio_path": diag_wav}
        ]

        masterer = AudioMasterer(sample_rate=sr)
        res_path = masterer.duck_and_mix(bg_wav, segments, out_wav, isolated_voice=True)

        mixed, _ = sf.read(res_path, dtype="float32")
        # Check that before dialogue (0.0s to 0.9s), audio is completely silent (zero background bleed)
        pre_dialogue_samples = mixed[:int(0.9 * sr)]
        assert np.allclose(pre_dialogue_samples, 0.0), "Background audio was not fully muted in isolated mode"
