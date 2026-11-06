"""Unit tests for config and hardware probe."""

from zerocostvoicefarm.config import (
    SUPPORTED_LANGUAGES,
    DEFAULT_WHISPER_MODEL,
    DEFAULT_TTS_MODEL,
    HardwareConfig,
    HardwareReport,
    get_ffmpeg_path,
)


def test_supported_languages_coverage():
    """Verify all 16 required XTTS-v2 languages plus presets are defined."""
    required = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko"]
    for code in required:
        assert code in SUPPORTED_LANGUAGES, f"Missing required language code: {code}"
        assert len(SUPPORTED_LANGUAGES[code]) > 0


def test_hardware_probe():
    """Verify hardware probe detects host environment without exceptions."""
    report = HardwareConfig.probe()
    assert isinstance(report, HardwareReport)
    assert report.cpu_count >= 1
    assert report.whisper_device in ("cpu", "mps", "cuda")
    assert report.xtts_device in ("cpu", "cuda")


def test_ffmpeg_detection():
    """Verify FFmpeg path resolution returns executable or None."""
    ffmpeg = get_ffmpeg_path()
    assert ffmpeg is not None
    assert "ffmpeg" in ffmpeg
