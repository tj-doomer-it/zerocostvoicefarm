"""zerocostvoicefarm: Zero-cloud-cost multilingual video dubbing and voice cloning CLI."""

__version__ = "0.1.0"

from zerocostvoicefarm.config import (
    SUPPORTED_LANGUAGES,
    DEFAULT_WHISPER_MODEL,
    DEFAULT_TTS_MODEL,
    HardwareConfig,
    apply_runtime_patches,
)
from zerocostvoicefarm.transcribe import Transcriber
from zerocostvoicefarm.translate import TranslationMapper
from zerocostvoicefarm.clone import VoiceCloner
from zerocostvoicefarm.master import AudioMasterer
from zerocostvoicefarm.mcp import create_mcp_server, run_mcp_server

__all__ = [
    "__version__",
    "SUPPORTED_LANGUAGES",
    "DEFAULT_WHISPER_MODEL",
    "DEFAULT_TTS_MODEL",
    "HardwareConfig",
    "apply_runtime_patches",
    "Transcriber",
    "TranslationMapper",
    "VoiceCloner",
    "AudioMasterer",
    "create_mcp_server",
    "run_mcp_server",
]
