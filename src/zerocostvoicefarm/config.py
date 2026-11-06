"""Configuration and Hardware Detection for zerocostvoicefarm.

Module Architecture:
--------------------
This module serves as the foundational hardware probe, runtime patch manager,
and preset configuration repository for zerocostvoicefarm. It is imported by
all pipeline modules (transcribe, clone, translate, master, cli) before heavy
dependencies are loaded.

Key Invariants & Decisions:
---------------------------
1. PyTorch 2.6+ Checkpoint Patch:
   PyTorch 2.6+ introduced a breaking security default (weights_only=True) in
   torch.load. Coqui XTTS checkpoints rely on custom class pickling. We patch
   torch.load at runtime to enforce weights_only=False for model loading.
2. Torchaudio Apple Silicon Patch:
   Torchaudio builds on macOS frequently lack Sox/SoundFile C++ backend bindings.
   We monkey-patch torchaudio.load with soundfile to ensure robust, crash-free
   audio decoding across all macOS versions.
3. Hardware Target Separation:
   - Whisper: Fully supports Apple Silicon MPS, CUDA, and CPU.
   - Coqui XTTS-v2: Coqui TTS does not natively support PyTorch MPS device
     mapping; attempting to pass MPS triggers unsupported kernel errors or
     silent NaNs. On Apple Silicon (M1/M2/M3/M4), CPU execution with Accelerate
     vectorization is fast (~0.2-0.5x real-time) and 100% crash-free. We therefore
     route XTTS to CPU on Apple Silicon and CUDA on Nvidia hosts.
"""

import os
import sys
import shutil
import platform
import subprocess
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# XTTS-v2 officially supported languages
SUPPORTED_LANGUAGES: Dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "pl": "Polish",
    "tr": "Turkish",
    "ru": "Russian",
    "nl": "Dutch",
    "cs": "Czech",
    "ar": "Arabic",
    "zh-cn": "Chinese (Simplified)",
    "ja": "Japanese",
    "hu": "Hungarian",
    "ko": "Korean",
    "hi": "Hindi",
}

DEFAULT_WHISPER_MODEL: str = "small"
DEFAULT_TTS_MODEL: str = "tts_models/multilingual/multi-dataset/xtts_v2"
DEFAULT_SAMPLE_RATE: int = 48000
DEFAULT_DUCK_LEVEL: float = 0.18
DEFAULT_RAMP_EASE_IN: float = 0.20
DEFAULT_RAMP_EASE_OUT: float = 0.30
DEFAULT_PEAK_DB: float = -1.0


def apply_runtime_patches() -> None:
    """Apply critical PyTorch and torchaudio monkey-patches required for XTTS.

    Why this approach is mandatory:
    - Coqui TOS agreement must be set before TTS imports or it blocks on stdin.
    - PyTorch 2.6+ raises UnsupportedPickleException without weights_only=False.
    - Torchaudio without sox_io crashes when loading audio on macOS.
    """
    os.environ["COQUI_TOS_AGREED"] = "1"

    # 1. PyTorch weights_only patch
    try:
        import torch
        if not getattr(torch, "_zerocost_patched", False):
            _orig_torch_load = torch.load

            def _patched_torch_load(*args, **kwargs):
                if "weights_only" not in kwargs:
                    kwargs["weights_only"] = False
                return _orig_torch_load(*args, **kwargs)

            torch.load = _patched_torch_load
            torch._zerocost_patched = True
    except ImportError:
        pass

    # 2. Torchaudio soundfile backend fallback patch
    try:
        import soundfile as sf
        import torch
        import torchaudio

        if not getattr(torchaudio, "_zerocost_patched", False):
            def _patched_torchaudio_load(filepath, **kwargs):
                data, sr = sf.read(filepath, dtype="float32")
                tensor = torch.from_numpy(data)
                if tensor.ndim == 1:
                    tensor = tensor.unsqueeze(0)
                else:
                    tensor = tensor.t()
                return tensor, sr

            torchaudio.load = _patched_torchaudio_load
            torchaudio._zerocost_patched = True
    except ImportError:
        pass


def get_ffmpeg_path() -> Optional[str]:
    """Locate the ffmpeg binary across common system and Homebrew paths."""
    # Check PATH first
    found = shutil.which("ffmpeg")
    if found:
        return found

    # Fallback to standard Apple Silicon / Intel Homebrew and MacPorts paths
    candidates = [
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/opt/local/bin/ffmpeg",
        "/usr/bin/ffmpeg",
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


@dataclass
class HardwareReport:
    """Structured report of host hardware, accelerator availability, and routing."""

    platform_system: str
    machine_arch: str
    is_apple_silicon: bool
    cpu_count: int
    ram_gb: Optional[float]
    torch_version: Optional[str]
    cuda_available: bool
    cuda_device_name: Optional[str]
    mps_available: bool
    whisper_device: str
    xtts_device: str
    ffmpeg_path: Optional[str]
    ffmpeg_version: Optional[str]
    xtts_weights_cached: bool


class HardwareConfig:
    """Evaluates host system capabilities and calculates optimal execution devices."""

    @classmethod
    def is_apple_silicon(cls) -> bool:
        """Return True if running natively on Apple Silicon (arm64 on Darwin)."""
        return sys.platform == "darwin" and platform.machine() in ("arm64", "aarch64")

    @classmethod
    def get_ram_gb(cls) -> Optional[float]:
        """Attempt to read total system memory in GB."""
        try:
            if sys.platform == "darwin":
                cmd = ["sysctl", "-n", "hw.memsize"]
                out = subprocess.check_output(cmd, text=True).strip()
                return round(int(out) / (1024**3), 2)
            elif sys.platform.startswith("linux"):
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            kb = int(line.split()[1])
                            return round(kb / (1024**2), 2)
        except Exception:
            pass
        return None

    @classmethod
    def is_xtts_cached(cls) -> bool:
        """Check if Coqui XTTS-v2 checkpoint exists in standard cache locations."""
        cache_paths = [
            os.path.expanduser("~/Library/Application Support/tts/tts_models--multilingual--multi-dataset--xtts_v2/model.pth"),
            os.path.expanduser("~/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/model.pth"),
        ]
        return any(os.path.isfile(p) for p in cache_paths)

    @classmethod
    def get_ffmpeg_version(cls, ffmpeg_path: Optional[str]) -> Optional[str]:
        """Extract first line of ffmpeg -version."""
        if not ffmpeg_path:
            return None
        try:
            out = subprocess.check_output([ffmpeg_path, "-version"], text=True, stderr=subprocess.STDOUT)
            return out.splitlines()[0]
        except Exception:
            return None

    @classmethod
    def probe(cls) -> HardwareReport:
        """Perform a full system probe and return a HardwareReport."""
        apply_runtime_patches()

        torch_ver = None
        cuda_avail = False
        cuda_name = None
        mps_avail = False

        try:
            import torch
            torch_ver = torch.__version__
            cuda_avail = torch.cuda.is_available()
            if cuda_avail:
                cuda_name = torch.cuda.get_device_name(0)
            if hasattr(torch.backends, "mps"):
                mps_avail = torch.backends.mps.is_available()
        except ImportError:
            pass

        apple_silicon = cls.is_apple_silicon()
        cpu_count = os.cpu_count() or 1
        ram_gb = cls.get_ram_gb()

        # Device assignment logic:
        # CUDA takes top priority for both models if an Nvidia GPU is present.
        # For Apple Silicon: Whisper works well on MPS or CPU; XTTS MUST use CPU
        # to prevent unsupported operator crashes.
        if cuda_avail:
            whisper_dev = "cuda"
            xtts_dev = "cuda"
        elif apple_silicon:
            # Whisper on Apple Silicon is fast on CPU or MPS. CPU fp32 is rock-solid.
            whisper_dev = "cpu"
            xtts_dev = "cpu"
        else:
            whisper_dev = "cpu"
            xtts_dev = "cpu"

        ffmpeg_bin = get_ffmpeg_path()
        ffmpeg_ver = cls.get_ffmpeg_version(ffmpeg_bin)
        xtts_cached = cls.is_xtts_cached()

        return HardwareReport(
            platform_system=platform.system(),
            machine_arch=platform.machine(),
            is_apple_silicon=apple_silicon,
            cpu_count=cpu_count,
            ram_gb=ram_gb,
            torch_version=torch_ver,
            cuda_available=cuda_avail,
            cuda_device_name=cuda_name,
            mps_available=mps_avail,
            whisper_device=whisper_dev,
            xtts_device=xtts_dev,
            ffmpeg_path=ffmpeg_bin,
            ffmpeg_version=ffmpeg_ver,
            xtts_weights_cached=xtts_cached,
        )
