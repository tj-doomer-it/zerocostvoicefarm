"""Model Context Protocol (MCP) Server for zerocostvoicefarm.

Module Architecture:
--------------------
This module exposes zerocostvoicefarm as a native MCP server for autonomous
AI coding and content agents (Claude Code, Cursor, Cline, Antigravity, OpenHands).

By running `voicefarm mcp`, agents can autonomously invoke voice cloning,
video transcription, and the complete end-to-end video dubbing pipeline directly
as native function tools.
"""

import os
import json
import tempfile
import time
from typing import Optional, Dict, Any

from zerocostvoicefarm.config import (
    SUPPORTED_LANGUAGES,
    DEFAULT_WHISPER_MODEL,
    DEFAULT_DUCK_LEVEL,
    HardwareConfig,
    apply_runtime_patches,
)
from zerocostvoicefarm.transcribe import Transcriber
from zerocostvoicefarm.translate import TranslationMapper
from zerocostvoicefarm.clone import VoiceCloner
from zerocostvoicefarm.master import AudioMasterer


def get_mcp_server_class():
    """Retrieve the MCPServer class supporting both mcp 2.x and 1.x."""
    try:
        from mcp.server.mcpserver import MCPServer
        return MCPServer
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP
            return FastMCP
        except ImportError:
            raise ImportError(
                "The 'mcp' package is required to run the MCP server. "
                "Install it with: pip install 'zerocostvoicefarm[mcp]' or pip install mcp"
            )


def create_mcp_server():
    """Construct and configure the zerocostvoicefarm MCP Server."""
    apply_runtime_patches()
    ServerCls = get_mcp_server_class()
    server = ServerCls("zerocostvoicefarm")

    @server.tool(
        name="voicefarm_dub",
        description=(
            "Autonomous end-to-end video dubbing pipeline. "
            "Transcribes dialogue with Whisper, translates to target_lang, clones speaker "
            "vocal timbre using Coqui XTTS-v2, paces foreign dialogue to cut windows via FFmpeg atempo, "
            "smoothly ducks background audio to 18%, and losslessly remuxes into a final video."
        )
    )
    def voicefarm_dub(
        video_path: str,
        target_lang: str,
        speaker_ref: str,
        output_path: Optional[str] = None,
        translations: Optional[str] = None,
        model: str = DEFAULT_WHISPER_MODEL,
        duck_level: float = DEFAULT_DUCK_LEVEL,
    ) -> str:
        """Run the full autonomous dubbing pipeline for an agent."""
        if not os.path.isfile(video_path):
            return json.dumps({"error": f"Source video not found: {video_path}"})
        if not os.path.isfile(speaker_ref):
            return json.dumps({"error": f"Speaker reference audio not found: {speaker_ref}"})

        lang_code = target_lang.lower().strip()
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        ext = os.path.splitext(video_path)[1] or ".m4v"

        if output_path is None:
            output_path = os.path.abspath(f"{base_name}_{lang_code}_dubbed{ext}")
        else:
            output_path = os.path.abspath(output_path)

        scratch_dir = tempfile.mkdtemp(prefix=f"voicefarm_mcp_{lang_code}_")
        t0 = time.time()

        try:
            # 1. Segments acquisition
            if translations and os.path.isfile(translations):
                segments = TranslationMapper.load_translation_file(translations, target_lang=lang_code)
            else:
                transcriber = Transcriber(model_name=model)
                trans_res = transcriber.transcribe(video_path, work_dir=scratch_dir)
                raw_segs = trans_res.get("segments", [])
                segments = TranslationMapper.translate_segments(
                    raw_segs,
                    target_lang=lang_code,
                    source_lang=trans_res.get("language", "auto")
                )

            # 2. Voice cloning & cut-window tempo fitting
            cloner = VoiceCloner()
            segs_with_audio = cloner.synthesize_segments(
                segments=segments,
                speaker_wav=speaker_ref,
                language=lang_code,
                cache_dir=scratch_dir,
                text_key="target_text",
            )

            # 3. Audio ducking and video remuxing
            masterer = AudioMasterer(duck_level=duck_level)
            final_file = masterer.run_mastering_pipeline(
                source_media_path=video_path,
                segments_with_audio=segs_with_audio,
                output_path=output_path,
                work_dir=scratch_dir,
            )

            elapsed = round(time.time() - t0, 2)
            return json.dumps({
                "status": "success",
                "output_video": final_file,
                "target_language": lang_code,
                "segments_count": len(segments),
                "processing_time_s": elapsed,
            }, indent=2)

        except Exception as exc:
            return json.dumps({"status": "error", "message": str(exc)}, indent=2)

    @server.tool(
        name="voicefarm_clone",
        description=(
            "Direct zero-cost voice cloning. Synthesizes input text in the vocal identity "
            "of speaker_ref WAV using Coqui XTTS-v2 and outputs a high-fidelity WAV file."
        )
    )
    def voicefarm_clone(
        text: str,
        speaker_ref: str,
        language: str = "en",
        output_wav: str = "cloned_voice.wav",
    ) -> str:
        """Clone voice for a single phrase."""
        if not os.path.isfile(speaker_ref):
            return json.dumps({"error": f"Speaker reference audio not found: {speaker_ref}"})

        output_wav = os.path.abspath(output_wav)
        t0 = time.time()
        try:
            cloner = VoiceCloner()
            out = cloner.clone_single(
                text=text,
                speaker_wav=speaker_ref,
                language=language.lower().strip(),
                output_wav=output_wav,
            )
            elapsed = round(time.time() - t0, 2)
            return json.dumps({
                "status": "success",
                "output_wav": out,
                "language": language,
                "processing_time_s": elapsed,
            }, indent=2)
        except Exception as exc:
            return json.dumps({"status": "error", "message": str(exc)}, indent=2)

    @server.tool(
        name="voicefarm_transcribe",
        description=(
            "Extracts audio from media and performs timestamped speech-to-text with OpenAI Whisper. "
            "Exports structured JSON segments, SRT subtitles, WebVTT, and plaintext transcript."
        )
    )
    def voicefarm_transcribe(
        media_path: str,
        model: str = DEFAULT_WHISPER_MODEL,
        out_dir: str = "./output",
    ) -> str:
        """Transcribe media and export transcripts."""
        if not os.path.isfile(media_path):
            return json.dumps({"error": f"Source media file not found: {media_path}"})

        out_dir = os.path.abspath(out_dir)
        base_name = os.path.splitext(os.path.basename(media_path))[0]
        t0 = time.time()

        try:
            transcriber = Transcriber(model_name=model)
            res = transcriber.transcribe(media_path=media_path)
            paths = Transcriber.export_transcripts(res, output_dir=out_dir, base_name=base_name)
            elapsed = round(time.time() - t0, 2)
            return json.dumps({
                "status": "success",
                "detected_language": res.get("language"),
                "segments_count": len(res.get("segments", [])),
                "exported_files": paths,
                "processing_time_s": elapsed,
            }, indent=2)
        except Exception as exc:
            return json.dumps({"status": "error", "message": str(exc)}, indent=2)

    @server.tool(
        name="voicefarm_presets",
        description="Returns supported foreign languages (17 languages) and diagnostic local hardware acceleration status."
    )
    def voicefarm_presets() -> str:
        """Return diagnostic info on supported languages and hardware accelerators."""
        report = HardwareConfig.probe()
        return json.dumps({
            "supported_languages": SUPPORTED_LANGUAGES,
            "hardware": {
                "platform": report.platform_system,
                "arch": report.machine_arch,
                "is_apple_silicon": report.is_apple_silicon,
                "cpu_count": report.cpu_count,
                "ram_gb": report.ram_gb,
                "torch_version": report.torch_version,
                "cuda_available": report.cuda_available,
                "mps_available": report.mps_available,
                "whisper_device": report.whisper_device,
                "xtts_device": report.xtts_device,
                "ffmpeg_path": report.ffmpeg_path,
                "xtts_weights_cached": report.xtts_weights_cached,
            }
        }, indent=2)

    return server


def run_mcp_server(transport: str = "stdio") -> None:
    """Launch the MCP server listening on stdio."""
    server = create_mcp_server()
    server.run(transport=transport)
