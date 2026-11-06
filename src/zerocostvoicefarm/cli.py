"""Command-Line Interface entrypoint for zerocostvoicefarm.

Module Architecture:
--------------------
This module exposes the unified 'voicefarm' CLI tool providing four core commands:
- clone: Quick zero-cost voice cloning from a reference sample to a target audio file.
- transcribe: Local speech-to-text with OpenAI Whisper, exporting JSON, SRT, and VTT.
- dub: Full end-to-end pipeline (transcribe -> translate -> clone -> duck audio -> remux video).
- presets: Displays supported foreign languages and diagnostic hardware acceleration status.
"""

import os
import sys
import time
import shutil
import tempfile
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from zerocostvoicefarm.config import (
    SUPPORTED_LANGUAGES,
    DEFAULT_WHISPER_MODEL,
    DEFAULT_TTS_MODEL,
    DEFAULT_DUCK_LEVEL,
    HardwareConfig,
    apply_runtime_patches,
)
from zerocostvoicefarm.transcribe import Transcriber
from zerocostvoicefarm.translate import TranslationMapper
from zerocostvoicefarm.clone import VoiceCloner
from zerocostvoicefarm.master import AudioMasterer

console = Console()


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version="0.1.0", prog_name="zerocostvoicefarm")
def cli():
    """zerocostvoicefarm: Zero-cloud-cost multilingual video dubbing and voice cloning CLI."""
    apply_runtime_patches()


# -------------------------------------------------------------------------
# Command 1: CLONE
# -------------------------------------------------------------------------
@cli.command("clone")
@click.option("--ref", "-r", "speaker_ref", required=True, type=click.Path(exists=True, dir_okay=False),
              help="Path to speaker reference audio WAV (10-30s sample).")
@click.option("--text", "-t", required=True, help="Dialogue text to synthesize in cloned voice.")
@click.option("--lang", "-l", default="en", show_default=True,
              help="Target language code (e.g. en, fr, ja, es, de, it).")
@click.option("--out", "-o", "output_wav", default="cloned_voice.wav", show_default=True,
              type=click.Path(dir_okay=False), help="Path for the generated output WAV.")
def clone_cmd(speaker_ref: str, text: str, lang: str, output_wav: str):
    """Direct zero-cost voice cloning from a reference sample."""
    lang_code = lang.lower().strip()
    if lang_code not in SUPPORTED_LANGUAGES:
        console.print(f"[yellow]Warning:[/yellow] Language '{lang_code}' not in standard list. Proceeding anyway.")

    console.print(Panel(
        f"[bold cyan]Input Text:[/bold cyan] {text}\n"
        f"[bold cyan]Reference:[/bold cyan] {speaker_ref}\n"
        f"[bold cyan]Language:[/bold cyan] {SUPPORTED_LANGUAGES.get(lang_code, lang_code)} ({lang_code})\n"
        f"[bold cyan]Output:[/bold cyan] {output_wav}",
        title="[bold green]Zero-Cost Voice Cloning[/bold green]",
        border_style="green"
    ))

    t0 = time.time()
    cloner = VoiceCloner()
    console.print("[dim]Loading Coqui XTTS-v2 checkpoint...[/dim]")
    cloner.clone_single(
        text=text,
        speaker_wav=speaker_ref,
        language=lang_code,
        output_wav=output_wav,
    )
    elapsed = time.time() - t0
    console.print(f"[bold green]✔ Synthesis complete![/bold green] Saved to {output_wav} in {elapsed:.2f}s")


# -------------------------------------------------------------------------
# Command 2: TRANSCRIBE
# -------------------------------------------------------------------------
@cli.command("transcribe")
@click.argument("media_path", type=click.Path(exists=True))
@click.option("--model", "-m", default=DEFAULT_WHISPER_MODEL, show_default=True,
              help="Whisper model size (tiny, base, small, medium, large).")
@click.option("--out-dir", "-o", default="./output", show_default=True,
              type=click.Path(file_okay=False), help="Directory to save JSON, SRT, VTT, and TXT files.")
@click.option("--prompt", "-p", default=None, help="Optional context or terminology prompt for Whisper.")
@click.option("--verbose", is_flag=True, default=False, help="Print Whisper progress timestamps.")
def transcribe_cmd(media_path: str, model: str, out_dir: str, prompt: Optional[str], verbose: bool):
    """Extract audio, transcribe dialogue, and output timestamped segments + subtitles."""
    console.print(Panel(
        f"[bold cyan]Source Media:[/bold cyan] {media_path}\n"
        f"[bold cyan]Model:[/bold cyan] {model}\n"
        f"[bold cyan]Destination:[/bold cyan] {out_dir}",
        title="[bold blue]Whisper Transcription[/bold blue]",
        border_style="blue"
    ))

    base_name = os.path.splitext(os.path.basename(media_path))[0]
    transcriber = Transcriber(model_name=model)

    t0 = time.time()
    console.print(f"[dim]Transcribing on {transcriber.device}...[/dim]")
    result = transcriber.transcribe(
        media_path=media_path,
        initial_prompt=prompt,
        verbose=verbose,
    )
    paths = Transcriber.export_transcripts(result, output_dir=out_dir, base_name=base_name)
    elapsed = time.time() - t0

    num_segments = len(result.get("segments", []))
    console.print(f"[bold green]✔ Transcribed {num_segments} segments in {elapsed:.2f}s![/bold green]")
    for fmt, p in paths.items():
        console.print(f"  • [bold]{fmt.upper()}:[/bold] {p}")


# -------------------------------------------------------------------------
# Command 3: DUB
# -------------------------------------------------------------------------
@cli.command("dub")
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--target-lang", "-l", required=True,
              help="Target language code (e.g. fr, ja, es, de, it).")
@click.option("--ref", "-r", "speaker_ref", required=True, type=click.Path(exists=True, dir_okay=False),
              help="Path to speaker reference audio WAV.")
@click.option("--out", "-o", "output_path", default=None,
              help="Output video path (defaults to <name>_<lang>_dubbed.m4v).")
@click.option("--translations", "-t", default=None, type=click.Path(exists=True, dir_okay=False),
              help="Optional pre-existing translation JSON file to bypass auto-translation.")
@click.option("--model", "-m", default=DEFAULT_WHISPER_MODEL, show_default=True,
              help="Whisper model size for transcription.")
@click.option("--duck-level", default=DEFAULT_DUCK_LEVEL, show_default=True, type=float,
              help="Background volume multiplier during speech (0.18 = 18%).")
@click.option("--work-dir", default=None, type=click.Path(file_okay=False),
              help="Scratch workspace for temporary stems and cached segments.")
@click.option("--keep-temp", is_flag=True, default=False, help="Retain temporary audio stems after completion.")
def dub_cmd(
    video_path: str,
    target_lang: str,
    speaker_ref: str,
    output_path: Optional[str],
    translations: Optional[str],
    model: str,
    duck_level: float,
    work_dir: Optional[str],
    keep_temp: bool,
):
    """Run the end-to-end pipeline (transcribe -> translate -> clone -> duck audio -> remux video)."""
    lang_code = target_lang.lower().strip()
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    ext = os.path.splitext(video_path)[1] or ".m4v"

    if output_path is None:
        output_path = f"{base_name}_{lang_code}_dubbed{ext}"

    # Setup scratch directory
    created_scratch = False
    if work_dir is None:
        scratch_dir = tempfile.mkdtemp(prefix=f"voicefarm_{lang_code}_")
        created_scratch = True
    else:
        scratch_dir = work_dir
        os.makedirs(scratch_dir, exist_ok=True)

    console.print(Panel(
        f"[bold cyan]Source Video:[/bold cyan] {video_path}\n"
        f"[bold cyan]Target Language:[/bold cyan] {SUPPORTED_LANGUAGES.get(lang_code, lang_code)} ({lang_code})\n"
        f"[bold cyan]Reference Voice:[/bold cyan] {speaker_ref}\n"
        f"[bold cyan]Translations Source:[/bold cyan] {'Provided (' + translations + ')' if translations else 'Auto-Translate (GTX)'}\n"
        f"[bold cyan]Output Destination:[/bold cyan] {output_path}\n"
        f"[bold cyan]Scratch Directory:[/bold cyan] {scratch_dir}",
        title="[bold magenta]Zero-Cost Video Dubbing Pipeline[/bold magenta]",
        border_style="magenta"
    ))

    t_pipeline_start = time.time()

    try:
        # Step 1: Segments Acquisition (Transcription or Pre-translated)
        if translations:
            console.print(f"[bold cyan][1/4][/bold cyan] Loading pre-translated segments from {translations}...")
            segments = TranslationMapper.load_translation_file(translations, target_lang=lang_code)
            console.print(f"  Loaded {len(segments)} segments.")
        else:
            console.print(f"[bold cyan][1/4][/bold cyan] Transcribing source video with Whisper ('{model}')...")
            transcriber = Transcriber(model_name=model)
            t_tr = time.time()
            trans_res = transcriber.transcribe(video_path, work_dir=scratch_dir)
            segments = trans_res.get("segments", [])
            console.print(f"  Transcribed {len(segments)} segments in {time.time() - t_tr:.2f}s.")

            # Step 2: Auto-Translation & Timing Windows
            console.print(f"[bold cyan][2/4][/bold cyan] Translating segments to {lang_code.upper()} and calculating cut boundaries...")
            t_tl = time.time()
            segments = TranslationMapper.translate_segments(
                segments,
                target_lang=lang_code,
                source_lang=trans_res.get("language", "auto")
            )
            console.print(f"  Translated {len(segments)} segments in {time.time() - t_tl:.2f}s.")

            # Save translated segments backup in scratch
            trans_backup = os.path.join(scratch_dir, f"translated_{lang_code}_segments.json")
            with open(trans_backup, "w", encoding="utf-8") as f:
                import json
                json.dump(segments, f, ensure_ascii=False, indent=2)

        # Step 3: Voice Cloning & Dynamic Tempo Fitting
        console.print(f"[bold cyan][3/4][/bold cyan] Synthesizing voice clones and fitting speech tempo...")
        cloner = VoiceCloner()
        t_clone = time.time()

        def on_segment_done(curr, total, seg):
            if curr % 5 == 0 or curr == total:
                console.print(f"  • [{curr}/{total}] Seg {seg['id']:02d}: \"{seg.get('target_text', '')[:35]}...\"")

        segments_with_audio = cloner.synthesize_segments(
            segments=segments,
            speaker_wav=speaker_ref,
            language=lang_code,
            cache_dir=scratch_dir,
            text_key="target_text",
            progress_callback=on_segment_done,
        )
        console.print(f"  Voice synthesis completed in {time.time() - t_clone:.2f}s.")

        # Step 4: Mastering, Ducking & Remuxing
        console.print(f"[bold cyan][4/4][/bold cyan] Ducking background audio ({int(duck_level*100)}%), mixing stereo master, and remuxing video...")
        t_master = time.time()
        masterer = AudioMasterer(duck_level=duck_level)
        final_file = masterer.run_mastering_pipeline(
            source_media_path=video_path,
            segments_with_audio=segments_with_audio,
            output_path=output_path,
            work_dir=scratch_dir,
        )
        console.print(f"  Mastering & remux completed in {time.time() - t_master:.2f}s.")

        total_elapsed = time.time() - t_pipeline_start
        console.print(f"\n[bold green]✔ PIPELINE COMPLETE![/bold green] Output generated at: [bold]{final_file}[/bold] ({total_elapsed:.2f}s)")

    finally:
        if created_scratch and not keep_temp:
            shutil.rmtree(scratch_dir, ignore_errors=True)


# -------------------------------------------------------------------------
# Command 4: PRESETS
# -------------------------------------------------------------------------
@cli.command("presets")
def presets_cmd():
    """List supported XTTS-v2 languages and display local hardware status."""
    report = HardwareConfig.probe()

    # Hardware Status Panel
    hw_table = Table(show_header=True, header_style="bold cyan", box=None)
    hw_table.add_column("Property", style="bold")
    hw_table.add_column("Value")

    hw_table.add_row("Operating System", f"{report.platform_system} ({report.machine_arch})")
    hw_table.add_row("Apple Silicon Native", "Yes (Apple Silicon M-Series)" if report.is_apple_silicon else "No")
    hw_table.add_row("CPU Logical Cores", str(report.cpu_count))
    hw_table.add_row("Host RAM", f"{report.ram_gb:.1f} GB" if report.ram_gb else "Unknown")
    hw_table.add_row("PyTorch Version", str(report.torch_version or "Not Installed"))
    hw_table.add_row("MPS (Apple Silicon GPU)", "Available" if report.mps_available else "Not Available")
    hw_table.add_row("CUDA (Nvidia GPU)", f"Available ({report.cuda_device_name})" if report.cuda_available else "Not Available")
    hw_table.add_row("Whisper Inference Target", f"[green]{report.whisper_device.upper()}[/green]")
    hw_table.add_row("XTTS Inference Target", f"[green]{report.xtts_device.upper()}[/green] (CPU rock-solid on Apple Silicon)")
    hw_table.add_row("FFmpeg Binary", report.ffmpeg_path or "[red]Missing (brew install ffmpeg)[/red]")
    if report.ffmpeg_version:
        hw_table.add_row("FFmpeg Version", report.ffmpeg_version)
    hw_table.add_row("XTTS-v2 Checkpoint Cached", "[green]Yes (1.86 GB)[/green]" if report.xtts_weights_cached else "[yellow]No (Will auto-download on first run)[/yellow]")

    console.print(Panel(hw_table, title="[bold green]Hardware Diagnostic & Compute Routing[/bold green]", border_style="green"))

    # Languages Table
    lang_table = Table(title="Supported XTTS-v2 Foreign Languages", show_header=True, header_style="bold magenta")
    lang_table.add_column("Code", style="bold yellow", width=10)
    lang_table.add_column("Language Name", style="white")

    for code, name in SUPPORTED_LANGUAGES.items():
        lang_table.add_row(code, name)

    console.print(lang_table)


def main():
    """Main binary entrypoint."""
    cli()


if __name__ == "__main__":
    main()
