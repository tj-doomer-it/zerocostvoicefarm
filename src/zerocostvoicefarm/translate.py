"""Translation & Timing Segment Mapping for zerocostvoicefarm.

Module Architecture:
--------------------
This module acts as the bridge between Step 1 (Transcription) and Step 3 (Voice Cloning).
It fulfills two critical responsibilities:
1. Translating spoken dialogue into the target language without cloud API subscriptions.
2. Calculating maximum allowed audio windows (max_duration) between cuts so downstream
   TTS synthesis can dynamically fit speech pacing without cutting into adjacent scenes.

Zero-Cloud-Cost Design:
-----------------------
- Provides zero-cost, API-key-free neural machine translation via the open Google GTX endpoint.
- Accepts pre-translated JSON files (such as human-refined or LLM-produced translation datasets)
  via the --translations flag.
- Normalizes timing segments across multi-language keys ('fr', 'ja', 'es', 'de', 'target_text', etc.).
"""

import os
import json
import time
import urllib.request
import urllib.parse
from typing import Dict, List, Optional, Any, Callable

from zerocostvoicefarm.config import SUPPORTED_LANGUAGES
from zerocostvoicefarm.transcribe import format_timestamp


def free_web_translate(text: str, target_lang: str, source_lang: str = "auto") -> str:
    """Zero-cloud-cost machine translation using the free Google translate web endpoint.

    Why this approach:
    - Zero subscription fees, zero API tokens required.
    - Extremely reliable for short dialogue phrases and vlog scripts.
    - Handles Unicode (Japanese Kanji, accents, Cyrillic, etc.) out-of-the-box.
    """
    clean_text = text.strip()
    if not clean_text:
        return ""

    # Map dialect codes if necessary (e.g. zh-cn -> zh-CN)
    tl = "zh-CN" if target_lang.lower() in ("zh-cn", "zh") else target_lang

    url = (
        "https://translate.googleapis.com/translate_a/single?"
        f"client=gtx&sl={source_lang}&tl={tl}&dt=t&q="
        + urllib.parse.quote(clean_text)
    )

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"},
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            # The GTX endpoint returns a nested array where parts are in data[0]
            translated_parts = [part[0] for part in data[0] if part and len(part) > 0 and part[0]]
            return "".join(translated_parts).strip()
    except Exception as exc:
        # If network error occurs, fall back to returning original text rather than crashing
        return clean_text


class TranslationMapper:
    """Calculates timing constraints and coordinates dialogue translation."""

    @staticmethod
    def map_timing_windows(
        segments: List[Dict[str, Any]],
        total_duration: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Calculate max_duration slot for each segment.

        Invariant:
        The maximum window before the next dialogue segment starts is:
            next_start - current_start
        If this is the final segment, max_duration is total_duration - start,
        or current duration + 1.5s padding if total_duration is unknown.
        """
        mapped: List[Dict[str, Any]] = []
        n = len(segments)

        for idx, seg in enumerate(segments):
            s = dict(seg)
            start_time = float(s["start"])
            end_time = float(s.get("end", start_time + 1.0))
            duration = max(0.1, float(s.get("duration", end_time - start_time)))

            if idx + 1 < n:
                next_start = float(segments[idx + 1]["start"])
            elif total_duration is not None:
                next_start = float(total_duration)
            else:
                next_start = start_time + duration + 1.5

            max_dur = max(0.5, round(next_start - start_time, 3))
            s["max_duration"] = max_dur
            s["duration"] = round(duration, 3)
            mapped.append(s)

        return mapped

    @classmethod
    def translate_segments(
        cls,
        segments: List[Dict[str, Any]],
        target_lang: str,
        source_lang: str = "auto",
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Translate all segments and attach mapped timing constraints."""
        mapped = cls.map_timing_windows(segments)
        total = len(mapped)

        for idx, seg in enumerate(mapped):
            src_text = seg.get("text") or seg.get("en", "")
            # If target language is already provided in the segment dict, preserve it
            if target_lang in seg and seg[target_lang]:
                target_text = seg[target_lang]
            elif "target_text" in seg and seg["target_text"]:
                target_text = seg["target_text"]
            else:
                target_text = free_web_translate(src_text, target_lang=target_lang, source_lang=source_lang)
                # Small sleep to be polite to the GTX endpoint
                time.sleep(0.05)

            seg["target_text"] = target_text
            seg[target_lang] = target_text
            seg["target_lang"] = target_lang

            if progress_callback:
                progress_callback(idx + 1, total, target_text)

        return mapped

    @classmethod
    def load_translation_file(
        cls,
        file_path: str,
        target_lang: str,
        total_duration: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Load an existing translation JSON file and adapt it to the pipeline.

        Supports schemas from:
        - build_translations.py (contains keys: 'id', 'start', 'end', 'en', 'fr', 'ja', etc.)
        - zerocostvoicefarm standard format (contains: 'id', 'start', 'end', 'target_text')
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Translation file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if not isinstance(raw_data, list):
            raise ValueError(f"Expected a JSON list of segments in {file_path}")

        adapted_segments: List[Dict[str, Any]] = []
        for i, item in enumerate(raw_data):
            seg_id = item.get("id", i + 1)
            start_s = float(item["start"])
            end_s = float(item.get("end", start_s + float(item.get("duration", 2.0))))
            duration = round(end_s - start_s, 3)

            # Determine translated text
            target_text = ""
            if target_lang in item:
                target_text = str(item[target_lang]).strip()
            elif "target_text" in item:
                target_text = str(item["target_text"]).strip()
            elif "text" in item:
                target_text = str(item["text"]).strip()

            seg = dict(item)
            seg["id"] = seg_id
            seg["start"] = start_s
            seg["end"] = end_s
            seg["duration"] = duration
            seg["target_text"] = target_text
            seg[target_lang] = target_text
            seg["target_lang"] = target_lang
            adapted_segments.append(seg)

        return cls.map_timing_windows(adapted_segments, total_duration=total_duration)

    @staticmethod
    def export_subtitles(
        segments: List[Dict[str, Any]],
        output_path: str,
        format_type: str = "srt",
        text_key: str = "target_text",
    ) -> str:
        """Export translated segments to .srt or .vtt subtitle file."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        is_vtt = format_type.lower() == "vtt" or output_path.endswith(".vtt")
        dec_sep = "." if is_vtt else ","

        with open(output_path, "w", encoding="utf-8") as f:
            if is_vtt:
                f.write("WEBVTT\n\n")
            for seg in segments:
                t_start = format_timestamp(seg["start"], decimal_sep=dec_sep)
                t_end = format_timestamp(seg["end"], decimal_sep=dec_sep)
                text = seg.get(text_key) or seg.get("text", "")
                f.write(f"{seg['id']}\n{t_start} --> {t_end}\n{text}\n\n")

        return output_path
