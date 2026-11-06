"""Unit tests for translation mapper, timing calculations, and subtitle formatting."""

import os
import json
import tempfile

from zerocostvoicefarm.transcribe import format_timestamp
from zerocostvoicefarm.translate import TranslationMapper


def test_format_timestamp():
    """Verify SRT comma and VTT dot formatting across second boundaries."""
    assert format_timestamp(0.0, decimal_sep=",") == "00:00:00,000"
    assert format_timestamp(31.5, decimal_sep=",") == "00:00:31,500"
    assert format_timestamp(3661.123, decimal_sep=".") == "01:01:01.123"


def test_map_timing_windows():
    """Verify max_duration calculates slot up to next segment start."""
    raw_segments = [
        {"id": 1, "start": 10.0, "end": 14.0, "duration": 4.0, "text": "First segment"},
        {"id": 2, "start": 18.0, "end": 22.0, "duration": 4.0, "text": "Second segment"},
        {"id": 3, "start": 25.0, "end": 30.0, "duration": 5.0, "text": "Final segment"},
    ]

    mapped = TranslationMapper.map_timing_windows(raw_segments, total_duration=35.0)

    # Segment 1: from 10.0 to next start (18.0) -> max_duration = 8.0
    assert mapped[0]["max_duration"] == 8.0
    # Segment 2: from 18.0 to next start (25.0) -> max_duration = 7.0
    assert mapped[1]["max_duration"] == 7.0
    # Segment 3: from 25.0 to total_duration (35.0) -> max_duration = 10.0
    assert mapped[2]["max_duration"] == 10.0


def test_load_translation_file():
    """Verify loading pre-existing multilingual translation datasets."""
    sample_data = [
        {"id": 1, "start": 5.0, "end": 10.0, "en": "Hello", "fr": "Bonjour", "ja": "こんにちは"},
        {"id": 2, "start": 12.0, "end": 15.0, "en": "World", "fr": "Monde", "ja": "世界"},
    ]

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(sample_data, f)
        temp_path = f.name

    try:
        fr_segs = TranslationMapper.load_translation_file(temp_path, target_lang="fr")
        assert len(fr_segs) == 2
        assert fr_segs[0]["target_text"] == "Bonjour"
        assert fr_segs[1]["target_text"] == "Monde"
        assert fr_segs[0]["max_duration"] == 7.0  # 12.0 - 5.0

        ja_segs = TranslationMapper.load_translation_file(temp_path, target_lang="ja")
        assert ja_segs[0]["target_text"] == "こんにちは"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_export_subtitles():
    """Verify subtitle generation writes compliant SRT format."""
    segments = [
        {"id": 1, "start": 2.0, "end": 4.5, "target_text": "Sample translated line"},
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".srt", delete=False) as f:
        temp_path = f.name

    try:
        TranslationMapper.export_subtitles(segments, temp_path, format_type="srt")
        with open(temp_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "1\n00:00:02,000 --> 00:00:04,500\nSample translated line" in content
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
