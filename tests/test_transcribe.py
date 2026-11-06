"""Unit tests for Whisper transcript export logic."""

import os
import tempfile
import json

from zerocostvoicefarm.transcribe import Transcriber


def test_export_transcripts():
    """Verify JSON, SRT, VTT, and TXT files are correctly written."""
    mock_result = {
        "text": "Hello world. Testing voice farm.",
        "language": "en",
        "segments": [
            {"id": 1, "start": 0.5, "end": 2.0, "duration": 1.5, "text": "Hello world."},
            {"id": 2, "start": 2.5, "end": 4.0, "duration": 1.5, "text": "Testing voice farm."},
        ]
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = Transcriber.export_transcripts(mock_result, output_dir=tmp_dir, base_name="test_export")
        assert os.path.isfile(paths["json"])
        assert os.path.isfile(paths["srt"])
        assert os.path.isfile(paths["vtt"])
        assert os.path.isfile(paths["txt"])

        # Check JSON integrity
        with open(paths["json"], "r", encoding="utf-8") as f:
            data = json.load(f)
            assert len(data) == 2
            assert data[0]["text"] == "Hello world."

        # Check SRT formatting
        with open(paths["srt"], "r", encoding="utf-8") as f:
            srt_content = f.read()
            assert "1\n00:00:00,500 --> 00:00:02,000\nHello world." in srt_content

        # Check VTT formatting
        with open(paths["vtt"], "r", encoding="utf-8") as f:
            vtt_content = f.read()
            assert "WEBVTT" in vtt_content
            assert "00:00:00.500 --> 00:00:02.000" in vtt_content
