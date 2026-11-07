"""Unit tests for the voicefarm CLI commands."""

from click.testing import CliRunner
from zerocostvoicefarm.cli import cli


def test_cli_help():
    """Verify main CLI help displays all required commands including mcp."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "clone" in result.output
    assert "transcribe" in result.output
    assert "dub" in result.output
    assert "presets" in result.output
    assert "mcp" in result.output


def test_cli_presets():
    """Verify 'voicefarm presets' runs and lists supported languages and hardware info."""
    runner = CliRunner()
    result = runner.invoke(cli, ["presets"])
    assert result.exit_code == 0
    assert "Supported XTTS-v2 Foreign Languages" in result.output
    assert "Hardware Diagnostic" in result.output
    assert "French" in result.output
    assert "Japanese" in result.output
    assert "Spanish" in result.output
    assert "German" in result.output


def test_cli_clone_missing_args():
    """Verify 'voicefarm clone' fails with missing mandatory arguments."""
    runner = CliRunner()
    result = runner.invoke(cli, ["clone"])
    assert result.exit_code != 0
    assert "Missing option" in result.output
