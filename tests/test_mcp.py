"""Unit tests for the MCP server tools and instantiation."""

import json
from zerocostvoicefarm.mcp import create_mcp_server


def test_create_mcp_server():
    """Verify MCP server registers all four tools."""
    server = create_mcp_server()
    assert server is not None

    # Retrieve registered tool names
    tools = server._tool_manager.list_tools()
    tool_names = [t.name for t in tools]
    assert "voicefarm_dub" in tool_names
    assert "voicefarm_clone" in tool_names
    assert "voicefarm_transcribe" in tool_names
    assert "voicefarm_presets" in tool_names


def test_mcp_presets_tool():
    """Verify voicefarm_presets tool returns valid JSON with languages and hardware."""
    server = create_mcp_server()
    # Find tool function
    tools = {t.name: t.fn for t in server._tool_manager.list_tools()}
    presets_fn = tools["voicefarm_presets"]
    raw_res = presets_fn()
    data = json.loads(raw_res)

    assert "supported_languages" in data
    assert "ja" in data["supported_languages"]
    assert "fr" in data["supported_languages"]
    assert "hardware" in data
    assert "is_apple_silicon" in data["hardware"]
