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


def test_mcp_live_stdio_protocol():
    """Verify live MCP server over stdio with full JSON-RPC 2.0 handshake and tool invocation."""
    import sys
    import subprocess

    proc = subprocess.Popen(
        [sys.executable, "-m", "zerocostvoicefarm.cli", "mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        def send_and_read(req):
            proc.stdin.write(json.dumps(req) + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
            return json.loads(line)

        # 1. Initialize
        init_res = send_and_read({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-agent", "version": "1.0.0"},
            },
        })
        assert "serverInfo" in init_res.get("result", {})
        assert init_res["result"]["serverInfo"]["name"] == "zerocostvoicefarm"

        # 2. Initialized notification
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
        proc.stdin.flush()

        # 3. List tools
        tools_res = send_and_read({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        })
        tool_names = [t["name"] for t in tools_res.get("result", {}).get("tools", [])]
        assert "voicefarm_dub" in tool_names
        assert "voicefarm_clone" in tool_names
        assert "voicefarm_transcribe" in tool_names
        assert "voicefarm_presets" in tool_names

        # 4. Call presets tool
        call_res = send_and_read({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "voicefarm_presets",
                "arguments": {},
            },
        })
        text_out = call_res.get("result", {}).get("content", [{}])[0].get("text", "")
        payload = json.loads(text_out)
        assert len(payload["supported_languages"]) == 17
    finally:
        proc.terminate()
        proc.wait(timeout=5)
