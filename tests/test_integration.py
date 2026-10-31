"""Integration tests using a real, in-process FastMCP server and client.

No Ollama server is required: a stub model drives the conversation. The
server uses in-memory state; no application database is involved.
"""

import asyncio
import json

import pytest

from fastmcp import FastMCP

from llm_tool_orchestrator import Orchestrator, OrchestratorConfig
from llm_tool_orchestrator.mcp_client import FastMcpAdapter
from llm_tool_orchestrator.models import Model
from llm_tool_orchestrator.orchestrator import mcp_tool_to_ollama


# ---------------------------------------------------------------------------
# Self-contained example MCP server (no application database dependency).
# ---------------------------------------------------------------------------

_SERVER_ITEMS: dict[int, dict] = {}
_SERVER_NEXT_ID = 1


def _server_next_id() -> int:
    global _SERVER_NEXT_ID
    item_id = _SERVER_NEXT_ID
    _SERVER_NEXT_ID += 1
    return item_id


server = FastMCP(
    name="test-tools",
    instructions="Test tools for integration tests (in-memory state only).",
)


@server.tool
def get_all_items() -> str:
    return json.dumps(list(_SERVER_ITEMS.values()))


@server.tool
def get_item_by_id(item_id: int) -> str:
    item = _SERVER_ITEMS.get(item_id)
    return json.dumps(item if item is not None else {"error": "not found"})


@server.tool
def get_items_by_type(item_type: str) -> str:
    matches = [it for it in _SERVER_ITEMS.values() if it.get("type") == item_type]
    return json.dumps(matches)


@server.tool
def generate_inventory_report() -> str:
    items = sorted(_SERVER_ITEMS.values(), key=lambda it: it.get("id", 0))
    lines = ["TEST REPORT", "=" * 20, f"Total items: {len(items)}", ""]
    for it in items:
        lines.append(
            f"#{it.get('id')} | {it.get('name')} | {it.get('type')} | "
            f"price={it.get('price')} | qty={it.get('quantity')}"
        )
    lines.append("=" * 20)
    return "\n".join(lines)


@server.tool
def add_item(name: str, price: float, quantity: int, item_type: str) -> str:
    global _SERVER_NEXT_ID
    item_id = _SERVER_NEXT_ID
    _SERVER_NEXT_ID += 1
    _SERVER_ITEMS[item_id] = {
        "id": item_id,
        "name": name,
        "price": price,
        "quantity": quantity,
        "type": item_type,
    }
    return f"Added item {item_id} successfully"


@server.tool
def update_item(item_id: int, price: float | None = None, quantity: int | None = None) -> str:
    if item_id not in _SERVER_ITEMS:
        return "Item not found"
    item = _SERVER_ITEMS[item_id]
    if price is not None:
        item["price"] = price
    if quantity is not None:
        item["quantity"] = quantity
    return f"Updated item {item_id} successfully"


@server.tool
def delete_item(item_id: int) -> str:
    if item_id not in _SERVER_ITEMS:
        return "Item not found"
    del _SERVER_ITEMS[item_id]
    return f"Deleted item {item_id} successfully"


# ---------------------------------------------------------------------------
# Stub model
# ---------------------------------------------------------------------------

class StubModel(Model):
    """Model that issues one scripted tool call, then answers."""

    def __init__(self, tool_call, final_answer):
        self._tool_call = tool_call
        self._final_answer = final_answer
        self._called = False

    async def chat(self, messages, tools=None, **kwargs):  # noqa: ARG002
        if not self._called:
            self._called = True
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [self._tool_call],
            }
        return {"role": "assistant", "content": self._final_answer}


async def _run_with_stub(tool_call, final_answer, max_steps=3):
    async with FastMcpAdapter(server) as mcp:
        orch = Orchestrator(
            StubModel(tool_call, final_answer), mcp, OrchestratorConfig(max_steps=max_steps)
        )
        return await orch.run_with_results("help me")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_mcp_server_registers_expected_tools():
    async def main():
        async with FastMcpAdapter(server) as mcp:
            return await mcp.list_tools()

    tools = asyncio.run(main())
    names = {t["name"] for t in tools}
    expected = {
        "get_all_items", "get_item_by_id", "get_items_by_type",
        "generate_inventory_report", "add_item", "update_item", "delete_item",
    }
    assert expected.issubset(names)
    for t in tools:
        assert t["inputSchema"].get("type") == "object"


def test_mcp_tool_to_ollama_conversion_matches_server():
    async def main():
        async with FastMcpAdapter(server) as mcp:
            return await mcp.list_tools()

    tools = asyncio.run(main())
    by_name = {t["name"]: t for t in tools}
    for name in ("get_all_items", "add_item", "generate_inventory_report"):
        assert name in by_name
        ollama_tool = mcp_tool_to_ollama(by_name[name])
        assert ollama_tool["type"] == "function"
        assert ollama_tool["function"]["name"] == name


def test_end_to_end_read_only_tool():
    tool_call = {"function": {"name": "get_all_items", "arguments": {}}}
    text, results = asyncio.run(_run_with_stub(tool_call, "Done listing."))
    assert "Done listing." in text
    assert len(results) == 1
    assert results[0].tool_name == "get_all_items"
    assert results[0].ok is True


def test_report_generation_is_deterministic():
    async def main():
        async with FastMcpAdapter(server) as mcp:
            r1 = await mcp.call_tool("generate_inventory_report", {})
            r2 = await mcp.call_tool("generate_inventory_report", {})
            return r1, r2

    r1, r2 = asyncio.run(main())
    assert r1 == r2
    assert "TEST REPORT" in r1
    assert "Total items:" in r1


def test_mutation_then_read_reflects_change():
    async def main():
        async with FastMcpAdapter(server) as mcp:
            add = await mcp.call_tool("add_item", {
                "name": "widget", "price": 9.99, "quantity": 5, "item_type": "gadget",
            })
            all_items = await mcp.call_tool("get_all_items", {})
            return add, all_items

    add, all_items = asyncio.run(main())
    assert "Added" in add
    parsed = json.loads(all_items) if isinstance(all_items, str) else all_items
    assert any(it.get("name") == "widget" for it in parsed)


def test_blocked_tool_rejected_by_orchestrator():
    tool_call = {"function": {"name": "evil_tool", "arguments": {}}}
    text, results = asyncio.run(_run_with_stub(tool_call, "Stopped."))
    assert "Stopped." in text
    assert len(results) == 1
    assert results[0].ok is False
    assert "not allowed" in results[0].error
