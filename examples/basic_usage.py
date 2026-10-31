"""Basic usage example for ollama-mcp-orchestrator.

Run from the repository root:

    python -m examples.basic_usage

This example shows how to wire a FastMCP server, a stub model, and the
Orchestrator together. Replace the stub model with OllamaModel for live use.
"""

from __future__ import annotations

import asyncio
import os
import sys

# Allow running as `python examples/basic_usage.py` as well as
# `python -m examples.basic_usage`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_tool_orchestrator import Orchestrator, OrchestratorConfig
from llm_tool_orchestrator.mcp_client import FastMcpAdapter
from llm_tool_orchestrator.models import Model
from examples.mcp_server import server


class StubModel(Model):
    """A scripted model for demonstration: issues one tool call, then answers."""

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


async def main() -> None:
    async with FastMcpAdapter(server) as mcp:
        model = StubModel(
            tool_call={"function": {"name": "get_all_items", "arguments": {}}},
            final_answer="Here are your items.",
        )
        config = OrchestratorConfig(max_steps=3)  # mutations disabled by default
        orch = Orchestrator(model, mcp, config)
        text, results = await orch.run_with_results("List all items.")
        print("Assistant:", text)
        for r in results:
            print(f"Tool {r.tool_name}: ok={r.ok}, result={r.result!r}")


if __name__ == "__main__":
    asyncio.run(main())
