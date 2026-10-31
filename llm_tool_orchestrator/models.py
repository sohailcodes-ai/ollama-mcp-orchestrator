"""Abstract interfaces (protocols) used by the orchestrator.

Decoupling the orchestrator from the concrete Ollama and FastMCP clients lets
the layer be unit-tested with mocked models and tool responses, and keeps the
integration free of any specific application.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .config import OrchestratorConfig


@runtime_checkable
class Model(Protocol):
    """A chat model that supports tool calling.

    `chat` returns the assistant message: a dict with optional `content` and
    `tool_calls`, where each tool call is
    ``{"function": {"name": ..., "arguments": {...}}}``.
    """

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]: ...


@runtime_checkable
class McpClient(Protocol):
    """An MCP client able to list and call tools on a server."""

    async def list_tools(self) -> list[dict[str, Any]]: ...

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> Any: ...


class ToolResult:
    """Outcome of a single tool-call step."""

    __slots__ = ("tool_name", "arguments", "ok", "result", "error")

    def __init__(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        ok: bool,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        self.tool_name = tool_name
        self.arguments = arguments
        self.ok = ok
        self.result = result
        self.error = error

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        status = "ok" if self.ok else f"error({self.error})"
        return f"ToolResult({self.tool_name!r}, {status})"
