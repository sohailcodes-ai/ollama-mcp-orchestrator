"""FastMCP client adapter implementing the :class:`McpClient` protocol.

Supports both in-process FastMCP servers (for tests) and remote SSE/HTTP
servers (for production use).
"""

from __future__ import annotations

from typing import Any

from fastmcp import Client

from .models import McpClient


class FastMcpAdapter:
    """Wraps ``fastmcp.Client`` to expose list/call tool operations."""

    def __init__(self, server_or_url) -> None:
        self._server_or_url = server_or_url
        self._client: Client | None = None

    async def __aenter__(self) -> "FastMcpAdapter":
        self._client = Client(self._server_or_url)
        await self._client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client is not None:
            await self._client.__aexit__(exc_type, exc, tb)
        self._client = None

    @property
    def client(self) -> Client:
        if self._client is None:
            raise RuntimeError("FastMcpAdapter is not connected; use async with")
        return self._client

    async def list_tools(self) -> list[dict[str, Any]]:
        raw = await self.client.list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "inputSchema": (
                    getattr(tool, "input_schema", None)
                    or getattr(tool, "inputSchema", None)
                    or {}
                ),
            }
            for tool in raw
        ]

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> Any:
        result = await self.client.call_tool(name, arguments or {})
        return _extract_content(result)


def _extract_content(result: Any) -> Any:
    """Best-effort extraction of usable tool content from a CallToolResult.

    Text content parts are preferred because FastMCP wraps plain string
    returns in ``structured_content = {"result": <string>}``, which is an
    awkward wrapper for callers.
    """
    content = getattr(result, "content", None)
    if content:
        parts = []
        for item in content:
            text = getattr(item, "text", None)
            if text is not None:
                parts.append(text)
            else:
                parts.append(str(item))
        if parts:
            return "\n".join(parts)
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        return structured
    return result
