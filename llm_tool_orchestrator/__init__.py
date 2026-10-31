"""Reusable Ollama + MCP tool-calling orchestration layer.

This package is intentionally decoupled from any specific application. It
provides:

* :class:`Orchestrator` — bounded multi-round tool-calling loop against Ollama.
* :class:`OllamaModel` — async Ollama inference client.
* :class:`FastMcpAdapter` — FastMCP client adapter.
* :class:`OrchestratorConfig` — explicit tool allowlist, schemas, timeouts,
  and authorization policy (read-only by default; mutations gated behind
  explicit application opt-in).
* :func:`validate_tool_call` — strict allowlist + JSON-schema validation.

No arbitrary SQL, shell, or model-generated code is ever executed. The
application must supply its own MCP server/tools and model instance.
"""

from __future__ import annotations

from .config import OrchestratorConfig
from .exceptions import (
    CancellationError,
    MaxStepsExceededError,
    OrchestratorError,
    ToolArgumentError,
    ToolExecutionError,
    ToolNotAllowedError,
    ToolTimeoutError,
)
from .models import McpClient, Model, ToolResult
from .orchestrator import Orchestrator, OllamaModel, mcp_tool_to_ollama
from .validation import validate_tool_call

__all__ = [
    "Orchestrator",
    "OllamaModel",
    "ToolResult",
    "OrchestratorConfig",
    "Model",
    "McpClient",
    "mcp_tool_to_ollama",
    "validate_tool_call",
    "OrchestratorError",
    "ToolNotAllowedError",
    "ToolArgumentError",
    "ToolExecutionError",
    "ToolTimeoutError",
    "MaxStepsExceededError",
    "CancellationError",
]
