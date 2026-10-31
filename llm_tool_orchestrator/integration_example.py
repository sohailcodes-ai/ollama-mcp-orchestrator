"""Integration example: bridging an application's ``chat_assistant`` flow to the
:class:`llm_tool_orchestrator.Orchestrator`.

This module reuses the existing orchestrator. It does not create a second
orchestration framework, does not invent tool names or business logic, and keeps
mutations disabled by default.

Application-specific wiring that cannot be inferred from this package is marked
``# TODO(application)``: the caller must supply a ``Model`` and an ``McpClient``
(and decide mutation authorization via its own policy).
"""

from __future__ import annotations

import asyncio
from typing import Any

from llm_tool_orchestrator import Orchestrator, OrchestratorConfig
from llm_tool_orchestrator.mcp_client import FastMcpAdapter
from llm_tool_orchestrator.models import Model


async def chat_assistant(
    is_admin: bool,
    data: dict[str, Any],
    model: Model | None = None,
    mcp_client: Any | None = None,
    *,
    allow_mutations: bool | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Example integration entry point mirroring ``chat_assistant(is_admin, data)``.

    Delegates to the existing :class:`Orchestrator`. Returns a dict with the
    final assistant text and the list of :class:`ToolResult` outcomes so the
    calling application can inspect which tools ran and whether they succeeded.

    Authorization policy:
      - Read-only tools are always available (on the explicit allowlist).
      - Mutation tools are disabled by default regardless of ``is_admin``. They
        become available only when the application explicitly opts in via
        ``allow_mutations=True``. ``is_admin`` alone is never treated by this
        layer as sufficient authorization for mutations unless the
        application's own policy establishes that guarantee.

    Parameters
    ----------
    is_admin:
        Admin flag from the calling application. Used only as one input to the
        application's authorization policy; not sufficient by itself for
        mutations.
    data:
        Original call payload. The prompt extraction is application-specific
        (schema unknown to this layer).
    model:
        Optional :class:`llm_tool_orchestrator.models.Model`. When omitted, an
        :class:`llm_tool_orchestrator.orchestrator.OllamaModel` is constructed
        against ``OLLAMA_HOST`` / ``OLLAMA_MODEL`` (defaults apply). Requires a
        running Ollama server with the model available -- not exercised in this
        checkout.
    mcp_client:
        Optional :class:`llm_tool_orchestrator.models.McpClient`. When omitted,
        the caller must still provide one; this example does not bundle an
        application-specific MCP server. Inject your own client here.
    allow_mutations:
        Explicit application-controlled override. When ``None`` (default),
        mutations stay disabled (read-only default policy). When ``True``, the
        application asserts its own policy permits mutation tools.
    max_steps:
        Override the bounded tool-calling step budget.

    Returns
    -------
    dict with keys ``text`` (final assistant response) and ``results`` (list of
    ToolResult).
    """
    # TODO(application): extract the actual user prompt from `data`. The shape of
    # `data` is application-specific and is not defined by this layer.
    user_prompt = data.get("prompt", "") if isinstance(data, dict) else str(data)

    # TODO(application): decide mutation authorization via application policy.
    # `is_admin` alone is not treated as sufficient authorization by this layer.
    if allow_mutations is None:
        allow_mutations = False

    config_kwargs: dict[str, Any] = {"allow_mutations": allow_mutations}
    if max_steps is not None:
        config_kwargs["max_steps"] = max_steps
    config = OrchestratorConfig(**config_kwargs)

    # TODO(application): provide your own MCP client (FastMcpAdapter wrapping
    # your server, or a remote adapter). This package does not bundle an
    # application-specific MCP server.
    if mcp_client is None:
        raise ValueError(
            "mcp_client is required: provide your own MCP client. "
            "This package does not bundle an application-specific server."
        )

    # TODO(application): supply a real Model (OllamaModel) when running against
    # a live Ollama server. A stub model is required for unit tests.
    if model is None:
        from llm_tool_orchestrator.orchestrator import OllamaModel

        model = OllamaModel(host=config.host, model=config.model)

    orch = Orchestrator(model, mcp_client, config)
    text, results = await orch.run_with_results(user_prompt)

    return {"text": text, "results": results}


def chat_assistant_sync(is_admin: bool, data: dict[str, Any]) -> dict[str, Any]:
    """Synchronous wrapper for call sites that cannot await.

    Runs ``chat_assistant`` on a fresh event loop. Prefer ``await
    chat_assistant(...)`` from an async context where possible to avoid
    blocking the event loop.
    """
    return asyncio.run(chat_assistant(is_admin, data))
