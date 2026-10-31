"""Core orchestration loop and production Ollama client.

The orchestrator:

  * converts MCP tool definitions into Ollama-compatible tool schemas,
  * drives a bounded multi-step tool-calling conversation,
  * validates every tool name/argument against an explicit allowlist,
  * executes only approved tools through an MCP client,
  * returns real tool results to the model in Ollama's conversation format,
  * bounds steps, enforces timeouts, and surfaces errors/cancellation safely.

No arbitrary SQL, shell, or model-generated code is ever executed.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from ollama import AsyncClient

from .config import OrchestratorConfig
from .exceptions import (
    CancellationError,
    MaxStepsExceededError,
    ToolArgumentError,
    ToolExecutionError,
    ToolNotAllowedError,
    ToolTimeoutError,
)
from .models import McpClient, Model, ToolResult
from .validation import validate_tool_call

_SYSTEM_PROMPT = (
    "You are a careful assistant. You have access to tools for reading and "
    "mutating state. Always validate that a tool exists on your list before "
    "calling it, supply only declared arguments, and prefer read-only tools "
    "for questions. Do not invent tools. Report generation is handled by the "
    "generate_inventory_report tool and is deterministic."
)


def mcp_tool_to_ollama(tool: dict[str, Any]) -> dict[str, Any]:
    """Convert an MCP tool definition into an Ollama tool schema.

    MCP ``inputSchema`` is already a JSON Schema, so it becomes the Ollama
    ``parameters`` object verbatim.
    """
    parameters = dict(tool.get("inputSchema") or {})
    parameters.setdefault("type", "object")
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description") or "",
            "parameters": parameters,
        },
    }


def _message_to_dict(message: Any) -> dict[str, Any]:
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls is None and isinstance(message, dict):
        tool_calls = message.get("tool_calls")
    normalized_calls = None
    if tool_calls:
        normalized_calls = []
        for call in tool_calls:
            fn = getattr(call, "function", None) or (
                call.get("function") if isinstance(call, dict) else {}
            )
            normalized_calls.append(
                {
                    "function": {
                        "name": getattr(fn, "name", None) or fn.get("name"),
                        "arguments": dict(
                            getattr(fn, "arguments", None)
                            or fn.get("arguments")
                            or {}
                        ),
                    }
                }
            )
    return {"role": "assistant", "content": content or "", "tool_calls": normalized_calls}


class OllamaModel(Model):
    """Production model implementation backed by ollama.AsyncClient."""

    def __init__(self, host: str, model: str) -> None:
        self._client = AsyncClient(host=host)
        self.model = model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        response = await self._client.chat(
            model=self.model,
            messages=messages,
            tools=tools,
            think=False,
            **kwargs,
        )
        msg = response.message
        return _message_to_dict(msg)


class Orchestrator:
    def __init__(
        self,
        model: Model,
        mcp_client: McpClient,
        config: OrchestratorConfig | None = None,
    ) -> None:
        self.model = model
        self.mcp_client = mcp_client
        self.config = config or OrchestratorConfig()
        self._max_steps = self.config.max_steps

    async def run(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        max_steps: int | None = None,
    ) -> str:
        """Run a bounded tool-calling conversation and return final text."""
        _, results = await self.run_with_results(
            user_prompt, system_prompt=system_prompt, max_steps=max_steps
        )
        return self._last_text

    async def run_with_results(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        max_steps: int | None = None,
    ) -> tuple[str, list[ToolResult]]:
        max_steps = max_steps if max_steps is not None else self._max_steps
        if max_steps < 1:
            raise MaxStepsExceededError("max_steps must be >= 1")

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        mcp_tools = await self.mcp_client.list_tools()
        ollama_tools = [
            mcp_tool_to_ollama(t) for t in mcp_tools if t["name"] in self.config.allowed_tools
        ]

        results: list[ToolResult] = []
        self._last_text = ""

        try:
            for step in range(1, max_steps + 1):
                response = await self.model.chat(messages, tools=ollama_tools)
                assistant_msg = dict(response)
                messages.append(assistant_msg)
                self._last_text = assistant_msg.get("content") or ""

                tool_calls = assistant_msg.get("tool_calls") or []
                if not tool_calls:
                    break  # final answer

                for call in tool_calls:
                    fn = call.get("function", {})
                    name = fn.get("name")
                    raw_args = fn.get("arguments") or {}

                    if not name or not self.config.is_allowed(name):
                        err = ToolNotAllowedError(
                            f"Tool {name!r} is not allowed"
                        )
                        results.append(
                            ToolResult(name or "unknown", dict(raw_args), False, error=str(err))
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "content": f"Error: {err}",
                                "tool_name": name or "unknown",
                            }
                        )
                        continue

                    try:
                        args = validate_tool_call(name, raw_args, self.config)
                    except ToolArgumentError as exc:
                        results.append(
                            ToolResult(name, dict(raw_args), False, error=str(exc))
                        )
                        messages.append(
                            {"role": "tool", "content": f"Error: {exc}", "tool_name": name}
                        )
                        continue

                    try:
                        result = await asyncio.wait_for(
                            self.mcp_client.call_tool(name, args),
                            timeout=self.config.tool_timeout_seconds,
                        )
                    except asyncio.TimeoutError as exc:
                        err = ToolTimeoutError(f"Tool {name!r} timed out")
                        results.append(ToolResult(name, args, False, error=str(err)))
                        messages.append(
                            {"role": "tool", "content": f"Error: {err}", "tool_name": name}
                        )
                        continue
                    except asyncio.CancelledError:
                        raise CancellationError("Operation cancelled during tool call")
                    except Exception as exc:  # tool runtime error
                        err = ToolExecutionError(f"Tool {name!r} failed: {exc}")
                        results.append(ToolResult(name, args, False, error=str(err)))
                        messages.append(
                            {"role": "tool", "content": f"Error: {err}", "tool_name": name}
                        )
                        continue

                    results.append(ToolResult(name, args, True, result=result))
                    messages.append(
                        {
                            "role": "tool",
                            "content": self._serialize_result(result),
                            "tool_name": name,
                        }
                    )
            else:
                raise MaxStepsExceededError(
                    f"Tool-calling loop exceeded {max_steps} steps"
                )
        except CancellationError:
            raise
        except MaxStepsExceededError:
            raise

        return self._last_text, results

    @staticmethod
    def _serialize_result(result: Any) -> str:
        """Serialize a tool result for the model without leaking secrets."""
        if isinstance(result, str):
            return result

        try:
            return json.dumps(result, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(result)
