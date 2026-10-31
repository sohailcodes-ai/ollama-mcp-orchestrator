"""Allowlist and schema validation for model-requested tool calls."""

from __future__ import annotations

from typing import Any

import jsonschema

from .config import OrchestratorConfig
from .exceptions import ToolArgumentError, ToolNotAllowedError


def validate_tool_call(
    name: str,
    arguments: dict[str, Any] | None,
    config: OrchestratorConfig,
) -> dict[str, Any]:
    """Validate a model-requested tool call against the explicit allowlist.

    Returns the normalized arguments (defaults applied) on success. Raises
    :class:`ToolNotAllowedError` or :class:`ToolArgumentError` otherwise.
    No tool is executed here; this is a pure guard.
    """
    arguments = dict(arguments or {})

    if not config.is_allowed(name):
        raise ToolNotAllowedError(f"Tool {name!r} is not on the explicit allowlist")

    schema = config.schema_for(name)
    if not schema:
        raise ToolArgumentError(f"No schema registered for allowed tool {name!r}")

    # Coerce types declared in the schema (int/float) where the model sent
    # a numeric string; reject anything else that violates the schema.
    _coerce_arguments(arguments, schema)

    try:
        jsonschema.validate(instance=arguments, schema=schema)
    except jsonschema.ValidationError as exc:
        raise ToolArgumentError(
            f"Invalid arguments for {name!r}: {exc.message}"
        ) from exc

    return arguments


def _coerce_arguments(arguments: dict[str, Any], schema: dict[str, Any]) -> None:
    """Best-effort type coercion for numeric fields (in-place).

    LLMs sometimes emit numbers as strings. We coerce only when the schema
    declares a numeric type and the value is a numeric string; otherwise we
    leave the value for jsonschema to accept or reject.
    """
    properties = schema.get("properties", {})
    for key, value in list(arguments.items()):
        prop = properties.get(key)
        if not isinstance(prop, dict) or not isinstance(value, str):
            continue
        expected = prop.get("type")
        if expected == "integer":
            try:
                arguments[key] = int(value)
            except (TypeError, ValueError):
                pass
        elif expected == "number":
            try:
                arguments[key] = float(value)
            except (TypeError, ValueError):
                pass
