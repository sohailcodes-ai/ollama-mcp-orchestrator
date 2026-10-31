"""Configuration and explicit tool allowlist for the LLM orchestration layer.

All values are readable from environment variables so deployment can be tuned
without editing code. The allowlist and schemas below are the *source of
truth* for what the model may invoke; they are cross-checked against the MCP
tool definitions before any execution.

Authorization policy:
  * Read-only tools are always allowed if on the explicit allowlist.
  * Mutation tools are disabled by default (read-only default policy). They
    become executable only when the application explicitly sets
    ``allow_mutations=True`` on the config. ``is_admin`` or any other
    untrusted input is never treated as sufficient authorization by this
    layer alone.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Final, Mapping

OLLAMA_HOST: Final[str] = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL: Final[str] = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

MAX_STEPS: Final[int] = int(os.environ.get("LLM_MAX_STEPS", "5"))
TOOL_TIMEOUT_SECONDS: Final[float] = float(
    os.environ.get("LLM_TOOL_TIMEOUT_SECONDS", "30.0")
)

# Read-only tools never modify inventory/application state.
READ_ONLY_TOOLS: Final[frozenset[str]] = frozenset(
    {
        "get_all_items",
        "get_item_by_id",
        "get_items_by_type",
        "generate_inventory_report",
    }
)

# Mutation tools change state and are kept logically separate. Disabled by
# default; enabled only via allow_mutations=True.
MUTATION_TOOLS: Final[frozenset[str]] = frozenset(
    {
        "add_item",
        "update_item",
        "delete_item",
    }
)

ALLOWED_TOOLS: Final[frozenset[str]] = READ_ONLY_TOOLS | MUTATION_TOOLS


def _schema(
    properties: Mapping[str, dict],
    required: list[str] | None = None,
    additional_properties: bool = False,
) -> dict:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": required or [],
        "additionalProperties": additional_properties,
    }


# Explicit JSON-schema allowlist. This is the contract used to validate
# model-supplied arguments before a tool is executed.
TOOL_SCHEMAS: Final[dict[str, dict]] = {
    "get_all_items": _schema({}, additional_properties=False),
    "get_item_by_id": _schema(
        {
            "item_id": {"type": "integer", "description": "Item ID."},
        },
        required=["item_id"],
    ),
    "get_items_by_type": _schema(
        {
            "item_type": {"type": "string", "description": "Item type/category."},
        },
        required=["item_type"],
    ),
    "generate_inventory_report": _schema({}, additional_properties=False),
    "add_item": _schema(
        {
            "name": {"type": "string", "description": "Item name."},
            "price": {"type": "number", "description": "Item price."},
            "quantity": {"type": "integer", "description": "Item quantity."},
            "item_type": {"type": "string", "description": "Item type/category."},
        },
        required=["name", "price", "quantity", "item_type"],
    ),
    "update_item": _schema(
        {
            "item_id": {"type": "integer", "description": "Item ID."},
            "price": {"type": "number", "description": "New price (optional)."},
            "quantity": {"type": "integer", "description": "New quantity (optional)."},
        },
        required=["item_id"],
    ),
    "delete_item": _schema(
        {
            "item_id": {"type": "integer", "description": "Item ID to delete."},
        },
        required=["item_id"],
    ),
}


@dataclass(frozen=True)
class OrchestratorConfig:
    host: str = OLLAMA_HOST
    model: str = OLLAMA_MODEL
    max_steps: int = MAX_STEPS
    tool_timeout_seconds: float = TOOL_TIMEOUT_SECONDS
    read_only_tools: frozenset[str] = field(default_factory=lambda: READ_ONLY_TOOLS)
    mutation_tools: frozenset[str] = field(default_factory=lambda: MUTATION_TOOLS)
    allowed_tools: frozenset[str] = field(default_factory=lambda: ALLOWED_TOOLS)
    tool_schemas: dict[str, dict] = field(default_factory=lambda: dict(TOOL_SCHEMAS))
    allow_mutations: bool = False

    def is_read_only(self, name: str) -> bool:
        return name in self.read_only_tools

    def is_mutation(self, name: str) -> bool:
        return name in self.mutation_tools

    def is_allowed(self, name: str) -> bool:
        """Application-controlled policy.

        Mutation tools are disabled by default regardless of any admin flag.
        A mutation tool is only allowed when the application explicitly sets
        ``allow_mutations=True`` on this config. Read-only tools are always
        allowed if on the allowlist.
        """
        if name in self.mutation_tools and not self.allow_mutations:
            return False
        return name in self.allowed_tools

    def schema_for(self, name: str) -> dict:
        return self.tool_schemas.get(name, {})
