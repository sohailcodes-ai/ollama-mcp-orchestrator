"""Example MCP server for the ollama-mcp-orchestrator.

This is a standalone, self-contained example. It does not depend on any
application database or proprietary business logic. Tools use in-memory state
for demonstration purposes only.

Replace this with your own MCP server when integrating into your application.
"""

from __future__ import annotations

import json
from typing import Any

from fastmcp import FastMCP

server = FastMCP(
    name="example-tools",
    instructions=(
        "Example tools for demonstrating the ollama-mcp-orchestrator. "
        "Read-only tools never change state; mutation tools add, update or "
        "delete items. All arguments must match the declared schemas."
    ),
)

# In-memory example state (demonstration only).
_ITEMS: dict[int, dict[str, Any]] = {}
_next_id = 1


def _next_item_id() -> int:
    global _next_id
    item_id = _next_id
    _next_id += 1
    return item_id


@server.tool
def get_all_items() -> str:
    """List all items (read-only)."""
    return json.dumps(list(_ITEMS.values()))


@server.tool
def get_item_by_id(item_id: int) -> str:
    """Get an item by its ID (read-only)."""
    item = _ITEMS.get(item_id)
    return json.dumps(item if item is not None else {"error": "not found"})


@server.tool
def get_items_by_type(item_type: str) -> str:
    """List items by type/category (read-only)."""
    matches = [
        it for it in _ITEMS.values() if it.get("type") == item_type
    ]
    return json.dumps(matches)


@server.tool
def generate_inventory_report() -> str:
    """Generate a deterministic text report (read-only)."""
    items = sorted(_ITEMS.values(), key=lambda it: it.get("id", 0))
    lines = ["EXAMPLE REPORT", "=" * 20, f"Total items: {len(items)}", ""]
    for it in items:
        lines.append(
            f"#{it.get('id')} | {it.get('name')} | {it.get('type')} | "
            f"price={it.get('price')} | qty={it.get('quantity')}"
        )
    lines.append("=" * 20)
    return "\n".join(lines)


@server.tool
def add_item(name: str, price: float, quantity: int, item_type: str) -> str:
    """Add a new item (mutation)."""
    global _next_id
    item_id = _next_id
    _next_id += 1
    _ITEMS[item_id] = {
        "id": item_id,
        "name": name,
        "price": price,
        "quantity": quantity,
        "type": item_type,
    }
    return f"Added item {item_id} successfully"


@server.tool
def update_item(item_id: int, price: float | None = None, quantity: int | None = None) -> str:
    """Update an item's price and/or quantity (mutation)."""
    if item_id not in _ITEMS:
        return "Item not found"
    item = _ITEMS[item_id]
    if price is not None:
        item["price"] = price
    if quantity is not None:
        item["quantity"] = quantity
    return f"Updated item {item_id} successfully"


@server.tool
def delete_item(item_id: int) -> str:
    """Delete an item by ID (mutation)."""
    if item_id not in _ITEMS:
        return "Item not found"
    del _ITEMS[item_id]
    return f"Deleted item {item_id} successfully"


if __name__ == "__main__":
    server.run()
