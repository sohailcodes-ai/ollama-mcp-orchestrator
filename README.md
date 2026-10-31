# ollama-mcp-orchestrator

A reusable Python orchestration layer for reliable Ollama tool calling with MCP,
strict schema validation, bounded multi-round execution, and authorization
controls.

## Installation

```bash
# Create a virtual environment
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# Install runtime dependencies
pip install -r requirements.txt

# Install development/test dependencies
pip install -r requirements-dev.txt
```

## Supported versions

- Python: >= 3.9
- ollama: >= 0.6.3, < 0.7.0
- fastmcp: >= 4.1.3, < 5.0.0
- jsonschema: >= 4.26.0, < 5.0.0
- pytest: >= 8.0.0 (dev)

## Quick start

```python
import asyncio
from llm_tool_orchestrator import Orchestrator, OrchestratorConfig
from llm_tool_orchestrator.mcp_client import FastMcpAdapter
from llm_tool_orchestrator.models import Model

# 1. Provide your own MCP server (FastMCP) and client.
#    This package does not bundle an application-specific MCP server.
from my_app.mcp_server import server  # your FastMCP instance

# 2. Provide your own model (stub for tests; OllamaModel for live use).
class MyModel(Model):
    async def chat(self, messages, tools=None, **kwargs):
        ...

# 3. Run the orchestrator.
async def main():
    async with FastMcpAdapter(server) as mcp:
        model = MyModel()
        config = OrchestratorConfig(max_steps=5)  # read-only by default
        orch = Orchestrator(model, mcp, config)
        text, results = await orch.run_with_results("List all items.")
        print(text)

asyncio.run(main())
```

For live Ollama inference, use the built-in `OllamaModel`:

```python
from llm_tool_orchestrator.orchestrator import OllamaModel

model = OllamaModel(host="http://localhost:11434", model="llama3.2:3b")
```

This requires a running Ollama server with the model pulled:

```bash
ollama pull llama3.2:3b
```

## Tool registration and schemas

MCP tool definitions are converted to Ollama function schemas via
`mcp_tool_to_ollama()`. The MCP `inputSchema` becomes the Ollama
`parameters` object verbatim. Your MCP server must declare a valid
`inputSchema` (JSON Schema) for each tool.

Example:

```python
from fastmcp import FastMCP

server = FastMCP(name="my-tools")

@server.tool
def get_item_by_id(item_id: int) -> str:
    """Get an item by ID."""
    ...
```

## Authorization policy

The orchestrator enforces an explicit allowlist plus a **read-only default
policy**:

- Read-only tools are always allowed if on the explicit allowlist.
- Mutation tools (`add_item`, `update_item`, `delete_item`) are **disabled by
  default**. They become executable only when the application explicitly sets
  `allow_mutations=True` on `OrchestratorConfig`.

`is_admin` or any other untrusted user input is **never** treated as
sufficient authorization by this layer alone. The application must make its
own authorization decision and pass `allow_mutations=True` only when its
policy permits it.

```python
# Mutations stay disabled (default):
config = OrchestratorConfig()

# Mutations enabled only when application policy explicitly permits:
config = OrchestratorConfig(allow_mutations=True)
```

The allowlist and JSON-schema contracts live in `llm_tool_orchestrator/config.py`
and are the source of truth. Every model-supplied tool name and argument is
validated against them before execution; unknown tools and invalid arguments
are rejected and never invoked.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2:3b` | Model to use |
| `LLM_MAX_STEPS` | `5` | Max tool-calling conversation steps |
| `LLM_TOOL_TIMEOUT_SECONDS` | `30.0` | Per-tool-call timeout |

Or set via `OrchestratorConfig(...)` directly.

## Error handling, timeouts, and cancellation

- Per-tool-call timeouts are enforced via `asyncio.wait_for`.
- Tool runtime errors are caught and returned as `ToolResult(ok=False, ...)`
  so the conversation can continue.
- `asyncio.CancelledError` during a tool call raises `CancellationError`.
- Exhausting the step budget raises `MaxStepsExceededError`.

## Integration example

`llm_tool_orchestrator.integration_example` provides `chat_assistant()`,
showing how an application's `chat_assistant(is_admin, data)` flow should
delegate to the orchestrator:

```python
from llm_tool_orchestrator.integration_example import chat_assistant

out = asyncio.run(chat_assistant(
    is_admin=False,
    data={"prompt": "List all items."},
    model=my_model,
    mcp_client=my_mcp_client,
))
print(out["text"])
print(out["results"])
```

The caller must supply `model` and `mcp_client`; this package does not bundle
an application-specific MCP server or model. See `examples/basic_usage.py`
for a complete working example with a stub model and in-memory MCP server.

## Tests

```bash
# From the repository root
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

## Examples

- `examples/mcp_server.py` — standalone example MCP server (in-memory state,
  no database dependency).
- `examples/basic_usage.py` — basic usage with stub model and example server.

## Limitations

- No live Ollama server or `llama3.2:3b` model is exercised in the test suite.
  Tests use a stub model and an in-process FastMCP server. Live-model
  tool-calling reliability has **not** been verified.
- The allowlist, schema validation, bounded steps, and read-only default
  reduce risk but do **not** eliminate model hallucinations. Do not claim
  hallucinations are eliminated or live-model reliability is proven.
- This package does not bundle an application-specific MCP server, model, or
  business logic. The application must provide these.
- No arbitrary SQL, shell commands, or model-generated code is ever executed.

## Security audit

- No API keys, tokens, passwords, or credentials are included in this project.
- No application database files, user data, or proprietary application code
  are included.
- Virtual environments, caches, and generated artifacts are excluded via
  `.gitignore`.
- Mutation tools are disabled by default and cannot be enabled by model output
  or untrusted user input; only the application can set `allow_mutations=True`.
