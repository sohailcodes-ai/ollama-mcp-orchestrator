# ollama-mcp-orchestrator

A Python library for connecting Ollama models to MCP tools.

It handles tool discovery, argument validation, execution, and returning tool results to the model. It also supports multiple tool-calling rounds, execution limits, and application-controlled authorization.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For development and testing:

```bash
pip install -r requirements-dev.txt
```

## Requirements

| Dependency | Supported version |
|---|---|
| Python | >= 3.9 |
| ollama | >= 0.6.3, < 0.7.0 |
| fastmcp | >= 4.1.3, < 5.0.0 |
| jsonschema | >= 4.26.0, < 5.0.0 |
| pytest | >= 8.0.0 (development) |

## How it works

The orchestrator sits between an Ollama model and an MCP client.

1. It retrieves the available MCP tools.
2. Tool definitions are converted into Ollama function schemas.
3. The model can request one or more tool calls.
4. Tool names and arguments are checked against the configured allowlist and JSON Schemas.
5. Approved calls are executed through MCP.
6. Results are returned to the model, which can request further tools or produce a final response.

Execution is bounded by a configurable step limit.

## Quick start

The application supplies its own MCP server and model implementation.

```python
import asyncio

from llm_tool_orchestrator import Orchestrator, OrchestratorConfig
from llm_tool_orchestrator.mcp_client import FastMcpAdapter
from llm_tool_orchestrator.orchestrator import OllamaModel

from my_app.mcp_server import server


async def main():
    model = OllamaModel(
        host="http://localhost:11434",
        model="llama3.2:3b",
    )

    config = OrchestratorConfig(max_steps=5)

    async with FastMcpAdapter(server) as mcp:
        orchestrator = Orchestrator(model, mcp, config)

        response, results = await orchestrator.run_with_results(
            "List all items."
        )

        print(response)


if __name__ == "__main__":
    asyncio.run(main())
```

Start Ollama and download the model before running the example:

```bash
ollama pull llama3.2:3b
```

The MCP server in this example is application-provided. The package does not include your database, inventory models, or business logic.

## Tool schemas

MCP tool definitions are converted into Ollama-compatible function definitions using `mcp_tool_to_ollama()`.

The MCP tool's `inputSchema` is used as the function's parameter schema. Tools should declare valid JSON Schemas, including required fields and argument types.

Example:

```python
from fastmcp import FastMCP

server = FastMCP(name="inventory-tools")


@server.tool
def get_item_by_id(item_id: int) -> str:
    """Get an inventory item by ID."""
    ...
```

The orchestrator checks model-supplied tool names and arguments before execution. Unknown tools and invalid arguments are rejected rather than passed to the MCP server.

The allowlist and schema definitions are maintained in:

```text
llm_tool_orchestrator/config.py
```

## Authorization

Tool access is controlled by an explicit allowlist and a read-only default policy.

- Read-only tools can run when they are included in the allowlist.
- Mutation tools are disabled by default.
- Mutations require the application to explicitly enable `allow_mutations=True`.

For example:

```python
# Default: mutations disabled
config = OrchestratorConfig()

# Enable only after the application has authorized it
config = OrchestratorConfig(allow_mutations=True)
```

The application is responsible for deciding whether a user is allowed to perform mutations. A model-generated tool call or a user-supplied `is_admin` value is not authorization by itself.

## Configuration

The orchestrator can be configured through `OrchestratorConfig`.

| Setting | Default | Purpose |
|---|---:|---|
| `max_steps` | `5` | Maximum tool-calling steps |
| `allow_mutations` | `False` | Whether mutation tools can execute |
| `tool_timeout_seconds` | `30.0` | Timeout for an individual tool call |

The Ollama model and host are configured through `OllamaModel`:

```python
model = OllamaModel(
    host="http://localhost:11434",
    model="llama3.2:3b",
)
```

## Errors and execution limits

The orchestrator handles tool execution failures and execution limits.

- Tool runtime failures are represented as `ToolResult(ok=False, ...)`.
- Tool calls have a configurable timeout.
- Cancellation during tool execution is handled by the orchestrator.
- Reaching the configured step limit raises `MaxStepsExceededError`.

See the implementation and tests for the exact exception behavior.

## Integrating with an existing application

The package includes an integration example showing how an existing `chat_assistant()` function can delegate tool orchestration:

```python
from llm_tool_orchestrator.integration_example import chat_assistant

out = await chat_assistant(
    is_admin=False,
    data={"prompt": "List all items."},
    model=my_model,
    mcp_client=my_mcp_client,
)

print(out["text"])
print(out["results"])
```

The caller supplies the model and MCP client. Application-specific authentication, database access, and business logic remain outside this package.

See `examples/basic_usage.py` for a runnable example using a stub model and an in-memory MCP server.

## Tests

Run the test suite from the repository root:

```bash
python -m pytest tests/ -v
```

The tests cover tool validation, execution, multiple tool calls, execution limits, authorization, and error handling using a stub model and an in-process MCP server.

## Project structure

```text
llm_tool_orchestrator/
    config.py
    mcp_client.py
    models.py
    orchestrator.py
    integration_example.py

examples/
    basic_usage.py
    mcp_server.py

tests/
```

## Limitations

The test suite does not run a live Ollama model. It uses a stub model and an in-process MCP server.

Consequently, live tool-calling behavior with `llama3.2:3b` has not been verified. The orchestrator validates and controls tool execution, but it cannot guarantee that a model will always choose the correct tool or produce a correct answer.

The package also does not provide an application-specific MCP server, database, or business logic. Those must be supplied by the integrating application.

Model-generated tool arguments are validated before execution. The package does not execute arbitrary SQL, shell commands, or generated code.

## License

See `LICENSE` for licensing terms.
