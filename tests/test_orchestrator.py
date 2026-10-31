"""Unit tests for the orchestration layer.

All model and MCP-client interactions are mocked; no Ollama server, MCP
server, or database is required. No application code is imported.
"""

import asyncio

import pytest

from llm_tool_orchestrator import Orchestrator, OrchestratorConfig, ToolResult
from llm_tool_orchestrator.exceptions import (
    CancellationError,
    MaxStepsExceededError,
    ToolArgumentError,
    ToolNotAllowedError,
)
from llm_tool_orchestrator.integration_example import chat_assistant
from llm_tool_orchestrator.models import McpClient, Model
from llm_tool_orchestrator.orchestrator import mcp_tool_to_ollama
from llm_tool_orchestrator.validation import validate_tool_call


class FakeModel(Model):
    """Scripted model: returns a list of canned chat responses in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def chat(self, messages, tools=None, **kwargs):  # noqa: ARG002
        self.calls.append((list(messages), tools))
        if not self._responses:
            return {"role": "assistant", "content": "(no more scripted responses)"}
        return self._responses.pop(0)


class FakeMcpClient(McpClient):
    """Scripted MCP client with controllable tool results/errors."""

    def __init__(self, tools=None, call_results=None):
        self._tools = list(tools or [])
        self._call_results = dict(call_results or {})
        self.calls = []

    async def list_tools(self):
        return list(self._tools)

    async def call_tool(self, name, arguments=None):
        self.calls.append((name, dict(arguments or {})))
        if name in self._call_results:
            outcome = self._call_results[name]
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        return f"result-for-{name}"


def _tool_def(name, properties=None, required=None):
    return {
        "name": name,
        "description": f"{name} tool",
        "inputSchema": {
            "type": "object",
            "properties": properties or {},
            "required": required or [],
            "additionalProperties": False,
        },
    }


def _tool_call(name, arguments):
    return {"function": {"name": name, "arguments": arguments}}


def _assistant(tool_calls=None, content=""):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = [_tool_call(n, a) for n, a in tool_calls]
    return msg


def test_model_final_answer_without_tools():
    config = OrchestratorConfig(max_steps=3)
    model = FakeModel([{"role": "assistant", "content": "Hello world"}])
    mcp = FakeMcpClient(tools=[_tool_def("get_all_items")])
    orch = Orchestrator(model, mcp, config)
    text, results = asyncio.run(orch.run_with_results("say hi"))
    assert text == "Hello world"
    assert results == []
    assert len(model.calls) == 1


def test_read_only_tool_executed_and_result_returned():
    config = OrchestratorConfig(max_steps=3)
    model = FakeModel([
        _assistant([("get_all_items", {})], content=""),
        {"role": "assistant", "content": "There are 2 products."},
    ])
    mcp = FakeMcpClient(
        tools=[_tool_def("get_all_items")],
        call_results={"get_all_items": [{"id": 1, "name": "widget"}]},
    )
    orch = Orchestrator(model, mcp, config)
    text, results = asyncio.run(orch.run_with_results("list products"))
    assert "There are 2 products." in text
    assert len(results) == 1
    r = results[0]
    assert r.tool_name == "get_all_items"
    assert r.ok is True
    assert r.result == [{"id": 1, "name": "widget"}]
    assert any(m.get("role") == "tool" for m in model.calls[1][0])


def test_blocked_tool_is_not_executed():
    config = OrchestratorConfig(max_steps=3)
    model = FakeModel([
        _assistant([("evil_tool", {})], content=""),
        {"role": "assistant", "content": "I cannot do that."},
    ])
    mcp = FakeMcpClient(tools=[])
    orch = Orchestrator(model, mcp, config)
    text, results = asyncio.run(orch.run_with_results("do evil"))
    assert "I cannot do that." in text
    assert len(results) == 1
    assert results[0].ok is False
    assert isinstance(results[0].error, str) and "not allowed" in results[0].error
    assert mcp.calls == []


def test_invalid_arguments_are_not_executed():
    config = OrchestratorConfig(max_steps=3)
    model = FakeModel([
        _assistant([("add_item", {"name": "w", "price": "not-a-number"})], content=""),
        {"role": "assistant", "content": "Invalid price."},
    ])
    mcp = FakeMcpClient(
        tools=[_tool_def("add_item", {
            "name": {"type": "string"},
            "price": {"type": "number"},
        }, required=["name", "price"])],
    )
    orch = Orchestrator(model, mcp, config)
    text, results = asyncio.run(orch.run_with_results("add"))
    assert "Invalid price." in text
    assert len(results) == 1
    assert results[0].ok is False
    assert isinstance(results[0].error, str)
    assert mcp.calls == []


def test_tool_runtime_error_is_surfaced_and_conversation_continues():
    runtime = ValueError("db is locked")
    config = OrchestratorConfig(max_steps=3)
    model = FakeModel([
        _assistant([("get_all_items", {})], content=""),
        {"role": "assistant", "content": "Retry failed."},
    ])
    mcp = FakeMcpClient(
        tools=[_tool_def("get_all_items")],
        call_results={"get_all_items": runtime},
    )
    orch = Orchestrator(model, mcp, config)
    text, results = asyncio.run(orch.run_with_results("list"))
    assert "Retry failed." in text
    assert results[0].ok is False
    assert "db is locked" in results[0].error


def test_tool_timeout_is_handled():
    config = OrchestratorConfig(max_steps=2, tool_timeout_seconds=0.01)

    async def slow_call(name, arguments=None):  # noqa: ARG001
        await asyncio.sleep(10)
        return "too late"

    class SlowMcp(McpClient):
        async def list_tools(self):
            return [_tool_def("get_all_items")]
        async def call_tool(self, name, arguments=None):
            await slow_call(name, arguments)

    model = FakeModel([_assistant([("get_all_items", {})], content="")])
    orch = Orchestrator(model, SlowMcp(), config)
    text, results = asyncio.run(orch.run_with_results("list"))
    assert results[0].ok is False
    assert "timed out" in results[0].error


def test_cancellation_propagates():
    config = OrchestratorConfig(max_steps=2)

    class CancellingMcp(McpClient):
        async def list_tools(self):
            return [_tool_def("get_all_items")]
        async def call_tool(self, name, arguments=None):
            await asyncio.sleep(10)

    model = FakeModel([_assistant([("get_all_items", {})], content="")])

    async def main():
        orch = Orchestrator(model, CancellingMcp(), config)
        task = asyncio.create_task(orch.run_with_results("list"))
        await asyncio.sleep(0.01)
        task.cancel()
        try:
            await task
        except CancellationError:
            return "cancelled"
        return "not-cancelled"

    assert asyncio.run(main()) == "cancelled"


def test_max_steps_exceeded():
    config = OrchestratorConfig(max_steps=1)
    model = FakeModel([
        _assistant([("get_all_items", {})], content=""),
    ])
    mcp = FakeMcpClient(
        tools=[_tool_def("get_all_items")],
        call_results={"get_all_items": []},
    )
    orch = Orchestrator(model, mcp, config)
    with pytest.raises(MaxStepsExceededError):
        asyncio.run(orch.run_with_results("list"))


def test_mcp_tool_to_ollama_shape():
    ollama_tool = mcp_tool_to_ollama(_tool_def("get_item_by_id", {
        "item_id": {"type": "integer"},
    }, required=["item_id"]))
    assert ollama_tool["type"] == "function"
    assert ollama_tool["function"]["name"] == "get_item_by_id"
    assert ollama_tool["function"]["parameters"]["type"] == "object"
    assert "item_id" in ollama_tool["function"]["parameters"]["properties"]


def test_validation_allowlist_and_schema():
    config = OrchestratorConfig()
    assert config.is_allowed("get_all_items") is True
    assert config.is_read_only("get_all_items") is True
    assert config.is_mutation("add_item") is True
    assert config.is_allowed("add_item") is False  # read-only default
    assert config.is_allowed("evil") is False

    auth_config = OrchestratorConfig(allow_mutations=True)
    assert auth_config.is_allowed("add_item") is True  # explicitly authorized

    args = validate_tool_call("add_item", {
        "name": "w", "price": "1.5", "quantity": "3", "item_type": "t",
    }, auth_config)
    assert args["price"] == 1.5
    assert args["quantity"] == 3

    with pytest.raises(ToolNotAllowedError):
        validate_tool_call("evil", {}, config)
    with pytest.raises(ToolNotAllowedError):
        validate_tool_call("add_item", {}, config)  # blocked by default policy
    with pytest.raises(ToolArgumentError):
        validate_tool_call("add_item", {"name": "w"}, auth_config)


def test_result_serialization_non_string():
    from llm_tool_orchestrator.orchestrator import Orchestrator
    assert Orchestrator._serialize_result({"a": 1}) == '{"a": 1}'
    assert Orchestrator._serialize_result("plain") == "plain"


def test_tool_result_fields():
    r = ToolResult("add_item", {"name": "w"}, True, result="ok")
    assert r.tool_name == "add_item"
    assert r.ok is True
    assert r.result == "ok"
    assert r.error is None
    err = ToolResult("delete_item", {"id": 1}, False, error="boom")
    assert err.ok is False
    assert err.error == "boom"


def test_incorrect_argument_types_rejected_by_validation():
    config = OrchestratorConfig()
    auth_config = OrchestratorConfig(allow_mutations=True)
    with pytest.raises(ToolArgumentError):
        validate_tool_call("get_item_by_id", {"item_id": "not-an-id"}, config)
    with pytest.raises(ToolArgumentError):
        validate_tool_call("add_item", {
            "name": "w", "price": "not-a-number",
            "quantity": "3", "item_type": "t",
        }, auth_config)
    with pytest.raises(ToolArgumentError):
        validate_tool_call("add_item", {
            "name": "w", "price": 1.5, "quantity": "not-a-qty",
            "item_type": "t",
        }, auth_config)
    with pytest.raises(ToolNotAllowedError):
        validate_tool_call("add_item", {"name": "w", "price": 1, "quantity": 1,
                                        "item_type": "t"}, config)


def test_incorrect_argument_types_not_executed_by_orchestrator():
    config = OrchestratorConfig(max_steps=3)
    model = FakeModel([
        _assistant([("get_item_by_id", {"item_id": "abc"})], content=""),
        {"role": "assistant", "content": "Bad id."},
    ])
    mcp = FakeMcpClient(tools=[_tool_def("get_item_by_id", {
        "item_id": {"type": "integer"},
    }, required=["item_id"])])
    orch = Orchestrator(model, mcp, config)
    text, results = asyncio.run(orch.run_with_results("get item"))
    assert "Bad id." in text
    assert len(results) == 1
    assert results[0].ok is False
    assert mcp.calls == []


def test_missing_required_arguments_not_executed():
    config = OrchestratorConfig(max_steps=3)
    model = FakeModel([
        _assistant([("add_item", {"name": "w"})], content=""),
        {"role": "assistant", "content": "Missing fields."},
    ])
    mcp = FakeMcpClient(
        tools=[_tool_def("add_item", {
            "name": {"type": "string"},
            "price": {"type": "number"},
            "quantity": {"type": "integer"},
            "item_type": {"type": "string"},
        }, required=["name", "price", "quantity", "item_type"])],
    )
    orch = Orchestrator(model, mcp, config)
    text, results = asyncio.run(orch.run_with_results("add"))
    assert "Missing fields." in text
    assert len(results) == 1
    assert results[0].ok is False
    assert mcp.calls == []


def test_multiple_tool_calls_executed_in_one_round():
    config = OrchestratorConfig(max_steps=3)
    model = FakeModel([
        _assistant([
            ("get_all_items", {}),
            ("get_item_by_id", {"item_id": 7}),
        ], content=""),
        {"role": "assistant", "content": "All fetched."},
    ])
    mcp = FakeMcpClient(
        tools=[
            _tool_def("get_all_items"),
            _tool_def("get_item_by_id", {"item_id": {"type": "integer"}},
                      required=["item_id"]),
        ],
        call_results={
            "get_all_items": [{"id": 1, "name": "widget"}],
            "get_item_by_id": [{"id": 7, "name": "thing"}],
        },
    )
    orch = Orchestrator(model, mcp, config)
    text, results = asyncio.run(orch.run_with_results("get everything"))
    assert "All fetched." in text
    assert len(results) == 2
    assert all(r.ok is True for r in results)
    assert {r.tool_name for r in results} == {"get_all_items", "get_item_by_id"}
    assert mcp.calls == [
        ("get_all_items", {}),
        ("get_item_by_id", {"item_id": 7}),
    ]


def test_multiple_tool_rounds_before_final_answer():
    config = OrchestratorConfig(max_steps=5)
    model = FakeModel([
        _assistant([("get_all_items", {})], content=""),
        _assistant([("get_item_by_id", {"item_id": 3})], content=""),
        {"role": "assistant", "content": "Final answer here."},
    ])
    mcp = FakeMcpClient(
        tools=[
            _tool_def("get_all_items"),
            _tool_def("get_item_by_id", {"item_id": {"type": "integer"}},
                      required=["item_id"]),
        ],
        call_results={
            "get_all_items": [{"id": 1}, {"id": 3}],
            "get_item_by_id": [{"id": 3, "name": "thing"}],
        },
    )
    orch = Orchestrator(model, mcp, config)
    text, results = asyncio.run(orch.run_with_results("help"))
    assert "Final answer here." in text
    assert len(results) == 2
    assert [r.tool_name for r in results] == ["get_all_items", "get_item_by_id"]
    assert len(model.calls) == 3


def test_mutation_tool_blocked_by_read_only_default():
    config = OrchestratorConfig()
    assert config.allow_mutations is False
    assert config.is_allowed("add_item") is False
    assert config.is_allowed("get_all_items") is True

    model = FakeModel([
        _assistant([("add_item", {"name": "w", "price": 1, "quantity": 1,
                                   "item_type": "t"})], content=""),
        {"role": "assistant", "content": "Mutations disabled."},
    ])
    mcp = FakeMcpClient(tools=[])
    orch = Orchestrator(model, mcp, config)
    text, results = asyncio.run(orch.run_with_results("add item"))
    assert "Mutations disabled." in text
    assert len(results) == 1
    assert results[0].ok is False
    assert "not allowed" in results[0].error
    assert mcp.calls == []


def test_mutation_tool_executed_only_when_explicitly_authorized():
    config = OrchestratorConfig(allow_mutations=True, max_steps=3)
    assert config.is_allowed("add_item") is True
    assert config.is_mutation("add_item") is True

    model = FakeModel([
        _assistant([("add_item", {
            "name": "widget", "price": "9.99", "quantity": "5",
            "item_type": "gadget",
        })], content=""),
        {"role": "assistant", "content": "Added."},
    ])
    mcp = FakeMcpClient(
        tools=[_tool_def("add_item", {
            "name": {"type": "string"},
            "price": {"type": "number"},
            "quantity": {"type": "integer"},
            "item_type": {"type": "string"},
        }, required=["name", "price", "quantity", "item_type"])],
        call_results={"add_item": "Added Successfully"},
    )
    orch = Orchestrator(model, mcp, config)
    text, results = asyncio.run(orch.run_with_results("add item"))
    assert "Added." in text
    assert len(results) == 1
    assert results[0].ok is True
    assert results[0].result == "Added Successfully"
    assert mcp.calls == [("add_item", {
        "name": "widget", "price": 9.99, "quantity": 5, "item_type": "gadget",
    })]


def test_authorization_failure_is_reported_as_tool_result():
    config = OrchestratorConfig()
    model = FakeModel([
        _assistant([("delete_item", {"item_id": 1})], content=""),
        {"role": "assistant", "content": "Not authorized."},
    ])
    mcp = FakeMcpClient(tools=[])
    orch = Orchestrator(model, mcp, config)
    text, results = asyncio.run(orch.run_with_results("delete item"))
    assert "Not authorized." in text
    assert len(results) == 1
    assert results[0].tool_name == "delete_item"
    assert results[0].ok is False
    assert results[0].error is not None
    assert mcp.calls == []


def test_chat_assistant_delegates_to_orchestrator():
    model = FakeModel([
        _assistant([("get_all_items", {})], content=""),
        {"role": "assistant", "content": "Here are your items."},
    ])
    mcp = FakeMcpClient(
        tools=[_tool_def("get_all_items")],
        call_results={"get_all_items": [{"id": 1, "name": "widget"}]},
    )
    out = asyncio.run(chat_assistant(
        is_admin=False,
        data={"prompt": "list items"},
        model=model,
        mcp_client=mcp,
    ))
    assert "Here are your items." in out["text"]
    assert len(out["results"]) == 1
    assert out["results"][0].tool_name == "get_all_items"
    assert out["results"][0].ok is True
    assert mcp.calls == [("get_all_items", {})]


def test_chat_assistant_blocks_mutations_by_default():
    model = FakeModel([
        _assistant([("add_item", {"name": "w", "price": 1, "quantity": 1,
                                   "item_type": "t"})], content=""),
        {"role": "assistant", "content": "Mutations disabled."},
    ])
    mcp = FakeMcpClient(tools=[])
    out = asyncio.run(chat_assistant(
        is_admin=True,  # admin flag alone does not enable mutations
        data={"prompt": "add item"},
        model=model,
        mcp_client=mcp,
    ))
    assert "Mutations disabled." in out["text"]
    assert len(out["results"]) == 1
    assert out["results"][0].tool_name == "add_item"
    assert out["results"][0].ok is False
    assert mcp.calls == []


def test_chat_assistant_allows_mutation_when_explicitly_authorized():
    model = FakeModel([
        _assistant([("add_item", {
            "name": "widget", "price": "9.99", "quantity": "5",
            "item_type": "gadget",
        })], content=""),
        {"role": "assistant", "content": "Added."},
    ])
    mcp = FakeMcpClient(
        tools=[_tool_def("add_item", {
            "name": {"type": "string"},
            "price": {"type": "number"},
            "quantity": {"type": "integer"},
            "item_type": {"type": "string"},
        }, required=["name", "price", "quantity", "item_type"])],
        call_results={"add_item": "Added Successfully"},
    )
    out = asyncio.run(chat_assistant(
        is_admin=False,
        data={"prompt": "add item"},
        model=model,
        mcp_client=mcp,
        allow_mutations=True,
    ))
    assert "Added." in out["text"]
    assert len(out["results"]) == 1
    assert out["results"][0].tool_name == "add_item"
    assert out["results"][0].ok is True
    assert mcp.calls == [("add_item", {
        "name": "widget", "price": 9.99, "quantity": 5, "item_type": "gadget",
    })]
