"""Exceptions for the LLM orchestration layer."""


class OrchestratorError(Exception):
    """Base exception for orchestration failures."""


class ToolNotAllowedError(OrchestratorError):
    """Raised when the model requests a tool not on the explicit allowlist."""


class ToolArgumentError(OrchestratorError):
    """Raised when tool arguments fail schema validation."""


class ToolExecutionError(OrchestratorError):
    """Raised when an approved tool raises during execution."""


class MaxStepsExceededError(OrchestratorError):
    """Raised when the bounded multi-step loop exhausts its step budget."""


class ToolTimeoutError(OrchestratorError):
    """Raised when a tool call exceeds its configured timeout."""


class CancellationError(OrchestratorError):
    """Raised when operation is cancelled by the caller."""
