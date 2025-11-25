"""Error hierarchy for ComfyClient."""

from typing import Any


class ComfyClientError(Exception):
    """Base exception for all ComfyClient errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


# Connection errors
class ConnectionError(ComfyClientError):
    """Failed to connect to ComfyUI server."""


class ServerUnreachableError(ConnectionError):
    """Server is not reachable after retries."""


# Authentication errors
class AuthenticationError(ComfyClientError):
    """Authentication failure."""


class FirebaseAuthError(AuthenticationError):
    """Firebase-specific authentication error."""


class InvalidCredentialsError(FirebaseAuthError):
    """Invalid email or password."""


class TokenExpiredError(AuthenticationError):
    """Authentication token has expired."""


# API errors
class APIError(ComfyClientError):
    """API call failure."""

    def __init__(self, message: str, status_code: int, endpoint: str):
        super().__init__(message, {"status_code": status_code, "endpoint": endpoint})
        self.status_code = status_code
        self.endpoint = endpoint


class NotFoundError(APIError):
    """Resource not found (404)."""


# Execution errors
class ExecutionError(ComfyClientError):
    """Workflow execution failure."""


class WorkflowError(ExecutionError):
    """Error during workflow execution."""


class TimeoutError(ExecutionError):
    """Execution timed out."""


# State errors
class NotConnectedError(ComfyClientError):
    """Client is not connected."""
