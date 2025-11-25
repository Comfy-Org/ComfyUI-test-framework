"""ComfyUI client package."""

import os

from .comfy_client import ComfyClient
from .errors import (
    APIError,
    AuthenticationError,
    ComfyClientError,
    ConnectionError,
    ExecutionError,
    FirebaseAuthError,
    InvalidCredentialsError,
    NotConnectedError,
    NotFoundError,
    ServerUnreachableError,
    TimeoutError,
    TokenExpiredError,
    WorkflowError,
)
from .generated_models import NodeInfo, QueueInfo
from .models import TestExecution


def create_client(
    server_address: str | None = None,
    *,
    use_ssl: bool = False,
    email: str | None = None,
    password: str | None = None,
    api_key: str | None = None,
) -> ComfyClient:
    """
    Factory function to create a ComfyClient.

    Args:
        server_address: Server address (host:port). Defaults to localhost:8188
        use_ssl: Use HTTPS/WSS
        email: Firebase email (or from COMFY_CLOUD_EMAIL env var)
        password: Firebase password (or from COMFY_CLOUD_PASSWORD env var)
        api_key: Firebase API key (or from FIREBASE_API_KEY env var)

    Returns:
        Configured ComfyClient

    Examples:
        # Local server
        client = create_client("localhost:8188")

        # Cloud with env vars
        client = create_client("cloud.comfy.org", use_ssl=True)

        # Cloud with explicit credentials
        client = create_client(
            "cloud.comfy.org",
            use_ssl=True,
            email="user@example.com",
            password="secret",
            api_key="firebase-key"
        )
    """
    # Default server address
    if server_address is None:
        server_address = os.environ.get("COMFY_SERVER", "localhost:8188")

    # Load credentials from environment if not provided
    email = email or os.environ.get("COMFY_CLOUD_EMAIL")
    password = password or os.environ.get("COMFY_CLOUD_PASSWORD")
    api_key = api_key or os.environ.get("FIREBASE_API_KEY")

    return ComfyClient(
        server_address=server_address,
        use_ssl=use_ssl,
        email=email,
        password=password,
        api_key=api_key,
    )


__all__ = [
    # Client
    "ComfyClient",
    "create_client",
    # Models
    "TestExecution",
    "QueueInfo",
    "NodeInfo",
    # Errors
    "ComfyClientError",
    "ConnectionError",
    "ServerUnreachableError",
    "AuthenticationError",
    "FirebaseAuthError",
    "InvalidCredentialsError",
    "TokenExpiredError",
    "APIError",
    "NotFoundError",
    "ExecutionError",
    "WorkflowError",
    "TimeoutError",
    "NotConnectedError",
]
