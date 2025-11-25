"""Unified ComfyUI client supporting any server with optional authentication."""

import json
import time
import uuid
from typing import Any
from urllib import error, request

import websocket

from .auth import FirebaseAuth
from .errors import (
    APIError,
    NotConnectedError,
    NotFoundError,
    ServerUnreachableError,
)
from .errors import (
    TimeoutError as ClientTimeoutError,
)
from .generated_models import NodeInfo, QueueInfo
from .models import TestExecution


class ComfyClient:
    """
    Unified client for any ComfyUI server.

    Works with:
    - Local servers: ComfyClient("localhost:8188")
    - Cloud servers: ComfyClient("cloud.comfy.org", email="...", password="...", api_key="...")
    - Custom servers: ComfyClient("myserver.com:443", use_ssl=True)

    Authentication is optional - only used when credentials are provided.
    """

    def __init__(
        self,
        server_address: str,
        *,
        use_ssl: bool = False,
        client_id: str | None = None,
        # Optional Firebase authentication
        email: str | None = None,
        password: str | None = None,
        api_key: str | None = None,
    ):
        """
        Initialize client.

        Args:
            server_address: Server address as "host:port" or "host" (default port 8188)
            use_ssl: Use HTTPS/WSS instead of HTTP/WS
            client_id: Custom client ID (auto-generated if not provided)
            email: Firebase email (for authenticated servers)
            password: Firebase password (for authenticated servers)
            api_key: Firebase API key (for authenticated servers)
        """
        # Parse server address
        if ":" in server_address:
            self._host, port_str = server_address.split(":", 1)
            self._port = int(port_str)
        else:
            self._host = server_address
            self._port = 443 if use_ssl else 8188

        self._use_ssl = use_ssl
        self._client_id = client_id or str(uuid.uuid4())

        # Setup optional authentication
        self._auth: FirebaseAuth | None = None
        if email and password and api_key:
            self._auth = FirebaseAuth(api_key=api_key, email=email, password=password)

        # Connection state
        self._ws: websocket.WebSocket | None = None
        self._connected = False

    @property
    def server_address(self) -> str:
        """Server address as host:port."""
        return f"{self._host}:{self._port}"

    @property
    def client_id(self) -> str:
        return self._client_id

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_authenticated(self) -> bool:
        """Whether client has valid authentication."""
        return self._auth is not None and self._auth.token is not None

    def _base_url(self) -> str:
        scheme = "https" if self._use_ssl else "http"
        return f"{scheme}://{self._host}:{self._port}"

    def _ws_url(self) -> str:
        scheme = "wss" if self._use_ssl else "ws"
        return f"{scheme}://{self._host}:{self._port}/ws?clientId={self._client_id}"

    def connect(self, retries: int = 5, retry_delay: float = 2.0) -> None:
        """
        Connect to server.

        If credentials were provided, authenticates first.
        Then establishes WebSocket connection.
        """
        last_error: Exception | None = None

        # Authenticate if credentials provided
        if self._auth:
            self._auth.authenticate()
            self._auth.create_session(self._base_url())

        # Connect WebSocket
        for attempt in range(retries):
            try:
                ws = websocket.WebSocket()
                ws_url = self._ws_url()

                # Add auth headers if authenticated
                headers: list[str] | None = None
                if self._auth:
                    headers = []
                    for key, value in self._auth.get_headers().items():
                        headers.append(f"{key}: {value}")
                    print(f"[DEBUG] WS connecting to {ws_url}")
                    print(f"[DEBUG] WS headers: {headers}")

                ws.connect(ws_url, header=headers)
                self._ws = ws
                self._connected = True
                return
            except (ConnectionRefusedError, OSError, websocket.WebSocketException) as e:
                last_error = e
                if attempt < retries - 1:
                    time.sleep(retry_delay)

        raise ServerUnreachableError(
            f"Failed to connect to {self.server_address} after {retries} attempts: {last_error}"
        )

    def close(self) -> None:
        """Close connection."""
        if self._ws:
            self._ws.close()
            self._ws = None
        self._connected = False

    def _get_headers(self, debug: bool = False) -> dict[str, str]:
        """Get request headers including auth if available."""
        headers = {"Content-Type": "application/json"}
        if self._auth:
            self._auth.ensure_valid(self._base_url())
            auth_headers = self._auth.get_headers()
            if debug:
                print(f"[DEBUG] Auth headers: {list(auth_headers.keys())}")
                if "Cookie" in auth_headers:
                    print(f"[DEBUG] Cookie: {auth_headers['Cookie'][:100]}...")
            headers.update(auth_headers)
        return headers

    def _http_request(
        self, method: str, endpoint: str, data: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Make HTTP request."""
        url = f"{self._base_url()}/api{endpoint}"
        headers = self._get_headers()

        try:
            if data is not None:
                body = json.dumps(data).encode()
                req = request.Request(url, data=body, headers=headers, method=method)
            else:
                req = request.Request(url, headers=headers, method=method)

            with request.urlopen(req) as resp:
                response_body = resp.read()
                if not response_body:
                    raise APIError(f"Empty response from {endpoint}", 0, endpoint)
                try:
                    return json.loads(response_body)
                except json.JSONDecodeError as je:
                    # Show first 200 chars of response for debugging
                    preview = response_body.decode()[:200]
                    raise APIError(
                        f"Invalid JSON from {endpoint}: {preview}...", 0, endpoint
                    ) from je
        except error.HTTPError as e:
            body = e.read().decode() if e.fp else ""
            if e.code == 404:
                raise NotFoundError(f"Not found: {endpoint}", 404, endpoint) from e
            raise APIError(f"HTTP {e.code}: {body}", e.code, endpoint) from e

    def queue_prompt(self, workflow: dict[str, Any]) -> str:
        """Submit workflow, return prompt_id."""
        payload = {"prompt": workflow, "client_id": self._client_id}
        result = self._http_request("POST", "/prompt", payload)
        return str(result["prompt_id"])

    def get_history(self, prompt_id: str) -> dict[str, Any]:
        """Get execution history for a prompt."""
        result = self._http_request("GET", f"/history/{prompt_id}")
        return result.get(prompt_id, {})

    def get_queue(self) -> QueueInfo:
        """Get queue status."""
        result = self._http_request("GET", "/queue")
        return QueueInfo.model_validate(result)

    def get_object_info(self, node_class: str | None = None) -> dict[str, NodeInfo]:
        """
        Get node information from /object_info.

        Args:
            node_class: Specific node class name, or None for all nodes

        Returns:
            Dictionary mapping node class names to NodeInfo objects
        """
        if node_class:
            result = self._http_request("GET", f"/object_info/{node_class}")
            return {node_class: NodeInfo.model_validate(result[node_class])}

        result = self._http_request("GET", "/object_info")
        return {k: NodeInfo.model_validate(v) for k, v in result.items()}

    def execute_workflow(
        self,
        workflow: dict[str, Any],
        timeout: int = 120,
        verbose: bool = False,
    ) -> TestExecution:
        """
        Execute workflow and track completion via WebSocket.

        Args:
            workflow: ComfyUI workflow dictionary
            timeout: Maximum seconds to wait for completion
            verbose: Print progress messages

        Returns:
            TestExecution with results
        """
        if not self._connected or not self._ws:
            raise NotConnectedError("Client not connected. Call connect() first.")

        prompt_id = self.queue_prompt(workflow)
        execution = TestExecution(prompt_id=prompt_id)

        if verbose:
            print(f"  Queued prompt: {prompt_id}")

        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise ClientTimeoutError(f"Execution timed out after {timeout} seconds")

            self._ws.settimeout(min(5.0, timeout - elapsed))

            try:
                message_data = self._ws.recv()
            except websocket.WebSocketTimeoutException:
                continue

            if not isinstance(message_data, str):
                if verbose:
                    print(f"  [WS] (binary data, {len(message_data)} bytes)")
                continue

            message = json.loads(message_data)
            msg_type = message.get("type")
            data = message.get("data", {})
            msg_prompt_id = data.get("prompt_id")

            if verbose:
                print(f"  [WS] {msg_type}: {message}")

            if msg_type == "executing" and msg_prompt_id == prompt_id:
                node_id = data.get("node")
                if node_id is None:
                    if verbose:
                        print("  Execution complete")
                    break
                execution.runs.add(node_id)
                if verbose:
                    print(f"  Executing node: {node_id}")

            elif msg_type == "execution_cached" and msg_prompt_id == prompt_id:
                cached_nodes = data.get("nodes", [])
                execution.cached.update(cached_nodes)
                if verbose and cached_nodes:
                    print(f"  Cached nodes: {', '.join(cached_nodes)}")

            elif msg_type == "execution_error" and msg_prompt_id == prompt_id:
                execution.error = data
                if verbose:
                    print(f"  Execution error: {data.get('exception_type', 'Unknown')}")
                break

            elif msg_type == "execution_success" and msg_prompt_id == prompt_id:
                if verbose:
                    print("  Execution complete (success)")
                break

        history = self.get_history(prompt_id)
        execution.outputs = history.get("outputs", {})

        return execution
