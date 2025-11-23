"""ComfyUI client for executing workflows and tracking results."""

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional, Set
from urllib import request, parse

import websocket


@dataclass
class TestExecution:
    """Tracks the execution state of a single test workflow."""
    prompt_id: str
    runs: Set[str] = field(default_factory=set)
    cached: Set[str] = field(default_factory=set)
    outputs: Dict[str, Dict] = field(default_factory=dict)
    error: Optional[Dict] = None

    def did_run(self, node_id: str) -> bool:
        """Check if a node was executed."""
        return node_id in self.runs

    def was_cached(self, node_id: str) -> bool:
        """Check if a node was cached."""
        return node_id in self.cached

    def was_executed(self, node_id: str) -> bool:
        """Check if a node was either run or cached."""
        return self.did_run(node_id) or self.was_cached(node_id)

    @property
    def has_error(self) -> bool:
        """Check if execution encountered an error."""
        return self.error is not None


class ComfyClient:
    """Client for communicating with a ComfyUI server."""

    def __init__(self, server_address: str, client_id: Optional[str] = None):
        """
        Initialize ComfyUI client.

        Args:
            server_address: Server address in format "host:port" (e.g., "localhost:8188")
            client_id: Optional client ID (generated if not provided)
        """
        self.server_address = server_address
        self.client_id = client_id or str(uuid.uuid4())
        self.ws: Optional[websocket.WebSocket] = None
        self._connected = False

    def connect(self, retries: int = 5, retry_delay: float = 2.0) -> None:
        """
        Connect to ComfyUI server via WebSocket.

        Args:
            retries: Number of connection attempts
            retry_delay: Delay between retries in seconds

        Raises:
            ConnectionError: If connection fails after all retries
        """
        last_error = None

        for attempt in range(retries):
            try:
                ws = websocket.WebSocket()
                ws_url = f"ws://{self.server_address}/ws?clientId={self.client_id}"
                ws.connect(ws_url)
                self.ws = ws
                self._connected = True
                return
            except (ConnectionRefusedError, OSError) as e:
                last_error = e
                if attempt < retries - 1:
                    time.sleep(retry_delay)

        raise ConnectionError(
            f"Failed to connect to ComfyUI at {self.server_address} after {retries} attempts: {last_error}"
        )

    def close(self) -> None:
        """Close WebSocket connection."""
        if self.ws:
            self.ws.close()
            self._connected = False

    def queue_prompt(self, workflow: Dict) -> str:
        """
        Submit a workflow to the ComfyUI server.

        Args:
            workflow: Workflow dictionary (node_id -> node_data)

        Returns:
            Prompt ID for tracking execution

        Raises:
            RuntimeError: If request fails
        """
        payload = {
            "prompt": workflow,
            "client_id": self.client_id,
        }

        data = json.dumps(payload).encode('utf-8')
        url = f"http://{self.server_address}/prompt"

        try:
            req = request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with request.urlopen(req) as response:
                result = json.loads(response.read())
                return result['prompt_id']
        except Exception as e:
            raise RuntimeError(f"Failed to queue prompt: {e}")

    def get_history(self, prompt_id: str) -> Dict:
        """
        Fetch execution history for a prompt.

        Args:
            prompt_id: The prompt ID to fetch history for

        Returns:
            History dictionary with outputs and status

        Raises:
            RuntimeError: If request fails
        """
        url = f"http://{self.server_address}/history/{prompt_id}"

        try:
            with request.urlopen(url) as response:
                history = json.loads(response.read())
                return history.get(prompt_id, {})
        except Exception as e:
            raise RuntimeError(f"Failed to get history for {prompt_id}: {e}")

    def execute_workflow(
        self,
        workflow: Dict,
        timeout: int = 120,
        verbose: bool = False,
    ) -> TestExecution:
        """
        Execute a workflow and track its execution.

        Args:
            workflow: Workflow dictionary
            timeout: Timeout in seconds
            verbose: Print verbose execution info

        Returns:
            TestExecution object with execution results

        Raises:
            TimeoutError: If execution exceeds timeout
            RuntimeError: If execution fails
        """
        if not self._connected:
            raise RuntimeError("Client not connected. Call connect() first.")

        # Queue the prompt
        prompt_id = self.queue_prompt(workflow)
        execution = TestExecution(prompt_id=prompt_id)

        if verbose:
            print(f"  Queued prompt: {prompt_id}")

        # Track execution via WebSocket
        start_time = time.time()

        while True:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Execution timed out after {timeout} seconds")

            # Receive message with timeout
            self.ws.settimeout(min(5.0, timeout - elapsed))

            try:
                message_data = self.ws.recv()
            except websocket.WebSocketTimeoutException:
                # Check if we've exceeded overall timeout
                continue

            # Only process string messages (JSON)
            if not isinstance(message_data, str):
                continue

            message = json.loads(message_data)
            msg_type = message.get('type')

            # Filter messages for our prompt_id
            data = message.get('data', {})
            msg_prompt_id = data.get('prompt_id')

            if msg_type == 'executing':
                if msg_prompt_id == prompt_id:
                    node_id = data.get('node')

                    if node_id is None:
                        # Execution complete
                        if verbose:
                            print(f"  Execution complete")
                        break

                    execution.runs.add(node_id)
                    if verbose:
                        print(f"  Executing node: {node_id}")

            elif msg_type == 'execution_cached':
                if msg_prompt_id == prompt_id:
                    cached_nodes = data.get('nodes', [])
                    execution.cached.update(cached_nodes)
                    if verbose and cached_nodes:
                        print(f"  Cached nodes: {', '.join(cached_nodes)}")

            elif msg_type == 'execution_error':
                if msg_prompt_id == prompt_id:
                    execution.error = data
                    if verbose:
                        print(f"  Execution error: {data.get('exception_type', 'Unknown')}")
                    break

        # Fetch final outputs from history
        history = self.get_history(prompt_id)
        execution.outputs = history.get('outputs', {})

        return execution
