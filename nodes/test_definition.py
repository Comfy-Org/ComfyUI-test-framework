from comfy_api.v0_0_2 import io


class TestDefinition(io.ComfyNode):
    """Defines test metadata including name, GPU requirements, and timeout"""

    SKILL_DOC = """
Every test workflow MUST contain exactly one `TestDefinition` node. It captures metadata
that the test runner (`comfyci`) uses to name, filter, and configure tests.

This node has no connections — all inputs are socketless (configured via the UI/JSON only).
It is an output node so it participates in execution.

### Inputs

- `name` — A short, descriptive test name (e.g. `"blur_preserves_dimensions"`). Used in
  test runner output and CI reports.
- `description` — Brief explanation of what the test verifies. Keep it to one sentence.
- `requiresGPU` — Set to `true` only if the test genuinely needs a GPU to run. Tests with
  `requiresGPU=false` run on every commit; GPU tests run less frequently due to cost.
  Default: `false`.
- `extraTime` — Additional timeout in seconds beyond the default 10s. Only increase if the
  operation legitimately takes longer (e.g. model inference). Range: 0–3600. Default: `0`.

### Usage Notes

- The `TestDefinition` node is standalone — it does not connect to any other node.
- It does not affect the workflow's execution graph; it only provides metadata.
- If `requiresGPU` is `false` but the test fails without a GPU, update it to `true`.
"""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TestDefinition",
            display_name="Test Definition",
            category="testing",
            description="Define test metadata: name, GPU requirements, and extra execution time",
            inputs=[
                io.String.Input(
                    "name",
                    default="",
                    socketless=True,
                ),
                io.String.Input(
                    "description",
                    default="",
                    multiline=True,
                    socketless=True,
                ),
                io.Boolean.Input(
                    "requiresGPU",
                    default=False,
                    socketless=True,
                ),
                io.Int.Input(
                    "extraTime",
                    default=0,
                    min=0,
                    max=3600,
                    display_mode=io.NumberDisplay.number,
                    socketless=True,
                ),
            ],
            outputs=[],
            is_output_node=True,
            is_dev_only=True,
        )

    @classmethod
    def execute(cls, name: str, description: str, requiresGPU: bool, extraTime: int) -> io.NodeOutput:
        """Store test definition metadata"""
        # This node just accepts the metadata and does nothing with it during execution
        # The actual test framework would read this metadata from the workflow
        return io.NodeOutput()
