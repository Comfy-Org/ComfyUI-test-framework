from comfy_api.v0_0_2 import io


class TestDefinition(io.ComfyNode):
    """Defines test metadata including name, GPU requirements, and timeout"""

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
