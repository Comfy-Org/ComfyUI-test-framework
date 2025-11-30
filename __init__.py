from __future__ import annotations

from comfy_api.latest import ComfyExtension, io
from .nodes import (
    TestImageGenerator,
    TestDefinition,
    AssertExecuted,
    AssertEqual,
    AssertNotEqual,
    AssertImageMatch,
    AssertContainsColor,
    AssertTensorShape,
    AssertInRange,
)


# Register web directory for frontend extensions
WEB_DIRECTORY = "web"


class TestFrameworkExtension(ComfyExtension):
    """Extension for testing framework nodes"""

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        """Return all test framework nodes"""
        return [
            # Utilities
            TestImageGenerator,
            TestDefinition,
            # Assertions
            AssertExecuted,
            AssertEqual,
            AssertNotEqual,
            AssertImageMatch,
            AssertContainsColor,
            AssertTensorShape,
            AssertInRange,
        ]


async def comfy_entrypoint() -> ComfyExtension:
    """Entry point called by ComfyUI"""
    return TestFrameworkExtension()
