from __future__ import annotations

from comfy_api.latest import ComfyExtension, io
from .nodes import TestImageGenerator, TestMustExecute, TestEqual, TestDefinition


class TestFrameworkExtension(ComfyExtension):
    """Extension for testing framework nodes"""

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        """Return all test framework nodes"""
        return [
            TestImageGenerator,
            TestMustExecute,
            TestEqual,
            TestDefinition,
        ]


async def comfy_entrypoint() -> ComfyExtension:
    """Entry point called by ComfyUI"""
    return TestFrameworkExtension()
