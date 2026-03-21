from __future__ import annotations

import importlib
import inspect
import pkgutil
import traceback

from comfy_api.latest import ComfyExtension, io

PREFIX = "[ComfyUI Test Framework]"

# Register web directory for frontend extensions
WEB_DIRECTORY = "web"


def collect_nodes(module) -> list[type[io.ComfyNode]]:
    """Find all io.ComfyNode subclasses defined in a module."""
    return [
        obj for _, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, io.ComfyNode)
        and obj is not io.ComfyNode
        and obj.__module__ == module.__name__
    ]


def load_nodes() -> list[type[io.ComfyNode]]:
    """Discover and load all io.ComfyNode subclasses from the nodes/ package."""
    node_classes: list[type[io.ComfyNode]] = []
    failed: list[tuple[str, Exception]] = []

    try:
        nodes_pkg = importlib.import_module(".nodes", package=__name__)
    except Exception as e:
        print(f"{PREFIX} Failed to import nodes package: {e}")
        traceback.print_exc()
        return node_classes

    for _, name, _ in pkgutil.walk_packages(nodes_pkg.__path__, prefix=nodes_pkg.__name__ + "."):
        try:
            mod = importlib.import_module(name)
            node_classes.extend(collect_nodes(mod))
        except Exception as e:
            failed.append((name, e))
            traceback.print_exc()

    node_names = ", ".join(cls.__name__ for cls in node_classes)
    print(f"{PREFIX} Loaded {len(node_classes)} nodes: {node_names}")

    if failed:
        for name, err in failed:
            print(f"{PREFIX} FAILED to load {name}: {type(err).__name__}: {err}")

    return node_classes


_discovered_nodes = load_nodes()


class TestFrameworkExtension(ComfyExtension):
    """Extension for testing framework nodes"""

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        """Return all discovered test framework nodes"""
        return _discovered_nodes


async def comfy_entrypoint() -> ComfyExtension:
    """Entry point called by ComfyUI"""
    return TestFrameworkExtension()
