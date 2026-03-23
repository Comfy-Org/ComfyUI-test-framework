#!/usr/bin/env python3
"""Generate manifest.json from node definitions.

Scans all node modules in nodes/ and extracts metadata from define_schema().
The manifest is consumed by external tools (e.g., workflow validators) that
need to know about test framework nodes without a running ComfyUI instance.

Usage:
    python scripts/generate_manifest.py
    python scripts/generate_manifest.py --check  # CI mode: fail if out of date
"""

import argparse
import importlib
import json
import os
import pkgutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NODES_DIR = REPO_ROOT / "nodes"
MANIFEST_PATH = REPO_ROOT / "manifest.json"


def collect_nodes() -> dict:
    """Import all node modules and extract schema metadata."""
    # Add repo root to path so 'nodes' package is importable
    sys.path.insert(0, str(REPO_ROOT))

    # Stub out ComfyUI dependencies that aren't available outside ComfyUI
    import types

    # Create minimal stubs for comfy_api.v0_0_2.io
    io_module = types.ModuleType("io")

    class _FakeType:
        """Stub for io type classes (io.Image, io.Mask, io.Int, etc.)"""
        def __init__(self, *a, **kw):
            pass

        @classmethod
        def Input(cls, *a, **kw):
            return {"args": a, "kwargs": kw}

        @classmethod
        def Output(cls, *a, **kw):
            return {"args": a, "kwargs": kw}

    class _NumberDisplay:
        number = "number"
        slider = "slider"

    class _Hidden:
        unique_id = "UNIQUE_ID"
        prompt = "PROMPT"
        extra_pnginfo = "EXTRA_PNGINFO"
        dynprompt = "DYNPROMPT"
        auth_token_comfy_org = "AUTH_TOKEN_COMFY_ORG"

    class Schema:
        def __init__(self, **kwargs):
            self._data = kwargs

    class ComfyNode:
        hidden = _Hidden()

        @classmethod
        def define_schema(cls):
            return None

    # Wire up the stubs
    for name in [
        "AnyType", "Image", "Mask", "Int", "Float", "String",
        "Boolean", "Combo", "Latent", "Model", "Clip", "Vae",
        "Conditioning",
    ]:
        setattr(io_module, name, type(name, (_FakeType,), {}))

    io_module.Schema = Schema
    io_module.ComfyNode = ComfyNode
    io_module.NumberDisplay = _NumberDisplay
    io_module.Hidden = _Hidden
    io_module.NodeOutput = lambda *a, **kw: None

    # Stub ui module
    ui_module = types.ModuleType("ui")

    class PreviewImage:
        def __init__(self, *a, **kw):
            pass

    ui_module.PreviewImage = PreviewImage

    # Stub ComfyAPISync
    class ComfyAPISync:
        @staticmethod
        def get_comfyui_version():
            return "0.0.0"

    comfy_api = types.ModuleType("comfy_api")
    v002 = types.ModuleType("comfy_api.v0_0_2")
    v002.io = io_module
    v002.ui = ui_module
    v002.ComfyAPISync = ComfyAPISync

    sys.modules["comfy_api"] = comfy_api
    sys.modules["comfy_api.v0_0_2"] = v002
    sys.modules["comfy_api.v0_0_2.io"] = io_module
    sys.modules["comfy_api.v0_0_2.ui"] = ui_module

    # Stub server module
    server_module = types.ModuleType("server")

    class FakePromptServer:
        class instance:
            @staticmethod
            def send_progress_text(*a, **kw):
                pass
    server_module.PromptServer = FakePromptServer
    sys.modules["server"] = server_module

    # Stub torch
    if "torch" not in sys.modules:
        torch_stub = types.ModuleType("torch")
        torch_stub.Tensor = type("Tensor", (), {})
        sys.modules["torch"] = torch_stub

    # Now import node modules
    manifest = {}
    for importer, modname, ispkg in pkgutil.iter_modules([str(NODES_DIR)]):
        if modname.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"nodes.{modname}")
        except Exception as e:
            print(f"Warning: Failed to import nodes.{modname}: {e}", file=sys.stderr)
            continue

        for attr_name in dir(mod):
            cls = getattr(mod, attr_name)
            if (
                isinstance(cls, type)
                and issubclass(cls, ComfyNode)
                and cls is not ComfyNode
                and hasattr(cls, "define_schema")
            ):
                try:
                    schema = cls.define_schema()
                    if schema and hasattr(schema, "_data"):
                        data = schema._data
                        node_id = data.get("node_id", attr_name)
                        manifest[node_id] = {
                            "display_name": data.get("display_name", node_id),
                            "category": data.get("category", ""),
                            "is_output_node": data.get("is_output_node", False),
                            "description": data.get("description", ""),
                            "module": modname,
                        }
                except Exception as e:
                    print(
                        f"Warning: Failed to get schema for {attr_name}: {e}",
                        file=sys.stderr,
                    )

    return dict(sorted(manifest.items()))


def main():
    parser = argparse.ArgumentParser(description="Generate test framework manifest")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check mode: fail if manifest.json is out of date",
    )
    args = parser.parse_args()

    manifest = collect_nodes()

    if not manifest:
        print("Error: No nodes found. Check that nodes/ directory exists.", file=sys.stderr)
        sys.exit(1)

    new_content = json.dumps(manifest, indent=2) + "\n"

    if args.check:
        if not MANIFEST_PATH.exists():
            print(f"FAIL: {MANIFEST_PATH} does not exist. Run: python scripts/generate_manifest.py")
            sys.exit(1)
        existing = MANIFEST_PATH.read_text()
        if existing != new_content:
            print(f"FAIL: {MANIFEST_PATH} is out of date. Run: python scripts/generate_manifest.py")
            sys.exit(1)
        print(f"OK: {MANIFEST_PATH} is up to date ({len(manifest)} nodes)")
        sys.exit(0)

    MANIFEST_PATH.write_text(new_content)
    print(f"Generated {MANIFEST_PATH} with {len(manifest)} nodes")

    # Summary
    output_nodes = [k for k, v in manifest.items() if v["is_output_node"]]
    regular_nodes = [k for k, v in manifest.items() if not v["is_output_node"]]
    print(f"  Output nodes ({len(output_nodes)}): {', '.join(output_nodes)}")
    print(f"  Regular nodes ({len(regular_nodes)}): {', '.join(regular_nodes)}")


if __name__ == "__main__":
    main()
