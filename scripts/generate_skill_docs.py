#!/usr/bin/env python3
"""Generate SKILL.md node reference sections from SKILL_DOC attributes in node classes.

Parses node source files using AST (no ComfyUI imports needed) and extracts:
- Class names and display names from define_schema()
- SKILL_DOC class attributes with usage-oriented documentation

Usage:
    python generate_skill_docs.py [--out FILE]

Without --out, prints the generated markdown to stdout.
With --out, replaces the node reference sections in the specified SKILL.md file.

The script looks for markers in the target file:
    <!-- BEGIN GENERATED NODE DOCS -->
    <!-- END GENERATED NODE DOCS -->
and replaces everything between them with the generated content.
"""

import ast
import sys
import textwrap
from pathlib import Path


# Node categories for organizing output
CATEGORIES = [
    ("Test Definition", ["TestDefinition"]),
    ("General Assertions", ["AssertExecuted", "AssertEqual", "AssertNotEqual", "AssertTensorShape", "AssertInRange"]),
    ("Image Assertions", ["AssertImageMatch", "AssertContainsColor"]),
    ("Mask Assertions", ["AssertMaskCoverage", "AssertMaskBinary", "AssertMaskFuzzy"]),
    ("String Assertions", ["AssertStringContains", "AssertStringNotContains", "AssertStringLength", "AssertStringMatch"]),
    ("Generator / Input Nodes", ["TestImageGenerator", "TestMaskGenerator"]),
]


def extract_string_value(node: ast.AST) -> str | None:
    """Extract a string value from an AST node (handles str, JoinedStr, concatenation)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def extract_schema_info(class_node: ast.ClassDef) -> dict:
    """Extract display_name and node_id from define_schema() if present."""
    info = {"node_id": class_node.name, "display_name": class_node.name}

    for item in ast.walk(class_node):
        if isinstance(item, ast.Call):
            # Look for io.Schema(...) calls
            func = item.func
            is_schema = False
            if isinstance(func, ast.Attribute) and func.attr == "Schema":
                is_schema = True
            elif isinstance(func, ast.Name) and func.id == "Schema":
                is_schema = True

            if is_schema:
                for kw in item.keywords:
                    if kw.arg == "display_name":
                        val = extract_string_value(kw.value)
                        if val:
                            info["display_name"] = val
                    elif kw.arg == "node_id":
                        val = extract_string_value(kw.value)
                        if val:
                            info["node_id"] = val
    return info


def extract_nodes_from_file(filepath: Path) -> dict[str, dict]:
    """Parse a Python file and extract class name -> {display_name, skill_doc} for each ComfyNode."""
    source = filepath.read_text()
    tree = ast.parse(source, filename=str(filepath))

    nodes = {}
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        skill_doc = None
        for item in node.body:
            # Look for SKILL_DOC = "..." assignments
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name) and target.id == "SKILL_DOC":
                        skill_doc = extract_string_value(item.value)

        if skill_doc is None:
            continue

        schema_info = extract_schema_info(node)
        nodes[node.name] = {
            "class_name": node.name,
            "display_name": schema_info["display_name"],
            "node_id": schema_info["node_id"],
            "skill_doc": textwrap.dedent(skill_doc).strip(),
        }

    return nodes


def generate_markdown(all_nodes: dict[str, dict]) -> str:
    """Generate the markdown content for all nodes, organized by category."""
    from datetime import datetime, timezone, timedelta

    pst = timezone(timedelta(hours=-8))
    now = datetime.now(pst)
    timestamp = now.strftime("%-m/%-d/%Y at %-I:%M %p PST")

    lines = [
        f"> Auto-generated from `ComfyUI-test-framework/scripts/generate_skill_docs.py`",
        f"> on {timestamp} ({len(all_nodes)} nodes)",
        "",
    ]

    for category_name, class_names in CATEGORIES:
        lines.append(f"## {category_name}")
        lines.append("")

        for class_name in class_names:
            if class_name not in all_nodes:
                lines.append(f"<!-- WARNING: {class_name} not found in source -->")
                lines.append("")
                continue

            node = all_nodes[class_name]
            lines.append(f"### {node['display_name']}")
            lines.append("")
            lines.append(node["skill_doc"])
            lines.append("")

    return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate SKILL.md node reference from SKILL_DOC attributes")
    parser.add_argument("--out", type=str, help="Target SKILL.md file to update (replaces between markers)")
    parser.add_argument("--nodes-dir", type=str, default=None, help="Path to nodes/ directory (default: auto-detect)")
    args = parser.parse_args()

    # Find the nodes directory
    if args.nodes_dir:
        nodes_dir = Path(args.nodes_dir)
    else:
        # Auto-detect: script is in scripts/, nodes/ is a sibling
        script_dir = Path(__file__).parent
        nodes_dir = script_dir.parent / "nodes"

    if not nodes_dir.is_dir():
        print(f"Error: nodes directory not found at {nodes_dir}", file=sys.stderr)
        sys.exit(1)

    # Collect all nodes from all .py files
    all_nodes = {}
    for py_file in sorted(nodes_dir.glob("*.py")):
        if py_file.name == "__init__.py":
            continue
        extracted = extract_nodes_from_file(py_file)
        all_nodes.update(extracted)

    # Check for nodes in CATEGORIES that weren't found
    all_expected = set()
    for _, class_names in CATEGORIES:
        all_expected.update(class_names)

    found = set(all_nodes.keys())
    missing = all_expected - found
    extra = found - all_expected
    if missing:
        print(f"Warning: nodes in CATEGORIES but not found in source: {missing}", file=sys.stderr)
    if extra:
        print(f"Warning: nodes with SKILL_DOC but not in CATEGORIES: {extra}", file=sys.stderr)

    # Generate markdown
    markdown = generate_markdown(all_nodes)

    if args.out:
        # Update the target file between markers
        out_path = Path(args.out)
        if not out_path.exists():
            print(f"Error: output file not found: {out_path}", file=sys.stderr)
            sys.exit(1)

        content = out_path.read_text()
        begin_marker = "<!-- BEGIN GENERATED NODE DOCS -->"
        end_marker = "<!-- END GENERATED NODE DOCS -->"

        if begin_marker not in content or end_marker not in content:
            print(f"Error: markers not found in {out_path}. Add these lines:", file=sys.stderr)
            print(f"  {begin_marker}", file=sys.stderr)
            print(f"  {end_marker}", file=sys.stderr)
            sys.exit(1)

        begin_idx = content.index(begin_marker) + len(begin_marker)
        end_idx = content.index(end_marker)

        new_content = content[:begin_idx] + "\n\n" + markdown + "\n" + content[end_idx:]
        out_path.write_text(new_content)
        print(f"Updated {out_path} ({len(all_nodes)} nodes)", file=sys.stderr)
    else:
        # Print to stdout
        print(markdown)


if __name__ == "__main__":
    main()
