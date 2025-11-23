"""Parse ComfyUI test workflow JSON files and extract metadata."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set


@dataclass
class TestMetadata:
    """Metadata extracted from TestDefinition node."""
    name: str
    description: str = ""
    requires_gpu: bool = False
    extra_time: int = 0


@dataclass
class WorkflowTest:
    """Represents a single test workflow with its metadata."""
    file_path: Path
    workflow: Dict
    metadata: Optional[TestMetadata]
    test_must_execute_nodes: Set[str]

    @property
    def name(self) -> str:
        """Get test name from metadata or file path."""
        if self.metadata:
            return self.metadata.name
        return self.file_path.stem

    @property
    def requires_gpu(self) -> bool:
        """Check if test requires GPU."""
        return self.metadata.requires_gpu if self.metadata else False

    @property
    def timeout(self) -> int:
        """Get timeout in seconds (base + extra time)."""
        base_timeout = 120  # 2 minutes default
        extra = self.metadata.extra_time if self.metadata else 0
        return base_timeout + extra


def parse_workflow_file(file_path: Path) -> WorkflowTest:
    """
    Parse a workflow JSON file and extract test metadata.

    Args:
        file_path: Path to the workflow JSON file

    Returns:
        WorkflowTest object with parsed data

    Raises:
        ValueError: If file cannot be parsed
    """
    try:
        with open(file_path, 'r') as f:
            workflow = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise ValueError(f"Failed to parse {file_path}: {e}")

    if not isinstance(workflow, dict):
        raise ValueError(f"Invalid workflow format in {file_path}: expected object")

    # Extract TestDefinition metadata
    metadata = None
    for node_id, node_data in workflow.items():
        if not isinstance(node_data, dict):
            continue

        class_type = node_data.get("class_type")
        if class_type == "TestDefinition":
            inputs = node_data.get("inputs", {})
            metadata = TestMetadata(
                name=inputs.get("name", file_path.stem),
                description=inputs.get("description", ""),
                requires_gpu=inputs.get("requiresGPU", False),
                extra_time=inputs.get("extraTime", 0),
            )
            break  # Only use first TestDefinition found

    # Find all TestMustExecute nodes
    test_must_execute_nodes = set()
    for node_id, node_data in workflow.items():
        if not isinstance(node_data, dict):
            continue

        if node_data.get("class_type") == "TestMustExecute":
            test_must_execute_nodes.add(node_id)

    return WorkflowTest(
        file_path=file_path,
        workflow=workflow,
        metadata=metadata,
        test_must_execute_nodes=test_must_execute_nodes,
    )


def discover_tests(patterns: List[str], base_path: Optional[Path] = None) -> List[WorkflowTest]:
    """
    Discover test workflow files matching the given patterns.

    Args:
        patterns: List of glob patterns (e.g., ["tests/**/*.json"])
        base_path: Base directory for relative patterns (default: current directory)

    Returns:
        List of WorkflowTest objects
    """
    if base_path is None:
        base_path = Path.cwd()

    test_files = set()
    for pattern in patterns:
        pattern_path = Path(pattern)

        # Handle absolute vs relative patterns
        if pattern_path.is_absolute():
            search_path = pattern_path.parent
            glob_pattern = pattern_path.name
        else:
            search_path = base_path
            glob_pattern = pattern

        # Use glob to find matching files
        if '**' in glob_pattern:
            matches = search_path.glob(glob_pattern)
        else:
            matches = search_path.glob(glob_pattern)

        for match in matches:
            if match.is_file() and match.suffix == '.json':
                test_files.add(match.resolve())

    # Parse all discovered test files
    tests = []
    for test_file in sorted(test_files):
        try:
            test = parse_workflow_file(test_file)
            tests.append(test)
        except ValueError as e:
            # Skip invalid files but log the error
            print(f"Warning: {e}")
            continue

    return tests
