# comfyci

A CLI tool for running ComfyUI test workflows in CI/CD environments.

## Overview

`comfyci` is a standalone command-line tool that executes ComfyUI workflow tests against a running ComfyUI instance. It validates test execution, verifies that required nodes run or are cached, and reports results in a standard testing format.

## Features

- **Glob pattern support**: Run multiple test files with patterns like `./tests/**/*.json`
- **GPU filtering**: Skip GPU-required tests with the `--cpu` flag
- **TestMustExecute validation**: Ensures all TestMustExecute nodes either ran or were cached
- **Standard output formats**: Simple text output (with future support for JUnit XML)
- **Timeout configuration**: Per-test timeouts with extraTime support
- **Fail-fast mode**: Stop on first failure for quick feedback
- **Verbose mode**: Detailed execution tracking and debugging
- **Architecture for parallelism**: Designed to support parallel execution in future versions

## Installation

This tool uses the `uv` package manager for fast, reliable dependency management.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Navigate to the cli directory
cd /path/to/ComfyUI/custom_nodes/ComfyUI-test-framework/cli

# Run directly with uv (no installation needed)
./comfyci.py --help
```

Alternatively, install as a package:

```bash
# Install in development mode
uv pip install -e .

# Now you can run from anywhere
comfyci --help
```

## Usage

### Basic Usage

```bash
# Run all tests in a directory
comfyci ./tests/**/*.json --target localhost:8188

# Run specific test files
comfyci test1.json test2.json --target localhost:8188

# Skip GPU-required tests
comfyci ./tests/**/*.json --cpu
```

### Command-Line Options

```
Usage: comfyci [OPTIONS] PATTERNS...

  Run ComfyUI test workflows.

  PATTERNS: Glob patterns for test workflow JSON files

Options:
  --target TEXT       ComfyUI server address (host:port) [default: localhost:8188]
  --cpu               Skip tests that require GPU
  -v, --verbose       Enable verbose output
  -x, --failfast      Stop on first test failure
  --timeout INTEGER   Override default timeout in seconds
  --no-color          Disable colored output
  --help              Show this message and exit
```

### Examples

**Run tests with verbose output:**
```bash
comfyci ./tests/**/*.json --verbose
```

**Run tests and stop on first failure:**
```bash
comfyci ./tests/**/*.json --failfast
```

**Run with custom timeout:**
```bash
comfyci ./tests/**/*.json --timeout 300
```

**Connect to remote ComfyUI instance:**
```bash
comfyci ./tests/**/*.json --target 192.168.1.100:8188
```

## Test Workflow Structure

Test workflows are standard ComfyUI workflow JSON files with special test nodes:

### TestDefinition Node

Defines test metadata:

```json
{
  "1": {
    "class_type": "TestDefinition",
    "inputs": {
      "name": "Image Generation Test",
      "description": "Tests basic image generation",
      "requiresGPU": false,
      "extraTime": 30
    }
  }
}
```

- `name`: Test name (shown in output)
- `description`: Test description
- `requiresGPU`: Set to `true` to skip with `--cpu` flag
- `extraTime`: Additional timeout in seconds (added to 120s default)

### TestMustExecute Node

Marks nodes that must execute or be cached for the test to pass:

```json
{
  "5": {
    "class_type": "TestMustExecute",
    "inputs": {
      "input": ["2", 0]
    }
  }
}
```

`comfyci` validates that all TestMustExecute nodes in the workflow either:
- Executed during the test run, or
- Were cached (already computed)

If any TestMustExecute node didn't run or cache, the test fails.

### TestEqual Node

Validates that two values are equal:

```json
{
  "6": {
    "class_type": "TestEqual",
    "inputs": {
      "input1": ["2", 0],
      "input2": ["3", 0]
    }
  }
}
```

The test fails if the inputs differ.

## Output Format

### Simple Format (Default)

Non-verbose mode shows quick feedback with dots:

```
Running 5 test(s)...
.....
======================================================================
PASSED
======================================================================
Tests run: 5
Passed: 5
```

- `.` = Passed
- `F` = Failed
- `s` = Skipped

### Verbose Format

Verbose mode shows detailed execution:

```
Connecting to ComfyUI at localhost:8188...
Connected successfully

[1/3] Image Generation Test
  Queued prompt: abc123
  Executing node: 2
  Executing node: 3
  Execution complete
  ✓ PASSED

[2/3] GPU Required Test
  - SKIPPED: GPU required (--cpu flag)

[3/3] Failing Test
  Queued prompt: def456
  Execution error: ValueError
  ✗ FAILED: ValueError in node 5: Input mismatch
```

## Exit Codes

- `0`: All tests passed
- `1`: One or more tests failed
- `130`: Interrupted by user (Ctrl+C)

## Architecture

The tool is designed with future parallel execution in mind:

### Components

- **workflow_parser.py**: Parses workflow JSON and extracts metadata
- **comfy_client.py**: WebSocket and HTTP client for ComfyUI communication
  - `ComfyClient`: Manages connection (reusable)
  - `TestExecution`: Tracks individual test execution (isolated state)
- **test_runner.py**: Execution strategies
  - `SequentialExecutor`: Current implementation
  - `ParallelExecutor`: Future support (pluggable design)
- **formatters/**: Output formatting
  - `OutputFormatter`: Abstract base class
  - `SimpleFormatter`: Current text-based implementation
  - Future: JUnit XML formatter

### Parallel Execution (Future)

The architecture supports adding parallel execution without refactoring:

1. **Isolated state**: Each `TestExecution` is independent
2. **Message routing**: WebSocket messages are routed by `prompt_id`
3. **Thread-safe formatters**: Designed for concurrent reporting
4. **Pluggable strategy**: Just implement `ParallelExecutor` class

Example future usage:
```bash
# This will work when ParallelExecutor is implemented
comfyci ./tests/**/*.json --parallel --workers 4
```

## Development

### Project Structure

```
cli/
├── comfyci.py              # Main CLI entry point
├── workflow_parser.py      # Workflow parsing and discovery
├── comfy_client.py         # ComfyUI client communication
├── test_runner.py          # Test execution strategies
├── formatters/
│   ├── __init__.py
│   ├── base.py            # Abstract formatter base
│   └── simple.py          # Simple text formatter
├── pyproject.toml         # Project configuration
└── README.md              # This file
```

### Adding New Output Formats

To add a new output format (e.g., JUnit XML):

1. Create `formatters/junit.py`
2. Inherit from `OutputFormatter`
3. Implement required methods:
   - `test_started()`
   - `test_passed()`
   - `test_failed()`
   - `test_skipped()`
   - `summary()`
4. Add CLI option to select format

### Running Tests

```bash
# Make sure ComfyUI is running
python -m comfy_ui.main

# In another terminal, run tests
./comfyci.py path/to/test/workflows/*.json --verbose
```

## Troubleshooting

### Connection refused

**Error**: `Failed to connect to ComfyUI at localhost:8188 after 5 attempts`

**Solution**: Make sure ComfyUI is running:
```bash
python main.py --listen 0.0.0.0 --port 8188
```

### No tests found

**Error**: `No test files found matching the given patterns`

**Solution**: Check your glob pattern and ensure JSON files exist:
```bash
# Use absolute paths if needed
comfyci /absolute/path/to/tests/**/*.json

# Or relative to current directory
cd /path/to/tests
comfyci ./**/*.json
```

### Tests timing out

**Error**: `Execution timed out after 120 seconds`

**Solutions**:
1. Increase timeout with `--timeout` flag
2. Add `extraTime` to TestDefinition in your workflow
3. Check if ComfyUI is actually processing the workflow

## License

This tool is part of the ComfyUI-test-framework project.
