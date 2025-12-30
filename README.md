# ComfyUI Test Framework

A testing framework for ComfyUI workflows. Provides custom nodes for defining tests and assertions, plus a CLI tool for running tests in CI/CD environments.

## Features

- **Test Definition Nodes**: Define test metadata, GPU requirements, and timeouts
- **Assertion Nodes**: Validate values, images, tensors, and execution state
- **CLI Tool (`comfyci`)**: Run test workflows against a ComfyUI instance
- **Perceptual Hashing**: Compare images using difference hashing (dHash)
- **CI/CD Ready**: Designed for automated testing pipelines

## Installation

### As a Custom Node

Clone this repository into your ComfyUI custom nodes directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Comfy-Org/ComfyUI-test-framework.git
```

Restart ComfyUI to load the test nodes.

### CLI Tool

The CLI tool uses [uv](https://github.com/astral-sh/uv) for dependency management:

```bash
cd ComfyUI-test-framework/cli

# Run directly with uv
./comfyci.py --help

# Or install as a package
uv pip install -e .
comfyci --help
```

## Nodes

### Test Definition

| Node | Description |
|------|-------------|
| **Test Definition** | Define test metadata: name, description, GPU requirements, extra timeout |

### Assertions

| Node | Description |
|------|-------------|
| **Assert Executed** | Pass-through that verifies the upstream nodes executed (not cached) |
| **Assert Equal** | Check if two values are deeply equal |
| **Assert Not Equal** | Check if two values differ |
| **Assert Image Match** | Compare image against a perceptual hash with configurable threshold |
| **Assert Contains Color** | Verify image contains pixels of a specific color |
| **Assert Tensor Shape** | Validate tensor dimensions (batch, height, width, channels) |
| **Assert In Range** | Check if all tensor values are within min/max bounds |

### Utilities

| Node | Description |
|------|-------------|
| **Test Image Generator** | Generate test images: solid black, white, noise, or face pattern |

## Usage

### Creating a Test Workflow

1. Add a **Test Definition** node to your workflow
2. Set the test name, description, and whether it requires GPU
3. Connect your workflow outputs to assertion nodes
4. Save the workflow as JSON

Example workflow structure:
```
[Load Image] → [Some Processing] → [Assert Image Match]
                                  ↘ [Assert Tensor Shape]
[Test Definition]
```

### Running Tests

```bash
# Run all tests in a directory
comfyci ./tests/**/*.json --server localhost:8188

# Skip GPU-required tests
comfyci ./tests/**/*.json --cpu

# Verbose output
comfyci ./tests/**/*.json -v

# Stop on first failure
comfyci ./tests/**/*.json -x

# Run against Comfy Cloud
comfyci ./tests/**/*.json --cloud --op-entry Employee/TestCI
```

### Assert Image Match Workflow

The **Assert Image Match** node uses perceptual hashing to compare images:

1. Add the node without a hash - it will calculate and display the hash
2. Click "Accept Hash" to copy the calculated hash into the input field
3. Adjust the `delta` threshold (0.0 = exact match, 1.0 = any image)
4. Re-run to validate future executions against this baseline

## CLI Reference

```
Usage: comfyci [OPTIONS] PATTERNS...

  Run ComfyUI test workflows.

  PATTERNS: Glob patterns for test workflow JSON files (e.g., ./tests/**/*.json)

Options:
  --server TEXT      ComfyUI server address (host:port)  [default: localhost:8188]
  --ssl              Use HTTPS/WSS for server connection
  --cloud            Use Comfy Cloud API (e.g., /jobs instead of /history)
  --cpu              Skip tests that require GPU
  -v, --verbose      Enable verbose output
  -x, --failfast     Stop on first test failure
  --timeout INTEGER  Override default timeout in seconds (default: 120 + test extraTime)
  --no-color         Disable colored output
  --op-exe TEXT      1Password CLI executable (op or op.exe)  [default: op.exe]
  --op-entry TEXT    1Password entry path (e.g., Employee/TestCI). Fetches
                     email, password, FIREBASE_API_KEY.
  --help             Show this message and exit.
```

### Exit Codes

- `0`: All tests passed
- `1`: One or more tests failed
- `130`: Interrupted by user (Ctrl+C)

## License

MIT License - see [LICENSE](LICENSE) for details.
