from typing import Any

from comfy_api.v0_0_2 import io
from server import PromptServer
import torch


class AssertExecuted(io.ComfyNode):
    """Pass-through node that marks a value as requiring execution (not cached)"""

    SKILL_DOC = """
Takes in any value and returns it unchanged, while asserting that the node actually
executed (was not served from cache). Use as a pass-through in the middle of a chain
when you need to verify that upstream nodes ran.

### Inputs

- `input` (ANY) — The value to pass through. Accepts any type (IMAGE, MASK, STRING, etc.).

### Outputs

- Returns the input value unchanged. Output type matches input type.

### When to Use

- When a node has no testable output but you need to confirm it ran.
- Insert between two nodes to assert the chain executed end-to-end.
- Combine with other assertions: route through `AssertExecuted` first, then to a
  type-specific assertion.

### Example

```
[NodeUnderTest] → [AssertExecuted] → [PreviewImage]
```

The `AssertExecuted` confirms the node ran; `PreviewImage` provides the output node.
"""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AssertExecuted",
            display_name="Assert Executed",
            category="testing",
            description="Pass-through node that verifies execution occurred (not cached)",
            inputs=[
                io.AnyType.Input("input"),
            ],
            outputs=[
                io.AnyType.Output("output"),
            ],
            is_output_node=False,
            is_dev_only=True,
        )

    @classmethod
    def execute(cls, input: Any) -> io.NodeOutput:
        """Accept any input and return it unchanged"""
        return io.NodeOutput(input)


class AssertEqual(io.ComfyNode):
    """Output node that checks if two inputs are equal"""

    SKILL_DOC = """
Compares two values for deep equality. Fails with a detailed error if they differ.
This is an output node (no outputs — terminates the chain).

### Inputs

- `input1` (ANY) — First value to compare.
- `input2` (ANY) — Second value to compare.

### Comparison Behavior

- **Tensors**: Checks shape equality, then uses `torch.allclose(rtol=1e-5, atol=1e-8)`
  for floating-point tolerance.
- **Dicts**: Checks key sets match, then recursively compares values.
- **Lists/Tuples**: Checks length, then element-by-element comparison.
- **Other types**: Uses `==` operator.
- Type mismatch between `input1` and `input2` always fails.

### When to Use

- Verify a node produces an exact expected output (e.g. string concatenation,
  mathematical operation, deterministic transform).
- Compare the output of two different nodes that should produce identical results.
- Validate that a node returns its input unchanged (identity test).

### Example

Test that a color inversion applied twice returns the original image:
```
[TestImageGenerator] → [InvertNode] → [InvertNode] → [AssertEqual.input1]
[TestImageGenerator] ─────────────────────────────→ [AssertEqual.input2]
```
"""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AssertEqual",
            display_name="Assert Equal",
            category="testing",
            description="Checks if two inputs are equal, throws error if not",
            inputs=[
                io.AnyType.Input("input1"),
                io.AnyType.Input("input2"),
            ],
            outputs=[],
            hidden=[io.Hidden.unique_id],
            is_output_node=True,
            is_dev_only=True,
        )

    @classmethod
    def _deep_compare(cls, obj1: Any, obj2: Any) -> tuple[bool, str]:
        """
        Deep comparison of two objects
        Returns (is_equal, error_message)
        """
        # Check if types are different
        if type(obj1) != type(obj2):
            return False, f"Type mismatch: {type(obj1).__name__} != {type(obj2).__name__}"

        # Handle dictionaries
        if isinstance(obj1, dict):
            if set(obj1.keys()) != set(obj2.keys()):
                missing_in_2 = set(obj1.keys()) - set(obj2.keys())
                missing_in_1 = set(obj2.keys()) - set(obj1.keys())
                msg = "Dictionary keys differ."
                if missing_in_2:
                    msg += f" Keys in input1 but not input2: {missing_in_2}."
                if missing_in_1:
                    msg += f" Keys in input2 but not input1: {missing_in_1}."
                return False, msg

            for key in obj1.keys():
                is_equal, msg = cls._deep_compare(obj1[key], obj2[key])
                if not is_equal:
                    return False, f"Dictionary key '{key}': {msg}"

            return True, ""

        # Handle lists and tuples
        elif isinstance(obj1, (list, tuple)):
            if len(obj1) != len(obj2):
                return False, f"Sequence length mismatch: {len(obj1)} != {len(obj2)}"

            for i, (item1, item2) in enumerate(zip(obj1, obj2)):
                is_equal, msg = cls._deep_compare(item1, item2)
                if not is_equal:
                    return False, f"Index {i}: {msg}"

            return True, ""

        # Handle tensors
        elif isinstance(obj1, torch.Tensor):
            if obj1.shape != obj2.shape:
                return False, f"Tensor shape mismatch: {obj1.shape} != {obj2.shape}"

            if not torch.allclose(obj1, obj2, rtol=1e-5, atol=1e-8):
                return False, "Tensor values differ"

            return True, ""

        # Handle other types with equality operator
        else:
            if obj1 != obj2:
                return False, f"Values differ: {obj1} != {obj2}"
            return True, ""

    @classmethod
    def execute(cls, input1: Any, input2: Any) -> io.NodeOutput:
        """Check if inputs are equal, raise error if not"""

        is_equal, error_msg = cls._deep_compare(input1, input2)

        if not is_equal:
            raise ValueError(f"Inputs are not equal: {error_msg}")

        # If equal, return successfully with checkmark
        PromptServer.instance.send_progress_text("✅ Test Passed", cls.hidden.unique_id)
        return io.NodeOutput()


class AssertNotEqual(io.ComfyNode):
    """Output node that checks if two inputs are NOT equal"""

    SKILL_DOC = """
Checks that two values are NOT equal. Fails if they are identical.
This is an output node (no outputs — terminates the chain).

### Inputs

- `input1` (ANY) — First value.
- `input2` (ANY) — Second value.

### Comparison Behavior

Same deep comparison logic as `AssertEqual` (tensor allclose, recursive dict/list,
`==` for others) — but the assertion is inverted.

### When to Use

- Verify that a processing node actually changed its input (e.g. a blur, noise
  addition, or color correction produced a different output than the original).
- Confirm that two different parameter settings produce different outputs.

### Example

Test that a blur node actually modifies the image:
```
[TestImageGenerator] → [BlurNode] → [AssertNotEqual.input1]
[TestImageGenerator] ────────────→ [AssertNotEqual.input2]
```
"""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AssertNotEqual",
            display_name="Assert Not Equal",
            category="testing",
            description="Checks if two inputs are not equal, throws error if they are equal",
            inputs=[
                io.AnyType.Input("input1"),
                io.AnyType.Input("input2"),
            ],
            outputs=[],
            hidden=[io.Hidden.unique_id],
            is_output_node=True,
            is_dev_only=True,
        )

    @classmethod
    def _are_equal(cls, obj1: Any, obj2: Any) -> bool:
        """Check if two objects are equal (reuses logic from AssertEqual)"""
        # Check if types are different
        if type(obj1) != type(obj2):
            return False

        # Handle dictionaries
        if isinstance(obj1, dict):
            if set(obj1.keys()) != set(obj2.keys()):
                return False
            for key in obj1.keys():
                if not cls._are_equal(obj1[key], obj2[key]):
                    return False
            return True

        # Handle lists and tuples
        elif isinstance(obj1, (list, tuple)):
            if len(obj1) != len(obj2):
                return False
            for item1, item2 in zip(obj1, obj2):
                if not cls._are_equal(item1, item2):
                    return False
            return True

        # Handle tensors
        elif isinstance(obj1, torch.Tensor):
            if obj1.shape != obj2.shape:
                return False
            return torch.allclose(obj1, obj2, rtol=1e-5, atol=1e-8)

        # Handle other types with equality operator
        else:
            return obj1 == obj2

    @classmethod
    def execute(cls, input1: Any, input2: Any) -> io.NodeOutput:
        """Check if inputs are NOT equal, raise error if they are equal"""

        if cls._are_equal(input1, input2):
            raise ValueError("Inputs are equal when they should be different")

        PromptServer.instance.send_progress_text("✅ Test Passed (values differ)", cls.hidden.unique_id)
        return io.NodeOutput()


class AssertTensorShape(io.ComfyNode):
    """Output node that checks tensor dimensions"""

    SKILL_DOC = """
Validates that a tensor has the expected dimensions. Supports both 3D `[B,H,W]` masks
and 4D `[B,H,W,C]` images. Use `-1` for any dimension you want to accept any value for.
This is an output node.

### Inputs

- `tensor` (ANY) — The tensor to validate. Must be a `torch.Tensor`.
- `batch` (INT) — Expected batch size. Default: `-1` (any).
- `height` (INT) — Expected height. Default: `-1` (any).
- `width` (INT) — Expected width. Default: `-1` (any).
- `channels` (INT) — Expected channel count. Default: `-1` (any). Set to `-1` for
  3D mask tensors (which have no channel dimension).

### When to Use

- Verify a node preserves or transforms image dimensions correctly.
- Check that a batch-processing node outputs the expected number of frames.
- Validate that a mask node outputs 3D `[B,H,W]` and not 4D `[B,H,W,C]`.
- Use after any node where dimensions might change (resize, crop, pad, tile).

### Tips

- For images: expect 4D `[B,H,W,3]` — set `channels=3`.
- For masks: expect 3D `[B,H,W]` — leave `channels=-1`.
- Set only the dimensions you care about; use `-1` for the rest.
- All mismatches are reported at once (not just the first one).

### Example

Verify an upscale node doubles the resolution:
```
[TestImageGenerator(width=256, height=256)] → [Upscale2x] → [AssertTensorShape(batch=1, height=512, width=512, channels=3)]
```
"""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AssertTensorShape",
            display_name="Assert Tensor Shape",
            category="testing",
            description="Checks if tensor has expected dimensions. Supports 3D [B,H,W] masks and 4D [B,H,W,C] images. Use -1 to accept any value, or set channels=-1 for 3D tensors.",
            inputs=[
                io.AnyType.Input("tensor"),
                io.Int.Input(
                    "batch",
                    default=-1,
                    min=-1,
                    max=1024,
                    display_mode=io.NumberDisplay.number,
                ),
                io.Int.Input(
                    "height",
                    default=-1,
                    min=-1,
                    max=16384,
                    display_mode=io.NumberDisplay.number,
                ),
                io.Int.Input(
                    "width",
                    default=-1,
                    min=-1,
                    max=16384,
                    display_mode=io.NumberDisplay.number,
                ),
                io.Int.Input(
                    "channels",
                    default=-1,
                    min=-1,
                    max=1024,
                    display_mode=io.NumberDisplay.number,
                ),
            ],
            outputs=[],
            hidden=[io.Hidden.unique_id],
            is_output_node=True,
            is_dev_only=True,
        )

    @classmethod
    def execute(
        cls,
        tensor: Any,
        batch: int,
        height: int,
        width: int,
        channels: int,
    ) -> io.NodeOutput:
        """Check if tensor has expected shape"""

        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Input is not a tensor, got {type(tensor).__name__}")

        shape = tensor.shape
        errors = []

        # Handle both 3D tensors [B,H,W] (masks) and 4D tensors [B,H,W,C] (images)
        if len(shape) == 3:
            actual_batch, actual_height, actual_width = shape
            actual_channels = None  # 3D tensors have no channel dimension

            if batch != -1 and actual_batch != batch:
                errors.append(f"batch: expected {batch}, got {actual_batch}")
            if height != -1 and actual_height != height:
                errors.append(f"height: expected {height}, got {actual_height}")
            if width != -1 and actual_width != width:
                errors.append(f"width: expected {width}, got {actual_width}")
            if channels != -1:
                errors.append(f"channels: expected {channels}, but tensor is 3D (no channel dimension)")

        elif len(shape) == 4:
            actual_batch, actual_height, actual_width, actual_channels = shape

            if batch != -1 and actual_batch != batch:
                errors.append(f"batch: expected {batch}, got {actual_batch}")
            if height != -1 and actual_height != height:
                errors.append(f"height: expected {height}, got {actual_height}")
            if width != -1 and actual_width != width:
                errors.append(f"width: expected {width}, got {actual_width}")
            if channels != -1 and actual_channels != channels:
                errors.append(f"channels: expected {channels}, got {actual_channels}")

        else:
            raise ValueError(f"Expected 3D [B,H,W] or 4D [B,H,W,C] tensor, got {len(shape)}D with shape {list(shape)}")

        if errors:
            raise ValueError(f"Tensor shape mismatch:\n" + "\n".join(errors) + f"\nActual shape: {list(shape)}")

        if actual_channels is None:
            shape_str = f"[{actual_batch}, {actual_height}, {actual_width}]"
        else:
            shape_str = f"[{actual_batch}, {actual_height}, {actual_width}, {actual_channels}]"

        PromptServer.instance.send_progress_text(
            f"✅ Test Passed\nShape: {shape_str}",
            cls.hidden.unique_id
        )
        return io.NodeOutput()


class AssertInRange(io.ComfyNode):
    """Output node that checks if all tensor values are within bounds"""

    SKILL_DOC = """
Checks that ALL values in a tensor fall within `[min_value, max_value]`. Useful for
catching NaN, inf, or out-of-range pixel values after processing. This is an output node.

### Inputs

- `tensor` (ANY) — The tensor to validate. Must be a `torch.Tensor`.
- `min_value` (FLOAT) — Lower bound (inclusive). Default: `0.0`.
- `max_value` (FLOAT) — Upper bound (inclusive). Default: `1.0`.

### When to Use

- After any image/mask processing node to verify outputs stay in valid `[0.0, 1.0]` range.
- After mathematical operations that might produce negative or >1 values.
- To detect NaN or inf values (they will be outside any finite range).
- For nodes that should produce values in a specific range (e.g. depth maps in `[0, 1]`).

### Tips

- The default range `[0.0, 1.0]` is correct for most ComfyUI image and mask tensors.
- The error message reports the actual min and max found in the tensor.

### Example

Verify a color correction node keeps values in valid range:
```
[TestImageGenerator(image_type="noise")] → [ColorCorrection] → [AssertInRange(min_value=0.0, max_value=1.0)]
```
"""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AssertInRange",
            display_name="Assert In Range",
            category="testing",
            description="Checks if all tensor values are within min/max bounds",
            inputs=[
                io.AnyType.Input("tensor"),
                io.Float.Input(
                    "min_value",
                    default=0.0,
                    min=-1e10,
                    max=1e10,
                    step=0.01,
                    display_mode=io.NumberDisplay.number,
                ),
                io.Float.Input(
                    "max_value",
                    default=1.0,
                    min=-1e10,
                    max=1e10,
                    step=0.01,
                    display_mode=io.NumberDisplay.number,
                ),
            ],
            outputs=[],
            hidden=[io.Hidden.unique_id],
            is_output_node=True,
            is_dev_only=True,
        )

    @classmethod
    def execute(
        cls,
        tensor: Any,
        min_value: float,
        max_value: float,
    ) -> io.NodeOutput:
        """Check if all tensor values are within bounds"""

        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Input is not a tensor, got {type(tensor).__name__}")

        actual_min = tensor.min().item()
        actual_max = tensor.max().item()

        errors = []
        if actual_min < min_value:
            errors.append(f"Minimum value {actual_min:.6f} is below threshold {min_value}")
        if actual_max > max_value:
            errors.append(f"Maximum value {actual_max:.6f} exceeds threshold {max_value}")

        if errors:
            raise ValueError(
                "Tensor values out of range:\n" + "\n".join(errors) +
                f"\nActual range: [{actual_min:.6f}, {actual_max:.6f}]"
            )

        PromptServer.instance.send_progress_text(
            f"✅ Test Passed\nRange: [{actual_min:.4f}, {actual_max:.4f}]",
            cls.hidden.unique_id
        )
        return io.NodeOutput()
