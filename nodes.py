from typing import Any

from comfy_api.v0_0_2 import io, ui, ComfyAPISync
from server import PromptServer
import torch
from pathlib import Path


class TestImageGenerator(io.ComfyNode):
    """Generates test images based on selected type"""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TestImageGenerator",
            display_name="Test Image Generator",
            category="testing",
            description="Generate test images: solid black, solid white, noise, or face",
            inputs=[
                io.Combo.Input(
                    "image_type",
                    options=["black", "white", "noise", "face"],
                    default="black"
                ),
                io.Int.Input(
                    "width",
                    default=512,
                    min=64,
                    max=4096,
                    step=8,
                    display_mode=io.NumberDisplay.number
                ),
                io.Int.Input(
                    "height",
                    default=512,
                    min=64,
                    max=4096,
                    step=8,
                    display_mode=io.NumberDisplay.number
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFF,
                    display_mode=io.NumberDisplay.number
                ),
            ],
            outputs=[
                io.Image.Output(display_name="Image"),
            ]
        )

    @classmethod
    def execute(
        cls,
        image_type: str,
        width: int,
        height: int,
        seed: int,
    ) -> io.NodeOutput:
        """Generate test image based on type"""

        # Create image tensor with shape [B,H,W,C] where B=1, C=3 (RGB)
        batch = 1
        channels = 3

        if image_type == "black":
            # Solid black image
            image = torch.zeros(batch, height, width, channels)

        elif image_type == "white":
            # Solid white image
            image = torch.ones(batch, height, width, channels)

        elif image_type == "noise":
            # Random noise image with seed
            generator = torch.Generator()
            generator.manual_seed(seed)
            image = torch.rand(batch, height, width, channels, generator=generator)

        elif image_type == "face":
            # Load face image from file
            # Get the path to the face.png file in the same directory as this module
            module_dir = Path(__file__).parent
            face_path = module_dir / "face.png"

            if not face_path.exists():
                # If face.png doesn't exist, create a placeholder pattern
                # Create a simple pattern that vaguely looks like a face
                image = torch.ones(batch, height, width, channels) * 0.9
                # Add some simple shapes for eyes and mouth
                center_y, center_x = height // 2, width // 2
                # Left eye
                y1, y2 = center_y - height // 6, center_y - height // 8
                x1, x2 = center_x - width // 6, center_x - width // 10
                image[:, y1:y2, x1:x2, :] = 0.2
                # Right eye
                x1, x2 = center_x + width // 10, center_x + width // 6
                image[:, y1:y2, x1:x2, :] = 0.2
                # Mouth
                y1, y2 = center_y + height // 8, center_y + height // 6
                x1, x2 = center_x - width // 8, center_x + width // 8
                image[:, y1:y2, x1:x2, :] = 0.2
            else:
                # Load the actual face image
                from PIL import Image
                import numpy as np

                pil_image = Image.open(face_path).convert('RGB')
                # Resize to requested dimensions
                pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
                # Convert to tensor [H,W,C]
                image_np = np.array(pil_image).astype(np.float32) / 255.0
                # Add batch dimension [B,H,W,C]
                image = torch.from_numpy(image_np).unsqueeze(0)

        else:
            # Default to black if unknown type
            image = torch.zeros(batch, height, width, channels)

        # Ensure values are in valid range [0, 1]
        image = torch.clamp(image, 0.0, 1.0)

        return io.NodeOutput(image)


class AssertExecuted(io.ComfyNode):
    """Pass-through node that marks a value as requiring execution (not cached)"""

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
        )

    @classmethod
    def execute(cls, input: Any) -> io.NodeOutput:
        """Accept any input and return it unchanged"""
        return io.NodeOutput(input)


class AssertEqual(io.ComfyNode):
    """Output node that checks if two inputs are equal"""

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
        )

    @classmethod
    def execute(cls, name: str, description: str, requiresGPU: bool, extraTime: int) -> io.NodeOutput:
        """Store test definition metadata"""
        # This node just accepts the metadata and does nothing with it during execution
        # The actual test framework would read this metadata from the workflow
        return io.NodeOutput()


class AssertImageMatch(io.ComfyNode):
    """Output node that validates image against perceptual hash"""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AssertImageMatch",
            display_name="Assert Image Match",
            category="testing",
            description="Compare image against perceptual hash and fail if difference exceeds threshold",
            inputs=[
                io.Image.Input("image"),
                io.String.Input(
                    "perceptual_hash",
                    default="",
                    multiline=False,
                ),
                io.Float.Input(
                    "delta",
                    default=0.05,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    display_mode=io.NumberDisplay.slider
                ),
                io.Combo.Input(
                    "hash_function",
                    options=["dhash"],
                    default="dhash"
                ),
            ],
            outputs=[],
            hidden=[io.Hidden.unique_id],
            is_output_node=True,
        )

    @classmethod
    def _calculate_dhash(cls, image_tensor: torch.Tensor, hash_size: int = 8) -> str:
        """
        Calculate difference hash (dHash) using minimal dependencies

        Args:
            image_tensor: ComfyUI IMAGE format [B,H,W,C]
            hash_size: Size of hash grid (8 = 64-bit hash)

        Returns:
            Binary string representing the hash
        """
        from PIL import Image
        import numpy as np

        # Convert first image in batch to PIL Image
        img_np = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img_np)

        # Convert to grayscale and resize to hash_size+1 x hash_size
        # We need one extra column to calculate horizontal differences
        img = img.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = np.array(img)

        # Calculate horizontal gradients (compare each pixel with its right neighbor)
        diff = pixels[:, 1:] > pixels[:, :-1]

        # Convert boolean array to binary string
        return ''.join(['1' if d else '0' for row in diff for d in row])

    @classmethod
    def _compare_hashes(cls, hash1: str, hash2: str) -> float:
        """
        Compare two hashes using Hamming distance

        Args:
            hash1: First hash string
            hash2: Second hash string

        Returns:
            Difference as a float between 0.0 (identical) and 1.0 (completely different)
        """
        if not hash1 or not hash2:
            return 1.0  # Maximum difference if either is empty

        if len(hash1) != len(hash2):
            raise ValueError(f"Hash length mismatch: {len(hash1)} vs {len(hash2)}")

        # Calculate Hamming distance (number of differing bits)
        differences = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

        # Return as normalized difference (0.0 to 1.0)
        return differences / len(hash1)

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        perceptual_hash: str,
        delta: float,
        hash_function: str,
    ) -> io.NodeOutput:
        """Validate image against perceptual hash"""

        # Calculate hash based on selected function
        if hash_function == "dhash":
            calculated_hash = cls._calculate_dhash(image)
        else:
            raise ValueError(f"Unknown hash function: {hash_function}")

        # If no expected hash provided, just show the calculated hash
        if not perceptual_hash or perceptual_hash.strip() == "":
            # Send preview image
            api_sync = ComfyAPISync()
            api_sync.execution.set_progress(
                value=1.0,
                max_value=1.0,
                preview_image=image
            )

            # Send calculated hash as text
            PromptServer.instance.send_progress_text(
                calculated_hash,
                cls.hidden.unique_id
            )

            return io.NodeOutput()

        # Compare hashes
        difference = cls._compare_hashes(calculated_hash, perceptual_hash)

        # Check if difference exceeds threshold
        if difference > delta:
            hamming_distance = int(difference * len(calculated_hash))

            # Send preview image BEFORE throwing error
            api_sync = ComfyAPISync()
            api_sync.execution.set_progress(
                value=1.0,
                max_value=1.0,
                preview_image=image
            )

            # Send error details as text BEFORE throwing error
            error_message = (
                f"❌ Image hash mismatch!\n"
                f"Difference: {difference:.4f} > threshold: {delta:.4f}\n"
                f"Expected: {perceptual_hash}\n"
                f"Actual:   {calculated_hash}\n"
                f"Hamming distance: {hamming_distance} bits out of {len(calculated_hash)}"
            )

            PromptServer.instance.send_progress_text(
                calculated_hash,
                cls.hidden.unique_id
            )

            # NOW throw the error - previews already sent
            raise ValueError(error_message)

        # Success - send preview image
        api_sync = ComfyAPISync()
        api_sync.execution.set_progress(
            value=1.0,
            max_value=1.0,
            preview_image=image
        )

        # Send success message with hash details
        success_text = (
            f"✅ Test Passed\n"
            f"Hash: {calculated_hash}\n"
            f"Difference: {difference:.4f} (threshold: {delta:.4f})"
        )

        PromptServer.instance.send_progress_text(
            success_text,
            cls.hidden.unique_id
        )

        # Return empty output (previews already sent)
        return io.NodeOutput()


class AssertNotEqual(io.ComfyNode):
    """Output node that checks if two inputs are NOT equal"""

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


class AssertContainsColor(io.ComfyNode):
    """Output node that checks if an image contains a specific color"""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AssertContainsColor",
            display_name="Assert Contains Color",
            category="testing",
            description="Checks if image contains pixels of a specific color (useful for openpose, segmentation)",
            inputs=[
                io.Image.Input("image"),
                io.String.Input(
                    "color",
                    default="#FF0000",
                ),
                io.Int.Input(
                    "tolerance",
                    default=10,
                    min=0,
                    max=255,
                    display_mode=io.NumberDisplay.slider,
                ),
                io.Int.Input(
                    "min_pixels",
                    default=1,
                    min=1,
                    max=1000000,
                    display_mode=io.NumberDisplay.number,
                ),
            ],
            outputs=[],
            hidden=[io.Hidden.unique_id],
            is_output_node=True,
        )

    @classmethod
    def _parse_color(cls, color_str: str) -> tuple[int, int, int]:
        """Parse color string to RGB tuple (0-255 range)"""
        color_str = color_str.strip()

        # Handle hex format (#RRGGBB or RRGGBB)
        if color_str.startswith("#"):
            color_str = color_str[1:]

        if len(color_str) == 6:
            try:
                r = int(color_str[0:2], 16)
                g = int(color_str[2:4], 16)
                b = int(color_str[4:6], 16)
                return (r, g, b)
            except ValueError:
                pass

        # Handle RGB tuple format (r,g,b) or (r, g, b)
        if "," in color_str:
            color_str = color_str.strip("()[] ")
            parts = [p.strip() for p in color_str.split(",")]
            if len(parts) == 3:
                try:
                    return (int(parts[0]), int(parts[1]), int(parts[2]))
                except ValueError:
                    pass

        raise ValueError(f"Invalid color format: '{color_str}'. Use hex (#FF0000) or RGB (255,0,0)")

    @classmethod
    def _color_distance(cls, c1: tuple[int, int, int], c2: tuple[int, int, int]) -> float:
        """Calculate Euclidean distance between two colors"""
        return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2 + (c1[2] - c2[2]) ** 2) ** 0.5

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        color: str,
        tolerance: int,
        min_pixels: int,
    ) -> io.NodeOutput:
        """Check if image contains the specified color"""
        import numpy as np

        # Parse target color
        target_rgb = cls._parse_color(color)

        # Convert image tensor to numpy (first image in batch)
        # Image tensor is [B,H,W,C] with values 0-1
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)

        # Count pixels within tolerance
        matching_pixels = 0
        height, width, _ = img_np.shape

        for y in range(height):
            for x in range(width):
                pixel_rgb = (int(img_np[y, x, 0]), int(img_np[y, x, 1]), int(img_np[y, x, 2]))
                if cls._color_distance(pixel_rgb, target_rgb) <= tolerance:
                    matching_pixels += 1
                    if matching_pixels >= min_pixels:
                        # Early exit once we've found enough
                        PromptServer.instance.send_progress_text(
                            f"✅ Test Passed\nFound {matching_pixels}+ pixels matching {color}",
                            cls.hidden.unique_id
                        )
                        return io.NodeOutput()

        # Not enough matching pixels found
        raise ValueError(
            f"Color {color} not found in image.\n"
            f"Found {matching_pixels} matching pixels (need at least {min_pixels}).\n"
            f"Tolerance: {tolerance}"
        )


class AssertTensorShape(io.ComfyNode):
    """Output node that checks tensor dimensions"""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AssertTensorShape",
            display_name="Assert Tensor Shape",
            category="testing",
            description="Checks if tensor has expected dimensions (-1 = any value accepted)",
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
        if len(shape) != 4:
            raise ValueError(f"Expected 4D tensor [B,H,W,C], got shape {list(shape)}")

        actual_batch, actual_height, actual_width, actual_channels = shape
        errors = []

        if batch != -1 and actual_batch != batch:
            errors.append(f"batch: expected {batch}, got {actual_batch}")
        if height != -1 and actual_height != height:
            errors.append(f"height: expected {height}, got {actual_height}")
        if width != -1 and actual_width != width:
            errors.append(f"width: expected {width}, got {actual_width}")
        if channels != -1 and actual_channels != channels:
            errors.append(f"channels: expected {channels}, got {actual_channels}")

        if errors:
            raise ValueError(f"Tensor shape mismatch:\n" + "\n".join(errors) + f"\nActual shape: {list(shape)}")

        PromptServer.instance.send_progress_text(
            f"✅ Test Passed\nShape: [{actual_batch}, {actual_height}, {actual_width}, {actual_channels}]",
            cls.hidden.unique_id
        )
        return io.NodeOutput()


class AssertInRange(io.ComfyNode):
    """Output node that checks if all tensor values are within bounds"""

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
