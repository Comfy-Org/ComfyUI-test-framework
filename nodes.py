from typing import Any

from comfy_api.latest import io, ui
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


class TestMustExecute(io.ComfyNode):
    """Pass-through node that accepts any input and returns it unchanged"""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TestMustExecute",
            display_name="Test Must Execute",
            category="testing",
            description="Pass-through node that returns its input unchanged",
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


class TestEqual(io.ComfyNode):
    """Output node that checks if two inputs are equal"""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TestEqual",
            display_name="Test Equal",
            category="testing",
            description="Checks if two inputs are equal, throws error if not",
            inputs=[
                io.AnyType.Input("input1"),
                io.AnyType.Input("input2"),
            ],
            outputs=[],
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
        return io.NodeOutput(ui=ui.PreviewText("✅ Test Passed"))


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
