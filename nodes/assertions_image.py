from typing import Any

from comfy_api.v0_0_2 import io, ui, ComfyAPISync
from server import PromptServer
import torch


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
            is_dev_only=True,
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

            raise ValueError(f"No perceptual hash provided to compare against. Actual hash is {calculated_hash}")

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
            is_dev_only=True,
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
