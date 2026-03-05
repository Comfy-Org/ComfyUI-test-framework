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


class AssertImageSimilarity(io.ComfyNode):
    """Output node that measures image quality degradation between a reference and test image.

    Metrics and recommended thresholds:

      psnr (Peak Signal-to-Noise Ratio, dB) — comparison: >=
        Measures overall pixel-level fidelity. Higher is better.
        - >= 40 dB : near-lossless (e.g. quantize 32 levels ≈ 44 dB)
        - >= 30 dB : acceptable lossy (e.g. light noise std=0.03 ≈ 32 dB, color shift R+0.05 ≈ 31 dB)
        - < 20 dB  : severe degradation (e.g. heavy noise std=0.10 ≈ 22 dB)
        Best for: VAE roundtrip, compression, upscaling.

      ssim (Structural Similarity Index) — comparison: >=
        Perceptual quality metric sensitive to structural changes. Range [0, 1], higher is better.
        - >= 0.95 : near-lossless (e.g. quantize 32 levels ≈ 0.99)
        - >= 0.85 : acceptable lossy
        - < 0.60  : significant structural damage
        Note: SSIM is very sensitive to noise (std=0.03 noise drops SSIM to 0.57–0.74) and
        high-frequency detail destruction (blur on checkerboard patterns). Less sensitive to
        uniform color shifts. Best for: any lossy transform, especially blur/resampling detection.

      mse (Mean Squared Error) — comparison: <=
        Raw pixel difference magnitude. Range [0, 1] for float images, lower is better.
        - <= 0.0001 : near-lossless
        - <= 0.001  : acceptable lossy
        - > 0.01    : significant degradation
        Best for: quick sanity checks, strict numeric validation.

      per_channel_mean_shift (Max per-channel mean delta) — comparison: <=
        Catches color drift invisible to other metrics. Reports max of |R|, |G|, |B| mean shifts.
        - <= 0.01 : negligible color drift
        - <= 0.02 : acceptable for lossy pipelines
        - > 0.03  : noticeable color shift (e.g. R+0.05 shift reads ≈ 0.038)
        Best for: color correction, format conversion, gamma errors, channel swaps.

    The metrics are complementary — choose based on what degradation matters:
      - Pixel fidelity after lossy encode → psnr
      - Structural/perceptual quality → ssim
      - Color accuracy after processing → per_channel_mean_shift
      - Strict numeric bound → mse
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AssertImageSimilarity",
            display_name="Assert Image Similarity",
            category="testing",
            description=(
                "Compare two images using professional quality metrics (PSNR, SSIM, MSE, per-channel stats). "
                "Useful for testing lossy pipelines like VAE encode/decode, upscaling, color correction, compression."
            ),
            inputs=[
                io.Image.Input("reference"),
                io.Image.Input("test_image"),
                io.Combo.Input(
                    "metric",
                    options=["psnr", "ssim", "mse", "per_channel_mean_shift"],
                    default="psnr",
                ),
                io.Float.Input(
                    "threshold",
                    default=30.0,
                    min=-1e6,
                    max=1e6,
                    step=0.1,
                    display_mode=io.NumberDisplay.number,
                ),
                io.Combo.Input(
                    "comparison",
                    options=[">=", "<="],
                    default=">=",
                ),
            ],
            outputs=[],
            hidden=[io.Hidden.unique_id],
            is_output_node=True,
            is_dev_only=True,
        )

    @classmethod
    def _compute_mse(cls, ref: torch.Tensor, test: torch.Tensor) -> float:
        """Mean Squared Error across all pixels and channels."""
        return (ref - test).pow(2).mean().item()

    @classmethod
    def _compute_psnr(cls, ref: torch.Tensor, test: torch.Tensor, max_val: float = 1.0) -> float:
        """Peak Signal-to-Noise Ratio in dB. Returns float('inf') for identical images."""
        mse = cls._compute_mse(ref, test)
        if mse == 0.0:
            return float('inf')
        return 10.0 * torch.log10(torch.tensor(max_val ** 2 / mse)).item()

    @classmethod
    def _gaussian_kernel_1d(cls, size: int, sigma: float, device: torch.device) -> torch.Tensor:
        """Create a 1D Gaussian kernel, normalized to sum to 1."""
        coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2.0
        kernel = torch.exp(-coords.pow(2) / (2.0 * sigma * sigma))
        return kernel / kernel.sum()

    @classmethod
    def _compute_ssim(
        cls,
        ref: torch.Tensor,
        test: torch.Tensor,
        window_size: int = 11,
        sigma: float = 1.5,
    ) -> float:
        """Structural Similarity Index (Wang et al., 2004).

        Uses separable Gaussian convolution for efficiency.
        Operates on the first image in the batch.
        Constants C1, C2 use the standard L=1.0 (float image range).
        """
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # Work with first batch image: [H, W, C] -> [C, H, W]
        x = ref[0].permute(2, 0, 1).unsqueeze(0).float()   # [1, C, H, W]
        y = test[0].permute(2, 0, 1).unsqueeze(0).float()

        channels = x.shape[1]
        device = x.device

        # Build separable 2D Gaussian window from 1D kernel
        kernel_1d = cls._gaussian_kernel_1d(window_size, sigma, device)
        # Outer product -> 2D kernel, then reshape for depthwise conv
        kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)  # [W, W]
        window = kernel_2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1)  # [C, 1, W, W]

        pad = window_size // 2

        mu_x = torch.nn.functional.conv2d(x, window, padding=pad, groups=channels)
        mu_y = torch.nn.functional.conv2d(y, window, padding=pad, groups=channels)

        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y

        sigma_x_sq = torch.nn.functional.conv2d(x * x, window, padding=pad, groups=channels) - mu_x_sq
        sigma_y_sq = torch.nn.functional.conv2d(y * y, window, padding=pad, groups=channels) - mu_y_sq
        sigma_xy = torch.nn.functional.conv2d(x * y, window, padding=pad, groups=channels) - mu_xy

        # Clamp variances to avoid numerical noise going negative
        sigma_x_sq = sigma_x_sq.clamp(min=0.0)
        sigma_y_sq = sigma_y_sq.clamp(min=0.0)

        numerator = (2.0 * mu_xy + C1) * (2.0 * sigma_xy + C2)
        denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)

        ssim_map = numerator / denominator
        return ssim_map.mean().item()

    @classmethod
    def _compute_per_channel_mean_shift(cls, ref: torch.Tensor, test: torch.Tensor) -> tuple[float, list[float]]:
        """Max absolute per-channel mean shift across R, G, B.

        Returns (max_shift, [r_shift, g_shift, b_shift]) in [0, 1] range.
        """
        # First batch image [H, W, C]
        ref_means = ref[0].float().mean(dim=(0, 1))   # [C]
        test_means = test[0].float().mean(dim=(0, 1))
        shifts = (test_means - ref_means).abs()
        per_channel = shifts.tolist()
        return max(per_channel), per_channel

    @classmethod
    def execute(
        cls,
        reference: torch.Tensor,
        test_image: torch.Tensor,
        metric: str,
        threshold: float,
        comparison: str,
    ) -> io.NodeOutput:
        """Compare reference and test image using selected metric."""

        # Validate shapes match
        if reference.shape != test_image.shape:
            raise ValueError(
                f"Image shape mismatch: reference {list(reference.shape)} vs test {list(test_image.shape)}"
            )

        # Compute metric
        if metric == "psnr":
            value = cls._compute_psnr(reference, test_image)
            unit = "dB"
            detail = f"PSNR: {value:.2f} {unit}"
        elif metric == "ssim":
            value = cls._compute_ssim(reference, test_image)
            unit = ""
            detail = f"SSIM: {value:.6f}"
        elif metric == "mse":
            value = cls._compute_mse(reference, test_image)
            unit = ""
            detail = f"MSE: {value:.8f}"
        elif metric == "per_channel_mean_shift":
            value, per_ch = cls._compute_per_channel_mean_shift(reference, test_image)
            unit = ""
            detail = f"Max channel shift: {value:.6f} (R={per_ch[0]:.6f}, G={per_ch[1]:.6f}, B={per_ch[2]:.6f})"
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Evaluate threshold
        if comparison == ">=":
            passed = value >= threshold or value == float('inf')
            fail_reason = f"{value:.6f} < {threshold} (expected >=)"
        else:  # "<="
            passed = value <= threshold
            fail_reason = f"{value:.6f} > {threshold} (expected <=)"

        if not passed:
            raise ValueError(
                f"Image similarity assertion failed ({metric}):\n"
                f"{detail}\n"
                f"Threshold: {comparison} {threshold}\n"
                f"Result: {fail_reason}"
            )

        PromptServer.instance.send_progress_text(
            f"✅ Test Passed\n{detail}\nThreshold: {comparison} {threshold}",
            cls.hidden.unique_id
        )
        return io.NodeOutput()
