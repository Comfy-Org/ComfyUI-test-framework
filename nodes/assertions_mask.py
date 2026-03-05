from typing import Any

from comfy_api.v0_0_2 import io
from server import PromptServer
import torch


class AssertMaskCoverage(io.ComfyNode):
    """Output node that checks mask coverage (percentage of non-zero pixels).

    Coverage is defined as the fraction of pixels with value > 0, expressed as a
    percentage [0, 100]. Useful for verifying segmentation, threshold, and masking nodes.

    Example test patterns:
      - Segmentation node on known image → coverage should be within expected range
      - Threshold node on gradient mask → coverage depends on threshold value
      - Invert mask on solid_white → coverage should be 0%
      - Circle mask at default size → coverage should be ~63%
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AssertMaskCoverage",
            display_name="Assert Mask Coverage",
            category="testing",
            description="Check that mask coverage (% of non-zero pixels) is within an expected range",
            inputs=[
                io.Mask.Input("mask"),
                io.Float.Input(
                    "min_coverage",
                    default=0.0,
                    min=0.0,
                    max=100.0,
                    step=0.1,
                    display_mode=io.NumberDisplay.number,
                ),
                io.Float.Input(
                    "max_coverage",
                    default=100.0,
                    min=0.0,
                    max=100.0,
                    step=0.1,
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
        mask: torch.Tensor,
        min_coverage: float,
        max_coverage: float,
    ) -> io.NodeOutput:
        """Check mask coverage percentage."""

        if not isinstance(mask, torch.Tensor):
            raise ValueError(f"Input is not a tensor, got {type(mask).__name__}")

        if mask.ndim not in (2, 3):
            raise ValueError(
                f"Expected mask with 2 or 3 dimensions [H,W] or [B,H,W], got shape {list(mask.shape)}"
            )

        total_pixels = mask.numel()
        nonzero_pixels = (mask > 0).sum().item()
        coverage = (nonzero_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0

        errors = []
        if coverage < min_coverage:
            errors.append(f"Coverage {coverage:.2f}% is below minimum {min_coverage:.2f}%")
        if coverage > max_coverage:
            errors.append(f"Coverage {coverage:.2f}% exceeds maximum {max_coverage:.2f}%")

        if errors:
            raise ValueError(
                "Mask coverage assertion failed:\n" + "\n".join(errors) +
                f"\nActual coverage: {coverage:.2f}% ({nonzero_pixels}/{total_pixels} pixels)"
            )

        PromptServer.instance.send_progress_text(
            f"✅ Test Passed\nCoverage: {coverage:.2f}% ({nonzero_pixels}/{total_pixels} pixels)\n"
            f"Expected: [{min_coverage:.1f}%, {max_coverage:.1f}%]",
            cls.hidden.unique_id
        )
        return io.NodeOutput()


class AssertMaskBinary(io.ComfyNode):
    """Output node that checks all mask values are exactly 0 or 1.

    Useful for verifying threshold nodes, boolean mask operations, and any node
    that should produce a hard mask. Fails if any pixel has a value between 0 and 1.

    Example test patterns:
      - ThresholdMask output → should always be binary
      - MaskComposite with 'add' mode → may or may not be binary depending on inputs
      - Checkerboard test mask → always binary
      - Gradient test mask → NOT binary (will fail, useful as a negative test)
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AssertMaskBinary",
            display_name="Assert Mask Binary",
            category="testing",
            description="Check that all mask values are exactly 0 or 1 (hard mask)",
            inputs=[
                io.Mask.Input("mask"),
            ],
            outputs=[],
            hidden=[io.Hidden.unique_id],
            is_output_node=True,
            is_dev_only=True,
        )

    @classmethod
    def execute(
        cls,
        mask: torch.Tensor,
    ) -> io.NodeOutput:
        """Check that all mask values are exactly 0 or 1."""

        if not isinstance(mask, torch.Tensor):
            raise ValueError(f"Input is not a tensor, got {type(mask).__name__}")

        if mask.ndim not in (2, 3):
            raise ValueError(
                f"Expected mask with 2 or 3 dimensions [H,W] or [B,H,W], got shape {list(mask.shape)}"
            )

        is_binary = ((mask == 0.0) | (mask == 1.0)).all().item()

        if not is_binary:
            non_binary = mask[(mask != 0.0) & (mask != 1.0)]
            sample_values = non_binary[:5].tolist()
            count = non_binary.numel()
            raise ValueError(
                f"Mask is not binary: {count} pixels have values other than 0 or 1.\n"
                f"Sample non-binary values: {[f'{v:.6f}' for v in sample_values]}\n"
                f"Value range: [{mask.min().item():.6f}, {mask.max().item():.6f}]"
            )

        ones_count = (mask == 1.0).sum().item()
        total = mask.numel()
        PromptServer.instance.send_progress_text(
            f"✅ Test Passed\nMask is binary: {ones_count}/{total} pixels are 1.0 "
            f"({ones_count / total * 100:.1f}% coverage)",
            cls.hidden.unique_id
        )
        return io.NodeOutput()


class AssertMaskFuzzy(io.ComfyNode):
    """Output node that validates soft-edge masks from blur, feather, or anti-aliasing nodes.

    Classifies every pixel into three zones based on edge_tolerance:
      - "near 0" : value in [0, edge_tolerance]
      - "near 1" : value in [1 - edge_tolerance, 1]
      - "soft"   : everything else (the grey boundary region)

    Asserts that the percentage of soft pixels does not exceed max_soft_percentage.

    Recommended thresholds by use case:
      - Light feather (2-4px)   → edge_tolerance=0.1, max_soft_percentage=5
      - Medium blur (5-10px)    → edge_tolerance=0.1, max_soft_percentage=15
      - Heavy blur / Gaussian   → edge_tolerance=0.05, max_soft_percentage=30
      - Anti-aliased edges      → edge_tolerance=0.1, max_soft_percentage=10

    Example test patterns:
      - FeatherMask on circle → most pixels stay hard, only boundary ring is soft
      - GaussianBlur on half_left → wide soft band at midpoint
      - MaskComposite blend → controlled transition zone
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AssertMaskFuzzy",
            display_name="Assert Mask Fuzzy",
            category="testing",
            description="Check that a soft-edge mask has a controlled amount of grey boundary pixels",
            inputs=[
                io.Mask.Input("mask"),
                io.Float.Input(
                    "edge_tolerance",
                    default=0.1,
                    min=0.001,
                    max=0.5,
                    step=0.01,
                    display_mode=io.NumberDisplay.number,
                ),
                io.Float.Input(
                    "max_soft_percentage",
                    default=15.0,
                    min=0.0,
                    max=100.0,
                    step=0.1,
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
        mask: torch.Tensor,
        edge_tolerance: float,
        max_soft_percentage: float,
    ) -> io.NodeOutput:
        """Check that soft-edge pixels are within acceptable bounds."""

        if not isinstance(mask, torch.Tensor):
            raise ValueError(f"Input is not a tensor, got {type(mask).__name__}")

        if mask.ndim not in (2, 3):
            raise ValueError(
                f"Expected mask with 2 or 3 dimensions [H,W] or [B,H,W], got shape {list(mask.shape)}"
            )

        total = mask.numel()
        near_zero = (mask <= edge_tolerance).sum().item()
        near_one = (mask >= 1.0 - edge_tolerance).sum().item()
        soft = total - near_zero - near_one
        soft_pct = (soft / total) * 100.0 if total > 0 else 0.0

        if soft_pct > max_soft_percentage:
            raise ValueError(
                f"Mask has too many soft-edge pixels:\n"
                f"Soft pixels: {soft}/{total} ({soft_pct:.2f}%)\n"
                f"Maximum allowed: {max_soft_percentage:.1f}%\n"
                f"Near-0 (≤{edge_tolerance}): {near_zero} ({near_zero / total * 100:.1f}%)\n"
                f"Near-1 (≥{1.0 - edge_tolerance:.3f}): {near_one} ({near_one / total * 100:.1f}%)\n"
                f"Value range: [{mask.min().item():.6f}, {mask.max().item():.6f}]"
            )

        PromptServer.instance.send_progress_text(
            f"✅ Test Passed\n"
            f"Soft pixels: {soft}/{total} ({soft_pct:.2f}%, max {max_soft_percentage:.1f}%)\n"
            f"Near-0: {near_zero} ({near_zero / total * 100:.1f}%)  "
            f"Near-1: {near_one} ({near_one / total * 100:.1f}%)",
            cls.hidden.unique_id
        )
        return io.NodeOutput()
