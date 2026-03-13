from typing import Any

from comfy_api.v0_0_2 import io, ui, ComfyAPISync
from server import PromptServer
import torch
from pathlib import Path


def generate_synthetic_benchmark_pattern(width=640, height=640, batch_size=1, device="cpu"):
    """Generate a multi-zone test pattern for diffusion/image pipeline QA.

    Zones (top to bottom):
      - Color fidelity: 8 primary/secondary/neutral color patches
      - Luminance ramp: smooth 0-255 gradient across width
      - Detail stress: 5 high-frequency checkerboard variants
      - Texture stimulus: uniform noise, gaussian noise, upscaled low-res, diagonal gradient
      - Geometry stability: white circle on black background
    """

    img = torch.zeros((batch_size, height, width, 3), dtype=torch.uint8, device=device)

    h_color = int(height * 0.20)
    h_luma = int(height * 0.10)
    h_detail = int(height * 0.25)
    h_texture = int(height * 0.25)
    h_shapes = height - (h_color + h_luma + h_detail + h_texture)

    y = 0

    # Color fidelity
    colors = torch.tensor([
        (255,0,0),
        (0,255,0),
        (0,0,255),
        (0,255,255),
        (255,0,255),
        (255,255,0),
        (128,128,128),
        (198,134,66)
    ], dtype=torch.uint8, device=device)

    cols = len(colors)
    w_patch = width // cols

    for i,c in enumerate(colors):
        x0 = i*w_patch
        x1 = width if i == cols-1 else (i+1)*w_patch
        img[:, y:y+h_color, x0:x1] = c

    y += h_color

    # Luminance ramp
    ramp = torch.linspace(0,255,width,device=device).to(torch.uint8)
    ramp = ramp.unsqueeze(0).repeat(h_luma,1)
    ramp = ramp.unsqueeze(-1).repeat(1,1,3)

    img[:, y:y+h_luma] = ramp

    y += h_luma

    # Detail stress
    segments = 5
    seg_w = width // segments

    for i in range(segments):

        x0 = i * seg_w
        x1 = width if i == segments-1 else (i+1) * seg_w

        xx = torch.arange(x1-x0, device=device).view(1,-1)
        yy = torch.arange(h_detail, device=device).view(-1,1)

        if i == 0:
            pattern = ((xx + yy) % 2) * 255
        elif i == 1:
            pattern = ((xx//2 + yy//2) % 2) * 255
        elif i == 2:
            pattern = (xx % 2) * 255
        elif i == 3:
            pattern = (yy % 2) * 255
        else:
            pattern = ((xx + yy*2) % 2) * 255

        pattern = pattern.to(torch.uint8)
        pattern = pattern.unsqueeze(-1).repeat(1,1,3)

        img[:, y:y+h_detail, x0:x1] = pattern

    y += h_detail

    # Texture tests
    textures = 4
    block_w = width // textures

    noise = torch.randint(0,256,(batch_size,h_texture,block_w,3),dtype=torch.uint8,device=device)
    img[:, y:y+h_texture, 0:block_w] = noise

    g = torch.randn((batch_size,h_texture,block_w,3),device=device)*40+128
    g = torch.clamp(g,0,255).to(torch.uint8)
    img[:, y:y+h_texture, block_w:block_w*2] = g

    scale = 8
    small = torch.randint(0,256,(batch_size,h_texture//scale,block_w//scale,3),dtype=torch.uint8,device=device)

    low = torch.nn.functional.interpolate(
        small.permute(0,3,1,2).float(),
        size=(h_texture,block_w),
        mode="bilinear"
    ).permute(0,2,3,1).to(torch.uint8)

    img[:, y:y+h_texture, block_w*2:block_w*3] = low

    gx = torch.linspace(0,255,block_w,device=device).view(1,-1)
    gy = torch.linspace(0,255,h_texture,device=device).view(-1,1)

    grad = ((gx+gy)/2).to(torch.uint8)
    grad = grad.unsqueeze(-1).repeat(1,1,3)

    img[:, y:y+h_texture, block_w*3:width] = grad

    y += h_texture

    # Geometry stability
    center_y = y + h_shapes//2
    center_x = width//2
    radius = min(width,height)//10

    yy = torch.arange(y, y+h_shapes, device=device).view(-1,1)
    xx = torch.arange(width, device=device).view(1,-1)

    mask = ((xx-center_x)**2 + (yy-center_y)**2) < radius**2
    mask = mask.unsqueeze(0).unsqueeze(-1)

    img[:, y:y+h_shapes] = torch.where(
        mask,
        torch.tensor((255,255,255), dtype=torch.uint8, device=device),
        img[:, y:y+h_shapes]
    )

    return img


def generate_video_benchmark_pattern(width=640, height=480, batch_size=1, device="cpu"):
    """Generate a broadcast-style SMPTE color bar test pattern for video pipeline QA.

    Zones (top to bottom):
      - Calibration bars: 7 SMPTE-style color bars (gray, yellow, cyan, green, magenta, red, blue)
      - Chroma row: complementary/black alternating bars for chroma subsampling detection
      - Luminance / Blacks / Clipping: 7-step greyscale ramp with near-white clip swatch
    """

    # Colors
    top_colors = torch.tensor([
        (128,128,128), # gray
        (255,255,0),   # yellow
        (0,255,255),   # cyan
        (0,255,0),     # green
        (255,0,255),   # magenta
        (255,0,0),     # red
        (0,0,255)      # blue
    ], dtype=torch.uint8, device=device)

    mid_colors = torch.tensor([
        (0,0,255),
        (0,0,0),
        (255,0,255),
        (0,0,0),
        (0,255,255),
        (0,0,0),
        (200,200,200)
    ], dtype=torch.uint8, device=device)

    bottom_colors = torch.tensor([
        (255,255,255),
        (210,210,210),
        (170,170,170),
        (130,130,130),
        (90,90,90),
        (50,50,50),
        (0,0,0)
    ], dtype=torch.uint8, device=device)

    img = torch.zeros((batch_size, height, width, 3), dtype=torch.uint8, device=device)
    top_h = int(height * 0.66)
    mid_h = int(height * 0.08)
    bottom_h = height - top_h - mid_h

    # Calibration
    bar_w = width // len(top_colors)

    for i, color in enumerate(top_colors):
        x0 = i * bar_w
        x1 = (i + 1) * bar_w if i < len(top_colors) - 1 else width
        img[:, 0:top_h, x0:x1, :] = color

    # Chroma
    bar_w_mid = width // len(mid_colors)

    for i, color in enumerate(mid_colors):
        x0 = i * bar_w_mid
        x1 = (i + 1) * bar_w_mid if i < len(mid_colors) - 1 else width
        img[:, top_h:top_h + mid_h, x0:x1, :] = color

    # Luminance / Blacks / Clipping
    bar_w_bot = width // len(bottom_colors)

    for i, color in enumerate(bottom_colors):
        x0 = i * bar_w_bot
        x1 = (i + 1) * bar_w_bot if i < len(bottom_colors) - 1 else width
        y0 = top_h + mid_h
        y1 = height

        img[:, y0:y1, x0:x1, :] = color

        # clip swatch
        if i == 0:
            square_size = int(bar_w_bot * 0.5)
            cx = x0 + bar_w_bot // 2
            cy = y0 + bottom_h // 2

            xs0 = cx - square_size // 2
            xs1 = cx + square_size // 2
            ys0 = cy - square_size // 2
            ys1 = cy + square_size // 2

            img[:, ys0:ys1, xs0:xs1, :] = torch.tensor(
                (245,245,245), dtype=torch.uint8, device=device
            )

    return img


class TestImageGenerator(io.ComfyNode):
    """Generates test images based on selected type"""

    SKILL_DOC = """
Generates deterministic test images with known properties. Use instead of `LoadImage`
when you need reproducible inputs without external file dependencies.

### Inputs

- `image_type` (COMBO) — The type of image to generate:
  - `"black"` — Solid black (all zeros). Good for edge-case and baseline tests.
  - `"white"` — Solid white (all ones). Good for edge-case and baseline tests.
  - `"noise"` — Seeded random noise, deterministic for a given seed. Good for
    general-purpose image processing tests.
  - `"face"` — Loads a face photo bundled with the framework. Use for face detection,
    segmentation, or portrait-specific nodes.
  - `"synthetic_benchmark"` — Multi-zone QA pattern with color fidelity patches,
    luminance ramp, detail stress checkerboards, texture stimuli (noise, Gaussian,
    upscaled, gradient), and a geometry circle. **Use this for testing image editing,
    filtering, color correction, and style transfer nodes** — it provides diverse
    visual features that reveal artifacts across different processing domains.
  - `"video_benchmark"` — Broadcast-style SMPTE color bar pattern with calibration
    bars, chroma row, and luminance/blacks/clipping sections. **Use this for testing
    video processing, frame interpolation, and diffusion/inference nodes** — the
    standardized color bars make it easy to detect color shifts, banding, and temporal
    artifacts.
  Default: `"black"`.
- `width` (INT) — Image width. Range: 64–4096, step 8. Default: `512`.
- `height` (INT) — Image height. Range: 64–4096, step 8. Default: `512`.
- `batch_size` (INT) — Number of images in batch. Range: 1–64. Default: `1`.
  Use `batch_size > 1` for testing video/batch processing nodes.
- `seed` (INT) — Random seed for `"noise"` type. Ignored by other types. Default: `0`.

### Outputs

- `image` (IMAGE) — Tensor `[B,H,W,3]`, float32, values in `[0.0, 1.0]`.

### When to Use

- As the primary image source for test workflows. Prefer this over `LoadImage` because
  it requires no external files and is fully deterministic.
- Choose `image_type` based on what the node under test does:
  - Image filters/editing → `"synthetic_benchmark"`
  - Video processing → `"video_benchmark"` with `batch_size > 1`
  - Face/portrait nodes → `"face"`
  - Edge-case testing → `"black"` or `"white"`
  - General purpose → `"noise"`

### Example

```
[TestImageGenerator(image_type="synthetic_benchmark", width=512, height=512)] → [NodeUnderTest] → [AssertTensorShape(...)]
```
"""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TestImageGenerator",
            display_name="Test Image Generator",
            category="testing",
            description="Generate test images: solid colors, noise, face, synthetic benchmark pattern, or video benchmark pattern",
            inputs=[
                io.Combo.Input(
                    "image_type",
                    options=["black", "white", "noise", "face", "synthetic_benchmark", "video_benchmark"],
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
                    "batch_size",
                    default=1,
                    min=1,
                    max=64,
                    step=1,
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
            ],
            is_dev_only=True,
        )

    @classmethod
    def execute(
        cls,
        image_type: str,
        width: int,
        height: int,
        batch_size: int,
        seed: int,
    ) -> io.NodeOutput:
        """Generate test image based on type"""

        # Create image tensor with shape [B,H,W,C] where C=3 (RGB)
        batch = batch_size
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

        elif image_type == "synthetic_benchmark":
            # Multi-zone diffusion/image pipeline QA pattern
            image_uint8 = generate_synthetic_benchmark_pattern(width, height, batch)
            image = image_uint8.float() / 255.0

        elif image_type == "video_benchmark":
            # Broadcast-style color bar pattern for video pipeline QA
            image_uint8 = generate_video_benchmark_pattern(width, height, batch)
            image = image_uint8.float() / 255.0

        elif image_type == "face":
            # Load face image from file
            # face.png lives in the framework root (one level up from nodes/)
            framework_dir = Path(__file__).parent.parent
            face_path = framework_dir / "face.png"

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
                image = torch.from_numpy(image_np).unsqueeze(0).expand(batch, -1, -1, -1)

        else:
            # Default to black if unknown type
            image = torch.zeros(batch, height, width, channels)

        # Ensure values are in valid range [0, 1]
        image = torch.clamp(image, 0.0, 1.0)

        return io.NodeOutput(image)


class TestMaskGenerator(io.ComfyNode):
    """Generates test masks with known properties for assertion-based testing.

    Mask tensor format: float32 [B, H, W], values in [0, 1].

    Mask types and their properties:
      solid_black — All zeros. Coverage: 0%.
      solid_white — All ones. Coverage: 100%.
      circle — Centered filled circle. Coverage depends on radius relative to image size (~78% at default).
      gradient_horizontal — Left-to-right linear ramp 0→1. Coverage: ~100% (all pixels > 0 except col 0).
      gradient_vertical — Top-to-bottom linear ramp 0→1. Coverage: ~100%.
      checkerboard — Alternating 0/1 blocks. Coverage: ~50%. Always binary.
      noise — Random values in [0, 1] from seed. Coverage: ~100%.
      half_left — Left half = 1, right half = 0. Coverage: 50%. Always binary.
      half_top — Top half = 1, bottom half = 0. Coverage: 50%. Always binary.
    """

    SKILL_DOC = """
Generates deterministic test masks with known coverage and properties. Use instead of
`LoadImage` mask output when you need reproducible mask inputs.

### Inputs

- `mask_type` (COMBO) — The type of mask to generate:
  - `"solid_black"` — All zeros. Coverage: 0%. Binary.
  - `"solid_white"` — All ones. Coverage: 100%. Binary.
  - `"circle"` — Centered filled circle, radius = `min(H,W) * 0.45`. Coverage: ~78%.
    Binary.
  - `"gradient_horizontal"` — Left-to-right linear ramp 0→1. Coverage: ~100%. Soft.
  - `"gradient_vertical"` — Top-to-bottom linear ramp 0→1. Coverage: ~100%. Soft.
  - `"checkerboard"` — Alternating 0/1 blocks (8px). Coverage: ~50%. Binary.
  - `"noise"` — Seeded random values in `[0, 1]`. Coverage: ~100%. Soft.
  - `"half_left"` — Left half = 1, right half = 0. Coverage: 50%. Binary.
  - `"half_top"` — Top half = 1, bottom half = 0. Coverage: 50%. Binary.
  Default: `"solid_white"`.
- `width` (INT) — Mask width. Range: 64–4096, step 8. Default: `512`.
- `height` (INT) — Mask height. Range: 64–4096, step 8. Default: `512`.
- `batch_size` (INT) — Number of masks in batch. Range: 1–64. Default: `1`.
- `seed` (INT) — Random seed for `"noise"` type. Default: `0`.

### Outputs

- `mask` (MASK) — Tensor `[B,H,W]`, float32, values in `[0.0, 1.0]`.

### When to Use

- As the primary mask source for testing mask-processing nodes.
- Choose `mask_type` based on what you need to test:
  - Threshold/binary operations → `"gradient_horizontal"` (soft input, expect binary output)
  - Mask compositing → `"circle"` + `"half_left"` (two distinct shapes)
  - Blur/feather → `"circle"` or `"checkerboard"` (binary input, expect soft output)
  - Edge cases → `"solid_black"` or `"solid_white"`
  - General purpose → `"noise"` or `"circle"`

### Known Coverage Values (for `AssertMaskCoverage`)

| Mask Type | Coverage | Binary? |
|-----------|----------|---------|
| `solid_black` | 0% | Yes |
| `solid_white` | 100% | Yes |
| `circle` | ~78% | Yes |
| `checkerboard` | ~50% | Yes |
| `half_left` | 50% | Yes |
| `half_top` | 50% | Yes |
| `gradient_horizontal` | ~100% | No |
| `gradient_vertical` | ~100% | No |
| `noise` | ~100% | No |

### Example

```
[TestMaskGenerator(mask_type="circle")] → [MaskBlur(amount=5)] → [AssertMaskFuzzy(edge_tolerance=0.1, max_soft_percentage=15)]
```
"""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TestMaskGenerator",
            display_name="Test Mask Generator",
            category="testing",
            description="Generate test masks: solid, circle, gradient, checkerboard, noise, or half masks",
            inputs=[
                io.Combo.Input(
                    "mask_type",
                    options=[
                        "solid_black", "solid_white", "circle",
                        "gradient_horizontal", "gradient_vertical",
                        "checkerboard", "noise", "half_left", "half_top",
                    ],
                    default="solid_white",
                ),
                io.Int.Input(
                    "width",
                    default=512,
                    min=64,
                    max=4096,
                    step=8,
                    display_mode=io.NumberDisplay.number,
                ),
                io.Int.Input(
                    "height",
                    default=512,
                    min=64,
                    max=4096,
                    step=8,
                    display_mode=io.NumberDisplay.number,
                ),
                io.Int.Input(
                    "batch_size",
                    default=1,
                    min=1,
                    max=64,
                    step=1,
                    display_mode=io.NumberDisplay.number,
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFF,
                    display_mode=io.NumberDisplay.number,
                ),
            ],
            outputs=[
                io.Mask.Output(display_name="Mask"),
            ],
            is_dev_only=True,
        )

    @classmethod
    def execute(
        cls,
        mask_type: str,
        width: int,
        height: int,
        batch_size: int,
        seed: int,
    ) -> io.NodeOutput:
        """Generate test mask based on type"""

        if mask_type == "solid_black":
            mask = torch.zeros(batch_size, height, width)

        elif mask_type == "solid_white":
            mask = torch.ones(batch_size, height, width)

        elif mask_type == "circle":
            # Centered filled circle with radius = min(H, W) * 0.45
            gy = torch.arange(height, dtype=torch.float32).unsqueeze(1)
            gx = torch.arange(width, dtype=torch.float32).unsqueeze(0)
            cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
            radius = min(height, width) * 0.45
            dist_sq = (gy - cy).pow(2) + (gx - cx).pow(2)
            single = (dist_sq <= radius * radius).float()
            mask = single.unsqueeze(0).expand(batch_size, -1, -1)

        elif mask_type == "gradient_horizontal":
            row = torch.linspace(0.0, 1.0, steps=width).unsqueeze(0)
            single = row.expand(height, -1)
            mask = single.unsqueeze(0).expand(batch_size, -1, -1)

        elif mask_type == "gradient_vertical":
            col = torch.linspace(0.0, 1.0, steps=height).unsqueeze(1)
            single = col.expand(-1, width)
            mask = single.unsqueeze(0).expand(batch_size, -1, -1)

        elif mask_type == "checkerboard":
            gy = torch.arange(height)
            gx = torch.arange(width)
            # 8-pixel blocks
            single = ((gy.unsqueeze(1) // 8 + gx.unsqueeze(0) // 8) % 2).float()
            mask = single.unsqueeze(0).expand(batch_size, -1, -1)

        elif mask_type == "noise":
            generator = torch.Generator()
            generator.manual_seed(seed)
            mask = torch.rand(batch_size, height, width, generator=generator)

        elif mask_type == "half_left":
            single = torch.zeros(height, width)
            single[:, : width // 2] = 1.0
            mask = single.unsqueeze(0).expand(batch_size, -1, -1)

        elif mask_type == "half_top":
            single = torch.zeros(height, width)
            single[: height // 2, :] = 1.0
            mask = single.unsqueeze(0).expand(batch_size, -1, -1)

        else:
            mask = torch.zeros(batch_size, height, width)

        mask = torch.clamp(mask, 0.0, 1.0)
        return io.NodeOutput(mask)
