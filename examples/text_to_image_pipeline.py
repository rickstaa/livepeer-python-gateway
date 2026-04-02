"""Example: a text-to-image pipeline using the Livepeer Pipeline SDK.

This demonstrates how to define a Pipeline subclass with typed inputs
and outputs.  In a real deployment the ``setup()`` method would load
a diffusion model and ``predict()`` would run actual inference.

Usage:

    # Run a local prediction
    livepeer predict examples/text_to_image_pipeline.py -i prompt="a cat in space"

    # Print the auto-generated schema
    livepeer schema examples/text_to_image_pipeline.py

    # Start a local HTTP server
    livepeer serve examples/text_to_image_pipeline.py

    # Package for deployment (dry run)
    livepeer push examples/text_to_image_pipeline.py --dry-run
"""

from livepeer_gateway.runner import Input, Output, Pipeline


class TextToImage(Pipeline):
    """Generate images from text prompts."""

    pipeline_id = "text-to-image"
    version = "1.0.0"
    description = "Generate images from text prompts using diffusion models"
    gpu = "A100"
    min_vram_gb = 24

    @classmethod
    def prepare_models(cls) -> None:
        """Download model checkpoints during Docker build."""
        print("TextToImage: downloading model weights (stub)")

    def setup(self) -> None:
        """Load the diffusion model.

        In production this would load the weights downloaded by prepare_models().
        """
        print("TextToImage: setup complete (stub)")

    def predict(
        self,
        prompt: str = Input(description="Text description of the image to generate"),
        negative_prompt: str = Input(
            description="Things to avoid in the generated image",
            default="",
        ),
        steps: int = Input(
            description="Number of diffusion steps",
            default=30,
            ge=1,
            le=100,
        ),
        guidance_scale: float = Input(
            description="Classifier-free guidance scale",
            default=7.5,
            ge=1.0,
            le=20.0,
        ),
        width: int = Input(
            description="Image width in pixels",
            default=512,
            choices=[256, 512, 768, 1024],
        ),
        height: int = Input(
            description="Image height in pixels",
            default=512,
            choices=[256, 512, 768, 1024],
        ),
    ) -> Output(type="image", media_type="image/png"):
        """Run text-to-image inference.

        This stub returns a minimal 1x1 PNG.  A real implementation
        would invoke the diffusion model loaded in ``setup()``.
        """
        # Minimal valid 1x1 white PNG for demonstration
        import struct
        import zlib

        def _minimal_png(w: int = 1, h: int = 1) -> bytes:
            raw_row = b"\x00" + b"\xff\xff\xff" * w
            raw_data = raw_row * h
            compressed = zlib.compress(raw_data)

            def _chunk(chunk_type: bytes, data: bytes) -> bytes:
                c = chunk_type + data
                return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

            ihdr_data = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
            return (
                b"\x89PNG\r\n\x1a\n"
                + _chunk(b"IHDR", ihdr_data)
                + _chunk(b"IDAT", compressed)
                + _chunk(b"IEND", b"")
            )

        print(f"TextToImage: generating {width}x{height} image for prompt={prompt!r} "
              f"(steps={steps}, guidance={guidance_scale})")
        return _minimal_png(1, 1)
