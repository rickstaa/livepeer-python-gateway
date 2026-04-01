"""Example: a real-time streaming pipeline using the Livepeer Pipeline SDK.

This demonstrates how to define a StreamPipeline subclass for live
video processing with dynamic parameter updates.

Usage:

    # Print the auto-generated schema
    livepeer schema examples/stream_pipeline.py

    # Start a local HTTP server
    livepeer serve examples/stream_pipeline.py
"""

from livepeer_gateway.runner import Input, Pipeline, StreamPipeline


class StyleTransfer(StreamPipeline):
    """Apply style transfer to live video frames."""

    gpu = "T4"

    def setup(self) -> None:
        """Load the style transfer model."""
        self.current_style = "starry_night"
        self.strength = 0.8
        print("StyleTransfer: setup complete (stub)")

    def on_frame(
        self,
        frame: bytes,
        style: str = Input(
            description="Style to apply",
            default="starry_night",
            choices=["starry_night", "mosaic", "candy", "udnie"],
        ),
        strength: float = Input(
            description="Style strength",
            default=0.8,
            ge=0.0,
            le=1.0,
        ),
    ) -> bytes:
        """Process a single video frame.

        This stub returns the frame unchanged.  A real implementation
        would apply neural style transfer.
        """
        return frame

    def on_params_update(self, params: dict) -> None:
        """Handle live parameter updates from control channel."""
        if "style" in params:
            self.current_style = params["style"]
            print(f"StyleTransfer: switched to style={self.current_style}")
        if "strength" in params:
            self.strength = params["strength"]
            print(f"StyleTransfer: strength={self.strength}")
