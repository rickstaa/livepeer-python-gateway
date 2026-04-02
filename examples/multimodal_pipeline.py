"""Example: a multi-modal streaming pipeline (audio + video).

This demonstrates how to define a StreamPipeline that processes both
audio and video frames from the same trickle stream. Audio frames
update internal state, video frames use that state to produce output.

Usage:

    # Print the auto-generated schema
    livepeer schema examples/multimodal_pipeline.py

    # Start a local HTTP server
    livepeer serve examples/multimodal_pipeline.py
"""

from livepeer_gateway.runner import Input, StreamPipeline


class LipSync(StreamPipeline):
    """Lip sync: audio drives mouth movement on video frames."""

    pipeline_id = "lip-sync"
    version = "1.0.0"
    description = "Audio-driven lip sync on video frames"
    gpu = "T4"

    # Declare both audio and video as inputs, video as output
    inputs = ["video", "audio"]
    outputs = ["video"]

    def setup(self) -> None:
        """Load lip sync model."""
        self._audio_features = None
        print("LipSync: setup complete (stub)")

    def on_video_frame(self, frame, **params):
        """Apply lip sync using latest audio features.

        This is called for every decoded video frame. The audio
        features are updated asynchronously by on_audio_frame().
        """
        # In production: use self._audio_features to drive mouth movement
        return frame  # passthrough stub

    def on_audio_frame(self, frame, **params):
        """Extract phoneme features from audio.

        Returns None — audio doesn't produce output frames,
        it just updates internal state for the video processing.
        """
        # In production: extract phonemes/visemes from audio
        self._audio_features = {"phoneme": "aa", "energy": 0.5}
        return None  # don't produce output from audio

    def on_params_update(self, params):
        if "sensitivity" in params:
            print(f"LipSync: sensitivity={params['sensitivity']}")


class AudioReactiveVisuals(StreamPipeline):
    """Generate visuals that react to audio input."""

    pipeline_id = "audio-reactive"
    version = "1.0.0"
    description = "Audio-reactive visual effects on video"
    gpu = "T4"

    inputs = ["video", "audio"]
    outputs = ["video"]

    def setup(self) -> None:
        self._beat_energy = 0.0
        print("AudioReactiveVisuals: setup complete (stub)")

    def on_video_frame(self, frame, **params):
        """Apply audio-reactive effect to video frame."""
        # In production: use self._beat_energy to modulate effects
        return frame

    def on_audio_frame(self, frame, **params):
        """Analyze audio for beat/energy detection."""
        self._beat_energy = 0.7  # stub
        return None
