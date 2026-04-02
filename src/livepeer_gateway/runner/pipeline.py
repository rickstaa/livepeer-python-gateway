"""Base classes for defining Livepeer pipelines.

A :class:`Pipeline` handles request-response workloads (e.g. text-to-image).
A :class:`StreamPipeline` handles real-time streaming workloads (e.g. live
video style transfer) with dynamic parameter updates.

Developers subclass one of these, implement ``setup()`` and ``predict()``
(or ``on_frame()`` for streams), and the SDK takes care of serving, schema
generation, and deployment.
"""

from __future__ import annotations

import abc
import enum
import logging
from typing import Any, Optional

_LOG = logging.getLogger(__name__)


class PipelineState(str, enum.Enum):
    """Health state of a pipeline instance.

    Orchestrators and health probes use this to decide whether a
    container is ready to accept work.
    """

    LOADING = "loading"
    """Pipeline is loading model weights / warming up."""

    READY = "ready"
    """Pipeline is ready to accept requests."""

    ERROR = "error"
    """Pipeline encountered a fatal error during setup or inference."""

    IDLE = "idle"
    """Pipeline is loaded but not currently processing."""


class Pipeline(abc.ABC):
    """Base class for request-response pipelines.

    Subclasses must implement :meth:`setup` and :meth:`predict`.

    Class-level attributes control resource requirements and metadata::

        class TextToImage(Pipeline):
            pipeline_id = "text-to-image"
            version = "1.0.0"
            description = "Generate images from text prompts"
            gpu = "A100"
            min_vram_gb = 24

            def setup(self):
                ...

            def predict(self, prompt: str, steps: int = 30) -> ...:
                ...
    """

    # -- Pipeline identity (overridden by subclasses) --
    pipeline_id: Optional[str] = None
    version: Optional[str] = None
    description: Optional[str] = None

    # -- Resource hints (overridden by subclasses) --
    gpu: Optional[str] = None
    min_vram_gb: Optional[int] = None
    cpu_only: bool = False

    # -- Framework-managed state --
    _state: PipelineState = PipelineState.IDLE

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Auto-derive pipeline_id from class name if not set
        if cls.pipeline_id is None:
            cls.pipeline_id = cls.__name__.lower()
        cls._state = PipelineState.IDLE

    @property
    def state(self) -> PipelineState:
        """Current health state of the pipeline."""
        return self._state

    @abc.abstractmethod
    def setup(self) -> None:
        """Load model weights and prepare for inference.

        Called once when the pipeline container starts.  Heavy
        initialisation (downloading checkpoints, compiling models)
        should happen here, not in ``__init__``.
        """

    @abc.abstractmethod
    def predict(self, **kwargs: Any) -> Any:
        """Run a single inference request and return the result.

        Parameters are typed via :func:`~livepeer_gateway.runner.inputs.Input`
        descriptors and validated by the serving layer before this method is
        called.
        """

    @classmethod
    def prepare_models(cls) -> None:
        """Download and prepare model artifacts ahead of time.

        Override this to download model checkpoints, compile optimised
        kernels, or perform other expensive one-time preparation that
        should happen during Docker image build rather than at container
        startup.

        This is called separately from :meth:`setup` — typically via
        ``livepeer prepare <module>`` or the ``PREPARE_MODELS=1``
        environment variable during image build.

        The default implementation is a no-op.
        """
        _LOG.info("%s.prepare_models(): no-op (override to download models)", cls.__name__)


class StreamPipeline(abc.ABC):
    """Base class for real-time streaming pipelines.

    Subclasses declare which modalities they consume and produce via
    ``inputs`` and ``outputs``, then override the corresponding
    callbacks.

    **Video only** (default)::

        class StyleTransfer(StreamPipeline):
            def on_video_frame(self, frame, **params):
                return self.model.stylize(frame)

    **Audio only**::

        class NoiseRemoval(StreamPipeline):
            inputs = ["audio"]
            outputs = ["audio"]

            def on_audio_frame(self, frame, **params):
                return self.model.denoise(frame)

    **Video + Audio** (e.g. lip sync — audio updates state, video produces output)::

        class LipSync(StreamPipeline):
            inputs = ["video", "audio"]
            outputs = ["video"]

            def on_video_frame(self, frame, **params):
                return self.model.sync(frame, self._phonemes)

            def on_audio_frame(self, frame, **params):
                self._phonemes = self.model.extract(frame)

    The ``on_frame()`` method is kept as a backward-compatible alias
    for ``on_video_frame()`` — override either one.
    """

    # -- Pipeline identity (overridden by subclasses) --
    pipeline_id: Optional[str] = None
    version: Optional[str] = None
    description: Optional[str] = None

    # -- Modality declarations --
    inputs: list[str] = ["video"]
    outputs: list[str] = ["video"]

    # -- Resource hints --
    gpu: Optional[str] = None
    min_vram_gb: Optional[int] = None
    cpu_only: bool = False

    # -- Framework-managed state --
    _state: PipelineState = PipelineState.IDLE

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.pipeline_id is None:
            cls.pipeline_id = cls.__name__.lower()
        cls._state = PipelineState.IDLE

    @property
    def state(self) -> PipelineState:
        """Current health state of the pipeline."""
        return self._state

    @abc.abstractmethod
    def setup(self) -> None:
        """Load model weights and prepare for streaming inference."""

    def on_frame(self, frame: Any, **params: Any) -> Any:
        """Process a single input frame (backward-compatible alias).

        By default delegates to :meth:`on_video_frame`.  Override this
        OR ``on_video_frame`` — not both.
        """
        return self.on_video_frame(frame, **params)

    def on_video_frame(self, frame: Any, **params: Any) -> Any:
        """Process a single video frame and return the output frame.

        Override this for video processing pipelines.  *frame* is
        typically an ``av.VideoFrame`` or a
        :class:`~livepeer_gateway.VideoDecodedMediaFrame`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement on_video_frame(). "
            "Override on_video_frame() or on_frame()."
        )

    def on_audio_frame(self, frame: Any, **params: Any) -> Any:
        """Process a single audio frame.

        Override this for audio processing pipelines.  *frame* is
        typically an ``av.AudioFrame`` or a
        :class:`~livepeer_gateway.AudioDecodedMediaFrame`.

        Return a frame to publish audio output, or ``None`` to only
        update internal state (e.g. for audio-reactive video pipelines).
        """
        return None

    def on_params_update(self, params: dict[str, Any]) -> None:
        """Handle a live parameter update.

        Override this to react to control-channel messages that change
        inference parameters mid-stream (e.g. adjusting style strength
        or switching prompts).

        The default implementation is a no-op.
        """

    @classmethod
    def prepare_models(cls) -> None:
        """Download and prepare model artifacts ahead of time.

        See :meth:`Pipeline.prepare_models` for details.
        """
        _LOG.info("%s.prepare_models(): no-op (override to download models)", cls.__name__)
