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
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class PipelineMeta:
    """Runtime metadata populated by the framework before ``setup()``."""

    pipeline_id: Optional[str] = None
    version: Optional[str] = None


class Pipeline(abc.ABC):
    """Base class for request-response pipelines.

    Subclasses must implement :meth:`setup` and :meth:`predict`.

    Class-level attributes control resource requirements::

        class TextToImage(Pipeline):
            gpu = "A100"
            min_vram_gb = 24

            def setup(self):
                ...

            def predict(self, prompt: str, steps: int = 30) -> ...:
                ...
    """

    # -- Resource hints (overridden by subclasses) --
    gpu: Optional[str] = None
    min_vram_gb: Optional[int] = None
    cpu_only: bool = False

    # -- Framework-managed metadata --
    _meta: PipelineMeta = field(default_factory=PipelineMeta)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._meta = PipelineMeta()

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


class StreamPipeline(abc.ABC):
    """Base class for real-time streaming pipelines.

    Subclasses must implement :meth:`setup` and :meth:`on_frame`.
    Optionally override :meth:`on_params_update` to handle live
    parameter changes (e.g. via WebRTC data channel or control messages).

    Example::

        class StyleTransfer(StreamPipeline):
            gpu = "T4"

            def setup(self):
                ...

            def on_frame(self, frame, **params):
                ...

            def on_params_update(self, params):
                self.current_style = params.get("style", self.current_style)
    """

    gpu: Optional[str] = None
    min_vram_gb: Optional[int] = None
    cpu_only: bool = False

    _meta: PipelineMeta = field(default_factory=PipelineMeta)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._meta = PipelineMeta()

    @abc.abstractmethod
    def setup(self) -> None:
        """Load model weights and prepare for streaming inference."""

    @abc.abstractmethod
    def on_frame(self, frame: Any, **params: Any) -> Any:
        """Process a single input frame and return the output frame.

        *frame* is typically an ``av.VideoFrame`` but may also be raw
        bytes depending on the transport configuration.  Additional
        keyword arguments come from the most recent parameter update.
        """

    def on_params_update(self, params: dict[str, Any]) -> None:
        """Handle a live parameter update.

        Override this to react to control-channel messages that change
        inference parameters mid-stream (e.g. adjusting style strength
        or switching prompts).

        The default implementation is a no-op.
        """
