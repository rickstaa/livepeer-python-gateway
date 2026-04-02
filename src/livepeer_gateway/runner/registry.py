"""Pipeline registry for discovering and managing available pipelines.

The registry allows multiple pipelines to be registered in a single
container and discovered by the orchestrator or Studio.

Pipelines can be registered explicitly::

    from livepeer_gateway.runner import PipelineRegistry

    PipelineRegistry.register(MyPipeline)
    PipelineRegistry.register(AnotherPipeline)

Or discovered automatically when loaded via the CLI.
"""

from __future__ import annotations

import logging
from typing import Optional

from .pipeline import Pipeline, StreamPipeline

_LOG = logging.getLogger(__name__)


class PipelineRegistry:
    """In-process registry of available pipeline classes.

    All methods are classmethods operating on shared state, so there
    is a single global registry per process.
    """

    _pipelines: dict[str, type] = {}

    @classmethod
    def register(cls, pipeline_cls: type, pipeline_id: Optional[str] = None) -> None:
        """Register a pipeline class.

        Args:
            pipeline_cls: A :class:`Pipeline` or :class:`StreamPipeline` subclass.
            pipeline_id: Override the pipeline ID. Defaults to
                ``pipeline_cls.pipeline_id``.
        """
        if not (
            isinstance(pipeline_cls, type)
            and issubclass(pipeline_cls, (Pipeline, StreamPipeline))
        ):
            raise TypeError(
                f"Expected a Pipeline or StreamPipeline subclass, got {pipeline_cls!r}"
            )

        pid = pipeline_id or getattr(pipeline_cls, "pipeline_id", None)
        if pid is None:
            pid = pipeline_cls.__name__.lower()

        if pid in cls._pipelines:
            existing = cls._pipelines[pid]
            if existing is not pipeline_cls:
                _LOG.warning(
                    "Overwriting pipeline registry entry %r: %s -> %s",
                    pid,
                    existing.__name__,
                    pipeline_cls.__name__,
                )

        cls._pipelines[pid] = pipeline_cls
        _LOG.debug("Registered pipeline %r -> %s", pid, pipeline_cls.__name__)

    @classmethod
    def get(cls, pipeline_id: str) -> Optional[type]:
        """Look up a pipeline class by ID.

        Returns ``None`` if the ID is not registered.
        """
        return cls._pipelines.get(pipeline_id)

    @classmethod
    def list(cls) -> list[str]:
        """Return all registered pipeline IDs."""
        return list(cls._pipelines.keys())

    @classmethod
    def list_with_info(cls) -> list[dict]:
        """Return registered pipelines with their metadata.

        Each entry contains ``pipeline_id``, ``class_name``, ``type``
        (``"pipeline"`` or ``"stream"``), ``version``, ``description``,
        ``gpu``, and ``min_vram_gb``.
        """
        result = []
        for pid, pipeline_cls in cls._pipelines.items():
            is_stream = issubclass(pipeline_cls, StreamPipeline)
            result.append({
                "pipeline_id": pid,
                "class_name": pipeline_cls.__name__,
                "type": "stream" if is_stream else "pipeline",
                "version": getattr(pipeline_cls, "version", None),
                "description": getattr(pipeline_cls, "description", None),
                "gpu": getattr(pipeline_cls, "gpu", None),
                "min_vram_gb": getattr(pipeline_cls, "min_vram_gb", None),
            })
        return result

    @classmethod
    def clear(cls) -> None:
        """Remove all registered pipelines (primarily for testing)."""
        cls._pipelines.clear()
