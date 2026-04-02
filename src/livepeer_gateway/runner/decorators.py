"""Decorator shortcuts for defining simple pipelines.

For stateless or simple pipelines where a full class is unnecessary,
the ``@pipeline`` decorator wraps a plain function into a
:class:`~livepeer_gateway.runner.Pipeline` subclass automatically.

Example::

    from livepeer_gateway.runner.decorators import pipeline

    @pipeline(gpu="A100", pipeline_id="upscale", version="1.0.0")
    def upscale(
        image: bytes = Input(media_type="image/png"),
        scale: int = Input(default=2, choices=[2, 4]),
    ) -> Output(type="image"):
        return do_upscale(image, scale)

This is equivalent to::

    class Upscale(Pipeline):
        gpu = "A100"
        pipeline_id = "upscale"
        version = "1.0.0"

        def setup(self):
            pass

        def predict(self, image, scale=2):
            return do_upscale(image, scale)

The decorator approach is inspired by Modal and Chutes but the
class-based approach remains primary, since AI models are inherently
stateful (load weights once, predict many times).

For stateful pipelines with ``setup()`` or streaming with
``on_frame()``, use the class-based approach instead.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Optional

from .pipeline import Pipeline


def pipeline(
    *,
    gpu: Optional[str] = None,
    min_vram_gb: Optional[int] = None,
    cpu_only: bool = False,
    pipeline_id: Optional[str] = None,
    version: Optional[str] = None,
    description: Optional[str] = None,
    setup: Optional[Callable] = None,
) -> Callable:
    """Decorator that wraps a function into a Pipeline class.

    Args:
        gpu: GPU type hint (e.g. ``"A100"``).
        min_vram_gb: Minimum VRAM requirement.
        cpu_only: Set ``True`` for CPU-only pipelines.
        pipeline_id: Pipeline identifier (defaults to function name).
        version: Semantic version string.
        description: Human-readable description.
        setup: Optional setup callable invoked once before first predict.
    """

    def decorator(fn: Callable) -> type:
        pid = pipeline_id or fn.__name__

        # Build the class dynamically
        class_attrs: dict[str, Any] = {
            "gpu": gpu,
            "min_vram_gb": min_vram_gb,
            "cpu_only": cpu_only,
            "pipeline_id": pid,
            "version": version,
            "description": description or fn.__doc__,
        }

        # Capture the setup callable and predict function
        _setup_fn = setup
        _predict_fn = fn

        def _setup(self: Any) -> None:
            if _setup_fn is not None:
                _setup_fn()

        def _predict(self: Any, **kwargs: Any) -> Any:
            return _predict_fn(**kwargs)

        # Copy the original function's signature onto predict so that
        # schema introspection and _validate_inputs see the typed
        # parameters, not just **kwargs.
        import inspect

        orig_sig = inspect.signature(fn)
        # Prepend 'self' parameter for the method signature
        self_param = inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)
        new_params = [self_param] + list(orig_sig.parameters.values())
        _predict.__signature__ = orig_sig.replace(parameters=new_params)
        _predict.__annotations__ = fn.__annotations__.copy()

        class_attrs["setup"] = _setup
        class_attrs["predict"] = _predict

        # Create the Pipeline subclass
        cls = type(fn.__name__, (Pipeline,), class_attrs)

        # Store reference to original function for testing/inspection
        cls._original_fn = fn

        return cls

    return decorator
