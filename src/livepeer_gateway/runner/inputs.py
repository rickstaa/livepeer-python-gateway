"""Type-safe input and output descriptors for Pipeline definitions.

Developers annotate ``predict()`` / ``on_frame()`` parameters with
:func:`Input` and return types with :func:`Output` so the SDK can
introspect signatures and generate JSON schemas automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class InputDescriptor:
    """Metadata attached to a single pipeline input parameter."""

    description: Optional[str] = None
    default: Any = None
    ge: Optional[float] = None
    le: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    choices: Optional[list[Any]] = None
    media_type: Optional[str] = None

    @property
    def has_default(self) -> bool:
        return self.default is not None

    def json_schema_constraints(self) -> dict[str, Any]:
        """Return JSON-Schema constraint keywords derived from this descriptor."""
        constraints: dict[str, Any] = {}
        if self.ge is not None:
            constraints["minimum"] = self.ge
        if self.le is not None:
            constraints["maximum"] = self.le
        if self.min_length is not None:
            constraints["minLength"] = self.min_length
        if self.max_length is not None:
            constraints["maxLength"] = self.max_length
        if self.choices is not None:
            constraints["enum"] = list(self.choices)
        return constraints


def Input(
    *,
    description: Optional[str] = None,
    default: Any = None,
    ge: Optional[float] = None,
    le: Optional[float] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    choices: Optional[list[Any]] = None,
    media_type: Optional[str] = None,
) -> Any:
    """Declare an input parameter with optional constraints and metadata.

    Used as a default-value annotation on ``predict()`` / ``on_frame()``
    parameters::

        def predict(
            self,
            prompt: str = Input(description="The text prompt"),
            steps: int = Input(default=30, ge=1, le=100),
        ) -> ...:
    """
    return InputDescriptor(
        description=description,
        default=default,
        ge=ge,
        le=le,
        min_length=min_length,
        max_length=max_length,
        choices=choices,
        media_type=media_type,
    )


@dataclass(frozen=True)
class OutputDescriptor:
    """Metadata describing the output of a pipeline."""

    type: str = "json"
    media_type: Optional[str] = None
    description: Optional[str] = None


def Output(
    *,
    type: str = "json",
    media_type: Optional[str] = None,
    description: Optional[str] = None,
) -> Any:
    """Declare the output type of a pipeline's ``predict()`` method.

    Supported *type* values: ``"json"``, ``"image"``, ``"audio"``,
    ``"video"``, ``"text"``, ``"text_stream"``.

    Example::

        def predict(self, prompt: str = Input(...)) -> Output(type="image"):
            ...
    """
    return OutputDescriptor(
        type=type,
        media_type=media_type,
        description=description,
    )
