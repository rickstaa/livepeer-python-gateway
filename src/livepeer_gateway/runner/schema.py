"""Introspect Pipeline.predict() signatures to generate JSON schemas.

The generated schema is used for:

- Automatic API documentation
- Studio playground UI generation (form controls, validators)
- AI-agent code generation from natural language
- Request validation in the serve layer
"""

from __future__ import annotations

import inspect
from typing import Any, Optional, Union, get_args, get_origin

from .inputs import InputDescriptor, OutputDescriptor
from .pipeline import Pipeline, StreamPipeline


# Python type -> JSON Schema type mapping
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    bytes: "string",  # base64-encoded
}


def _json_type(annotation: Any) -> dict[str, Any]:
    """Convert a Python type annotation to a JSON Schema type fragment."""
    if annotation is inspect.Parameter.empty or annotation is None:
        return {"type": "string"}

    origin = get_origin(annotation)

    # Optional[X] -> nullable X
    if origin is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            schema = _json_type(args[0])
            schema["nullable"] = True
            return schema

    # list[X]
    if origin is list:
        inner = get_args(annotation)
        items = _json_type(inner[0]) if inner else {}
        return {"type": "array", "items": items}

    # dict[str, X]
    if origin is dict:
        args = get_args(annotation)
        additional = _json_type(args[1]) if len(args) > 1 else {}
        return {"type": "object", "additionalProperties": additional}

    if annotation in _TYPE_MAP:
        schema: dict[str, Any] = {"type": _TYPE_MAP[annotation]}
        if annotation is bytes:
            schema["format"] = "base64"
        return schema

    return {"type": "string"}


def _param_schema(
    name: str,
    param: inspect.Parameter,
) -> dict[str, Any]:
    """Build a JSON Schema property for a single parameter."""
    descriptor: Optional[InputDescriptor] = None
    default = param.default

    if isinstance(default, InputDescriptor):
        descriptor = default
        default = descriptor.default

    schema = _json_type(param.annotation)

    if descriptor is not None:
        if descriptor.description:
            schema["description"] = descriptor.description
        schema.update(descriptor.json_schema_constraints())
        if descriptor.media_type:
            schema["x-media-type"] = descriptor.media_type

    if default is not None and default is not inspect.Parameter.empty:
        schema["default"] = default

    return schema


def extract_schema(pipeline_cls: type) -> dict[str, Any]:
    """Generate a full JSON Schema document from a Pipeline or StreamPipeline class.

    Introspects the ``predict()`` (for :class:`Pipeline`) or ``on_frame()``
    (for :class:`StreamPipeline`) method signature.

    Returns a dict suitable for JSON serialisation::

        {
            "title": "TextToImage",
            "type": "object",
            "properties": { ... },
            "required": [ ... ],
            "x-gpu": "A100",
            "x-output": { "type": "image" },
        }
    """
    if issubclass(pipeline_cls, Pipeline):
        method_name = "predict"
    elif issubclass(pipeline_cls, StreamPipeline):
        method_name = "on_frame"
    else:
        raise TypeError(
            f"Expected a Pipeline or StreamPipeline subclass, got {pipeline_cls!r}"
        )

    method = getattr(pipeline_cls, method_name, None)
    if method is None:
        raise TypeError(f"{pipeline_cls.__name__} does not define {method_name}()")

    sig = inspect.signature(method)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name == "self":
            continue
        # StreamPipeline.on_frame has a positional `frame` arg handled by transport
        if method_name == "on_frame" and name == "frame":
            continue
        # Skip **kwargs
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            continue

        prop = _param_schema(name, param)
        properties[name] = prop

        # Determine whether the parameter is required
        has_default = (
            param.default is not inspect.Parameter.empty
            and not isinstance(param.default, InputDescriptor)
        ) or (
            isinstance(param.default, InputDescriptor) and param.default.has_default
        )
        if not has_default:
            required.append(name)

    schema: dict[str, Any] = {
        "title": pipeline_cls.__name__,
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required

    # Pipeline identity metadata
    pipeline_id = getattr(pipeline_cls, "pipeline_id", None)
    if pipeline_id:
        schema["x-pipeline-id"] = pipeline_id
    version = getattr(pipeline_cls, "version", None)
    if version:
        schema["x-version"] = version
    description = getattr(pipeline_cls, "description", None)
    if description:
        schema["description"] = description

    # Modality declarations (StreamPipeline only)
    inputs = getattr(pipeline_cls, "inputs", None)
    if inputs and inputs != ["video"]:  # only include if non-default
        schema["x-inputs"] = inputs
    outputs = getattr(pipeline_cls, "outputs", None)
    if outputs and outputs != ["video"]:
        schema["x-outputs"] = outputs

    # Resource hints
    gpu = getattr(pipeline_cls, "gpu", None)
    if gpu:
        schema["x-gpu"] = gpu
    min_vram = getattr(pipeline_cls, "min_vram_gb", None)
    if min_vram:
        schema["x-min-vram-gb"] = min_vram

    # Output descriptor from return annotation
    return_annotation = sig.return_annotation
    if isinstance(return_annotation, OutputDescriptor):
        output_info: dict[str, Any] = {"type": return_annotation.type}
        if return_annotation.media_type:
            output_info["media_type"] = return_annotation.media_type
        if return_annotation.description:
            output_info["description"] = return_annotation.description
        if return_annotation.type == "text_stream":
            output_info["streaming"] = True
            output_info["protocol"] = "sse"
        schema["x-output"] = output_info

    return schema
