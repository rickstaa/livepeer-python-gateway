"""HTTP server that bridges Pipeline/StreamPipeline classes to Livepeer transport.

For **request-response** pipelines the server exposes:

- ``POST /predict``       -- run inference, return result (or SSE stream
                             when ``predict()`` yields tokens)
- ``GET  /schema``        -- return the JSON schema
- ``GET  /health``        -- liveness probe

When ``predict()`` is a generator (sync or async), or the pipeline
declares ``Output(type="text_stream")``, the ``/predict`` endpoint
automatically responds with ``text/event-stream`` (Server-Sent Events).
This follows the same SSE convention used by Replicate, OpenAI, and
other LLM serving platforms for token-by-token streaming.

For **stream** pipelines the server additionally manages trickle
subscribe/publish channels so ``on_frame()`` is called for every
inbound frame and output frames are published back.

The serve layer is intentionally thin: it validates inputs against
the schema, calls into the user's Pipeline, and serialises the
response.  Heavy lifting (trickle I/O, segment management) is
delegated to the existing ``livepeer_gateway`` transport primitives.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import io
import json
import logging
import types
from typing import Any, AsyncIterator, Iterator, Optional

from aiohttp import web

from .inputs import InputDescriptor, OutputDescriptor
from .pipeline import Pipeline, StreamPipeline
from .schema import extract_schema

_LOG = logging.getLogger(__name__)


def _coerce_value(value: Any, annotation: Any) -> Any:
    """Best-effort coercion of a JSON value to the annotated Python type."""
    if annotation is inspect.Parameter.empty or annotation is None:
        return value
    if annotation is int and isinstance(value, (str, float)):
        return int(value)
    if annotation is float and isinstance(value, (str, int)):
        return float(value)
    if annotation is bool and isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    if annotation is str and not isinstance(value, str):
        return str(value)
    return value


def _validate_inputs(
    params: dict[str, Any],
    sig: inspect.Signature,
) -> dict[str, Any]:
    """Validate and coerce *params* against the method signature.

    Returns a cleaned dict ready to be passed as ``**kwargs``.
    Raises :class:`ValueError` for constraint violations.
    """
    cleaned: dict[str, Any] = {}

    for name, param in sig.parameters.items():
        if name == "self" or name == "frame":
            continue
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            continue

        descriptor: Optional[InputDescriptor] = None
        if isinstance(param.default, InputDescriptor):
            descriptor = param.default

        if name in params:
            value = _coerce_value(params[name], param.annotation)
        elif descriptor is not None and descriptor.has_default:
            value = descriptor.default
        elif param.default is not inspect.Parameter.empty and not isinstance(
            param.default, InputDescriptor
        ):
            value = param.default
        else:
            raise ValueError(f"Missing required parameter: {name}")

        # Validate constraints from InputDescriptor
        if descriptor is not None:
            if descriptor.ge is not None and isinstance(value, (int, float)):
                if value < descriptor.ge:
                    raise ValueError(
                        f"Parameter '{name}' must be >= {descriptor.ge}, got {value}"
                    )
            if descriptor.le is not None and isinstance(value, (int, float)):
                if value > descriptor.le:
                    raise ValueError(
                        f"Parameter '{name}' must be <= {descriptor.le}, got {value}"
                    )
            if descriptor.min_length is not None and isinstance(value, str):
                if len(value) < descriptor.min_length:
                    raise ValueError(
                        f"Parameter '{name}' length must be >= {descriptor.min_length}"
                    )
            if descriptor.max_length is not None and isinstance(value, str):
                if len(value) > descriptor.max_length:
                    raise ValueError(
                        f"Parameter '{name}' length must be <= {descriptor.max_length}"
                    )
            if descriptor.choices is not None and value not in descriptor.choices:
                raise ValueError(
                    f"Parameter '{name}' must be one of {descriptor.choices}, got {value!r}"
                )

        cleaned[name] = value

    return cleaned


def _get_output_descriptor(pipeline_cls: type) -> Optional[OutputDescriptor]:
    """Extract the OutputDescriptor from the predict method's return annotation."""
    method_name = "predict" if issubclass(pipeline_cls, Pipeline) else "on_frame"
    method = getattr(pipeline_cls, method_name, None)
    if method is None:
        return None
    sig = inspect.signature(method)
    ann = sig.return_annotation
    return ann if isinstance(ann, OutputDescriptor) else None


def _serialise_result(result: Any, output: Optional[OutputDescriptor]) -> web.Response:
    """Convert a predict() return value to an HTTP response."""
    if result is None:
        return web.json_response({"status": "ok"})

    out_type = output.type if output else "json"

    if out_type == "image":
        # Result should be bytes or a file-like object
        if isinstance(result, bytes):
            media_type = (output.media_type if output and output.media_type else "image/png")
            return web.Response(body=result, content_type=media_type)
        if isinstance(result, io.IOBase):
            data = result.read()
            media_type = (output.media_type if output and output.media_type else "image/png")
            return web.Response(body=data, content_type=media_type)
        # Fall back: base64-encode if it's somehow a string
        return web.json_response({"image": base64.b64encode(result).decode() if isinstance(result, (bytes, bytearray)) else str(result)})

    if out_type == "audio":
        if isinstance(result, bytes):
            media_type = (output.media_type if output and output.media_type else "audio/wav")
            return web.Response(body=result, content_type=media_type)
        return web.json_response({"audio": str(result)})

    if out_type == "video":
        if isinstance(result, bytes):
            media_type = (output.media_type if output and output.media_type else "video/mp4")
            return web.Response(body=result, content_type=media_type)
        return web.json_response({"video": str(result)})

    if out_type == "text":
        text = result if isinstance(result, str) else str(result)
        return web.Response(text=text, content_type="text/plain")

    # Default: JSON
    if isinstance(result, dict):
        return web.json_response(result)
    if isinstance(result, (list, tuple)):
        return web.json_response(result)
    if isinstance(result, str):
        return web.json_response({"output": result})
    # Last resort
    return web.json_response({"output": str(result)})


def _is_generator(obj: Any) -> bool:
    """Check if *obj* is a sync or async generator/iterator."""
    return isinstance(obj, (types.GeneratorType, types.AsyncGeneratorType))


def _wants_sse(request: web.Request, output: Optional[OutputDescriptor]) -> bool:
    """Determine whether the response should be SSE.

    Returns True if the client sends ``Accept: text/event-stream`` or
    the pipeline declares ``Output(type="text_stream")``.
    """
    if output and output.type == "text_stream":
        return True
    accept = request.headers.get("Accept", "")
    return "text/event-stream" in accept


async def _stream_sse(
    result: Any,
    response: web.StreamResponse,
) -> None:
    """Consume a sync or async generator and write SSE events.

    Each yielded value becomes one SSE ``data:`` frame.  A final
    ``data: [DONE]`` event signals the end of the stream, matching
    the convention used by OpenAI and Replicate.
    """
    try:
        if isinstance(result, types.AsyncGeneratorType):
            async for token in result:
                event = _sse_encode(token)
                await response.write(event)
        elif isinstance(result, types.GeneratorType):
            for token in result:
                event = _sse_encode(token)
                await response.write(event)
        else:
            # Single value — send as one event
            event = _sse_encode(result)
            await response.write(event)
    except (ConnectionResetError, ConnectionError, ConnectionAbortedError) as e:
        # Client disconnected mid-stream (normal for SSE clients)
        _LOG.debug("SSE client disconnected: %s", e)
        return
    except Exception:
        _LOG.exception("Error during SSE streaming")
        try:
            error_event = f"event: error\ndata: {json.dumps({'error': 'Stream failed'})}\n\n"
            await response.write(error_event.encode("utf-8"))
        except Exception:
            pass
        return
    # Send the [DONE] sentinel
    try:
        await response.write(b"data: [DONE]\n\n")
    except Exception:
        pass


def _sse_encode(token: Any) -> bytes:
    """Encode a single token/chunk as an SSE ``data:`` frame."""
    if isinstance(token, str):
        payload = json.dumps({"output": token, "type": "token"})
    elif isinstance(token, dict):
        payload = json.dumps(token)
    else:
        payload = json.dumps({"output": str(token), "type": "token"})
    return f"data: {payload}\n\n".encode("utf-8")


class PipelineServer:
    """HTTP server wrapping a :class:`Pipeline` instance.

    Exposes ``/predict``, ``/schema``, and ``/health`` endpoints.

    When ``predict()`` returns a generator (sync or async), or the
    pipeline declares ``Output(type="text_stream")``, the ``/predict``
    endpoint responds with Server-Sent Events for token-by-token
    streaming — the standard pattern for LLM serving.
    """

    def __init__(self, pipeline: Pipeline, *, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.pipeline = pipeline
        self.host = host
        self.port = port
        self._schema = extract_schema(type(pipeline))
        self._output = _get_output_descriptor(type(pipeline))
        self._predict_sig = inspect.signature(pipeline.predict)

    async def _handle_predict(self, request: web.Request) -> web.Response:
        """Handle POST /predict requests.

        Automatically detects whether to return a single response or
        stream SSE events based on the predict() return type and
        client Accept header.
        """
        try:
            if request.content_type == "application/json":
                params = await request.json()
            else:
                params = dict(await request.post())
        except Exception:
            return web.json_response(
                {"error": "Invalid request body"}, status=400
            )

        try:
            cleaned = _validate_inputs(params, self._predict_sig)
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=422)

        try:
            result = self.pipeline.predict(**cleaned)
            # Await coroutines, but not generators (they stream as SSE)
            if inspect.iscoroutine(result):
                result = await result
        except Exception:
            _LOG.exception("Pipeline predict() failed")
            return web.json_response(
                {"error": "Inference failed"}, status=500
            )

        # SSE streaming path: generator result or text_stream output type
        if _is_generator(result) or (
            not _is_generator(result) and _wants_sse(request, self._output)
        ):
            response = web.StreamResponse(
                status=200,
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
            await response.prepare(request)
            await _stream_sse(result, response)
            await response.write_eof()
            return response

        return _serialise_result(result, self._output)

    async def _handle_schema(self, request: web.Request) -> web.Response:
        """Handle GET /schema requests."""
        return web.json_response(self._schema)

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Handle GET /health requests."""
        return web.json_response({"status": "healthy"})

    def create_app(self) -> web.Application:
        """Build and return the aiohttp Application (without starting it)."""
        app = web.Application()
        app.router.add_post("/predict", self._handle_predict)
        app.router.add_get("/schema", self._handle_schema)
        app.router.add_get("/health", self._handle_health)
        return app

    def run(self) -> None:
        """Set up the pipeline and start the HTTP server (blocking)."""
        _LOG.info("Running pipeline setup...")
        self.pipeline.setup()
        _LOG.info("Pipeline setup complete.")

        app = self.create_app()
        _LOG.info("Starting server on %s:%s", self.host, self.port)
        web.run_app(app, host=self.host, port=self.port, print=_LOG.info)


class StreamPipelineServer:
    """HTTP server wrapping a :class:`StreamPipeline` instance.

    Exposes the same endpoints as :class:`PipelineServer` plus trickle-based
    streaming I/O.  The ``/stream`` endpoint accepts a trickle subscribe URL
    and publish URL, then continuously reads input frames, runs
    ``on_frame()``, and publishes output frames.

    Additionally exposes ``POST /params`` for live parameter updates.
    """

    def __init__(
        self,
        pipeline: StreamPipeline,
        *,
        host: str = "0.0.0.0",
        port: int = 8000,
    ) -> None:
        self.pipeline = pipeline
        self.host = host
        self.port = port
        self._schema = extract_schema(type(pipeline))
        self._output = _get_output_descriptor(type(pipeline))
        self._current_params: dict[str, Any] = {}
        self._stream_task: Optional[asyncio.Task] = None

    async def _handle_schema(self, request: web.Request) -> web.Response:
        return web.json_response(self._schema)

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "healthy"})

    async def _handle_params(self, request: web.Request) -> web.Response:
        """Handle POST /params for live parameter updates."""
        try:
            params = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        self._current_params.update(params)
        try:
            self.pipeline.on_params_update(self._current_params)
        except Exception:
            _LOG.exception("on_params_update() failed")
            return web.json_response({"error": "Params update failed"}, status=500)

        return web.json_response({"status": "ok"})

    async def _handle_stream(self, request: web.Request) -> web.Response:
        """Handle POST /stream to start trickle-based streaming.

        Expects JSON body with ``subscribe_url`` and ``publish_url``.
        Launches a background task that reads frames from the subscribe
        channel, processes them through ``on_frame()``, and publishes
        results to the publish channel.
        """
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        subscribe_url = data.get("subscribe_url")
        publish_url = data.get("publish_url")
        if not subscribe_url or not publish_url:
            return web.json_response(
                {"error": "subscribe_url and publish_url are required"}, status=400
            )

        # Cancel any existing stream task
        if self._stream_task is not None and not self._stream_task.done():
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass

        self._stream_task = asyncio.create_task(
            self._stream_loop(subscribe_url, publish_url)
        )
        return web.json_response({"status": "streaming"})

    async def _stream_loop(
        self,
        subscribe_url: str,
        publish_url: str,
    ) -> None:
        """Core streaming loop using trickle transport primitives."""
        from ..trickle_subscriber import TrickleSubscriber
        from ..trickle_publisher import TricklePublisher

        _LOG.info("Starting stream loop: subscribe=%s publish=%s", subscribe_url, publish_url)

        async with TrickleSubscriber(subscribe_url) as subscriber:
            publisher = TricklePublisher(publish_url, "application/octet-stream")
            try:
                while True:
                    segment = await subscriber.next()
                    if segment is None:
                        _LOG.info("Stream ended (subscriber returned None)")
                        break

                    try:
                        input_data = await segment.read()
                        result = self.pipeline.on_frame(
                            input_data, **self._current_params
                        )
                        if inspect.iscoroutine(result):
                            result = await result

                        if result is not None:
                            if isinstance(result, bytes):
                                output_bytes = result
                            else:
                                output_bytes = json.dumps(result).encode("utf-8")

                            async with await publisher.next() as out_seg:
                                await out_seg.write(output_bytes)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        _LOG.exception("Error processing frame in stream loop")
                    finally:
                        await segment.close()
            finally:
                await publisher.close()

        _LOG.info("Stream loop ended")

    def create_app(self) -> web.Application:
        """Build and return the aiohttp Application."""
        app = web.Application()
        app.router.add_post("/stream", self._handle_stream)
        app.router.add_post("/params", self._handle_params)
        app.router.add_get("/schema", self._handle_schema)
        app.router.add_get("/health", self._handle_health)
        return app

    def run(self) -> None:
        """Set up the pipeline and start the HTTP server (blocking)."""
        _LOG.info("Running stream pipeline setup...")
        self.pipeline.setup()
        _LOG.info("Stream pipeline setup complete.")

        app = self.create_app()
        _LOG.info("Starting stream server on %s:%s", self.host, self.port)
        web.run_app(app, host=self.host, port=self.port, print=_LOG.info)
