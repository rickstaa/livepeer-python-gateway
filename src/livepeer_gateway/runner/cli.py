"""CLI entry points for the Livepeer Pipeline SDK.

Provides two main commands:

- ``livepeer predict <module_path> -i key=value ...``
    Run a local prediction without starting a server.

- ``livepeer push <module_path>``
    Package and deploy a pipeline to the Livepeer network.

- ``livepeer serve <module_path>``
    Start a local HTTP server for the pipeline.

- ``livepeer schema <module_path>``
    Print the JSON schema for a pipeline.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

from .pipeline import Pipeline, StreamPipeline

_LOG = logging.getLogger(__name__)


def _load_pipeline_class(module_path: str) -> type:
    """Load a Pipeline or StreamPipeline subclass from a Python file.

    Scans the module's top-level namespace for the first concrete
    subclass of Pipeline or StreamPipeline.
    """
    path = Path(module_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Module not found: {module_path}")
    if not path.suffix == ".py":
        raise ValueError(f"Expected a .py file, got: {module_path}")

    spec = importlib.util.spec_from_file_location("_user_pipeline", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    # Add the module's directory to sys.path so relative imports work
    parent_dir = str(path.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    spec.loader.exec_module(module)

    # Find the first concrete Pipeline or StreamPipeline subclass
    candidates: list[type] = []
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if not isinstance(attr, type):
            continue
        if attr in (Pipeline, StreamPipeline):
            continue
        if issubclass(attr, (Pipeline, StreamPipeline)):
            candidates.append(attr)

    if not candidates:
        raise TypeError(
            f"No Pipeline or StreamPipeline subclass found in {module_path}"
        )

    if len(candidates) > 1:
        _LOG.warning(
            "Multiple pipeline classes found in %s: %s. Using %s.",
            module_path,
            [c.__name__ for c in candidates],
            candidates[0].__name__,
        )

    return candidates[0]


def _parse_inputs(raw_inputs: list[str]) -> dict[str, Any]:
    """Parse ``key=value`` pairs from the CLI into a dict.

    Attempts JSON parsing for each value to support numbers,
    booleans, lists, and objects.  Falls back to plain string.
    """
    params: dict[str, Any] = {}
    for item in raw_inputs:
        if "=" not in item:
            raise ValueError(f"Invalid input format (expected key=value): {item!r}")
        key, _, value = item.partition("=")
        key = key.strip()
        value = value.strip()

        # Try JSON parsing for rich types
        try:
            params[key] = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            params[key] = value

    return params


def cmd_predict(args: argparse.Namespace) -> None:
    """Run a local prediction."""
    pipeline_cls = _load_pipeline_class(args.module)

    _LOG.info("Instantiating %s...", pipeline_cls.__name__)
    pipeline = pipeline_cls()
    pipeline.setup()

    params = _parse_inputs(args.input or [])

    _LOG.info("Running predict with params: %s", params)
    result = pipeline.predict(**params)
    if isinstance(result, bytes):
        # Write binary output to file
        output_path = args.output or "output.bin"
        Path(output_path).write_bytes(result)
        print(f"Output written to {output_path} ({len(result)} bytes)")
    elif isinstance(result, dict):
        print(json.dumps(result, indent=2))
    else:
        print(result)


def cmd_serve(args: argparse.Namespace) -> None:
    """Start a local HTTP server for the pipeline."""
    from .serve import PipelineServer, StreamPipelineServer

    pipeline_cls = _load_pipeline_class(args.module)
    pipeline = pipeline_cls()

    if issubclass(pipeline_cls, StreamPipeline):
        server = StreamPipelineServer(pipeline, host=args.host, port=args.port)
    else:
        server = PipelineServer(pipeline, host=args.host, port=args.port)

    server.run()


def cmd_schema(args: argparse.Namespace) -> None:
    """Print the JSON schema for a pipeline."""
    from .schema import extract_schema

    pipeline_cls = _load_pipeline_class(args.module)
    schema = extract_schema(pipeline_cls)
    print(json.dumps(schema, indent=2))


def cmd_push(args: argparse.Namespace) -> None:
    """Package and deploy a pipeline to the Livepeer network.

    Steps:
    1. Extract pipeline signature to generate schema JSON
    2. Analyse dependencies
    3. Generate and build Docker image
    4. Push to registry
    5. Register capability on network
    """
    from .schema import extract_schema

    pipeline_cls = _load_pipeline_class(args.module)
    schema = extract_schema(pipeline_cls)
    module_path = Path(args.module).resolve()

    print(f"Deploying {pipeline_cls.__name__}...")
    print(f"Schema: {json.dumps(schema, indent=2)}")

    # Step 1: Collect dependencies
    requirements = _collect_dependencies(module_path)

    # Step 2: Generate Dockerfile
    dockerfile = _generate_dockerfile(module_path, requirements)

    # Step 3: Build and push
    image_tag = args.tag or f"livepeer/{pipeline_cls.__name__.lower()}:latest"

    with tempfile.TemporaryDirectory() as build_dir:
        build_path = Path(build_dir)

        # Write Dockerfile
        (build_path / "Dockerfile").write_text(dockerfile)

        # Write schema
        (build_path / "schema.json").write_text(json.dumps(schema, indent=2))

        # Copy pipeline module
        (build_path / module_path.name).write_bytes(module_path.read_bytes())

        # Copy requirements if they exist
        if requirements:
            (build_path / "requirements.txt").write_text("\n".join(requirements))

        if args.dry_run:
            print("\n--- Dry run: generated Dockerfile ---")
            print(dockerfile)
            print(f"\nImage tag: {image_tag}")
            print("Schema written to schema.json")
            return

        # Build Docker image
        print(f"Building Docker image: {image_tag}")
        _run_docker_build(build_path, image_tag)

        if not args.local:
            print(f"Pushing image: {image_tag}")
            _run_docker_push(image_tag)
            print(f"Image pushed: {image_tag}")

    # Step 4: Register on network (placeholder for future implementation)
    if not args.local:
        print(f"\nPipeline {pipeline_cls.__name__} deployed as {image_tag}")
        print("Network registration will be available in a future release.")
    else:
        print(f"\nPipeline {pipeline_cls.__name__} built locally as {image_tag}")


def _collect_dependencies(module_path: Path) -> list[str]:
    """Collect Python dependencies for the pipeline.

    Looks for a requirements.txt in the same directory as the pipeline
    module.  Falls back to an empty list.
    """
    req_file = module_path.parent / "requirements.txt"
    if req_file.exists():
        return [
            line.strip()
            for line in req_file.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    return []


def _generate_dockerfile(module_path: Path, requirements: list[str]) -> str:
    """Generate a Dockerfile for the pipeline."""
    module_name = module_path.name
    req_install = ""
    if requirements:
        req_install = "COPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\n"

    return f"""\
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir livepeer-gateway

{req_install}COPY {module_name} .
COPY schema.json .

EXPOSE 8000

CMD ["livepeer", "serve", "{module_name}", "--host", "0.0.0.0", "--port", "8000"]
"""


def _run_docker_build(build_dir: Path, tag: str) -> None:
    """Build a Docker image from the given directory."""
    result = subprocess.run(
        ["docker", "build", "-t", tag, str(build_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Docker build failed:\n{result.stderr}")
    print(result.stdout)


def _run_docker_push(tag: str) -> None:
    """Push a Docker image to the registry."""
    result = subprocess.run(
        ["docker", "push", tag],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Docker push failed:\n{result.stderr}")
    print(result.stdout)


def main() -> None:
    """Main entry point for the ``livepeer`` CLI."""
    parser = argparse.ArgumentParser(
        prog="livepeer",
        description="Livepeer Pipeline SDK CLI",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- predict --
    predict_parser = subparsers.add_parser(
        "predict",
        help="Run a local prediction",
    )
    predict_parser.add_argument("module", help="Path to pipeline .py file")
    predict_parser.add_argument(
        "-i", "--input",
        action="append",
        metavar="KEY=VALUE",
        help="Input parameter (can be repeated)",
    )
    predict_parser.add_argument(
        "-o", "--output",
        help="Output file path for binary results",
    )

    # -- serve --
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start a local HTTP server for the pipeline",
    )
    serve_parser.add_argument("module", help="Path to pipeline .py file")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port number")

    # -- schema --
    schema_parser = subparsers.add_parser(
        "schema",
        help="Print the JSON schema for a pipeline",
    )
    schema_parser.add_argument("module", help="Path to pipeline .py file")

    # -- push --
    push_parser = subparsers.add_parser(
        "push",
        help="Package and deploy a pipeline to the Livepeer network",
    )
    push_parser.add_argument("module", help="Path to pipeline .py file")
    push_parser.add_argument("--tag", help="Docker image tag")
    push_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate Dockerfile without building",
    )
    push_parser.add_argument(
        "--local",
        action="store_true",
        help="Build locally without pushing to registry",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    commands = {
        "predict": cmd_predict,
        "serve": cmd_serve,
        "schema": cmd_schema,
        "push": cmd_push,
    }
    commands[args.command](args)
