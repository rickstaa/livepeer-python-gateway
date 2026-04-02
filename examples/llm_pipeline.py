"""Example: an LLM pipeline with SSE token streaming.

This demonstrates how to define a Pipeline that yields tokens via
a generator, which the SDK automatically streams as Server-Sent Events.
This follows the same pattern used by Replicate, OpenAI, and other
LLM serving platforms.

Usage:

    # Run a local prediction (tokens stream to stdout)
    livepeer predict examples/llm_pipeline.py -i prompt="Tell me about Livepeer"

    # Print the auto-generated schema
    livepeer schema examples/llm_pipeline.py

    # Start a local HTTP server (POST /predict returns SSE stream)
    livepeer serve examples/llm_pipeline.py

    # Test with curl:
    # curl -N -X POST http://localhost:8000/predict \\
    #   -H "Content-Type: application/json" \\
    #   -d '{"prompt": "Tell me about Livepeer"}'
"""

import time
from typing import Iterator

from livepeer_gateway.runner import Input, Output, Pipeline


class TextGenerator(Pipeline):
    """Generate text from a prompt, streaming tokens via SSE."""

    gpu = "A100"
    min_vram_gb = 40

    def setup(self) -> None:
        """Load the language model.

        In production this would load an LLM (e.g. LLaMA, Mistral).
        """
        print("TextGenerator: setup complete (stub)")

    def predict(
        self,
        prompt: str = Input(description="The input prompt"),
        max_tokens: int = Input(
            description="Maximum number of tokens to generate",
            default=256,
            ge=1,
            le=4096,
        ),
        temperature: float = Input(
            description="Sampling temperature",
            default=0.7,
            ge=0.0,
            le=2.0,
        ),
        system_prompt: str = Input(
            description="System prompt to guide the model",
            default="You are a helpful assistant.",
        ),
    ) -> Output(type="text_stream", description="Generated text tokens"):
        """Generate text tokens one at a time.

        Yields strings that the SDK streams as SSE events.
        A real implementation would run autoregressive decoding.
        """
        # Stub: simulate token-by-token generation
        words = (
            f"This is a simulated response to: '{prompt}'. "
            "In production this would be real LLM output streamed "
            "token by token as the model generates them."
        ).split()

        for word in words[:max_tokens]:
            time.sleep(0.05)  # Simulate inference latency
            yield word + " "
