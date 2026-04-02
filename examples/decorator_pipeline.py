"""Example: defining a pipeline using the decorator shortcut.

This demonstrates the decorator approach for simple stateless pipelines
where a full class definition is unnecessary.

Usage:

    # Run a local prediction
    livepeer predict examples/decorator_pipeline.py -i text="hello world"

    # Print the auto-generated schema
    livepeer schema examples/decorator_pipeline.py

    # Start a local HTTP server
    livepeer serve examples/decorator_pipeline.py
"""

from livepeer_gateway.runner import Input, Output
from livepeer_gateway.runner.decorators import pipeline


@pipeline(gpu="T4", pipeline_id="summarize", version="1.0.0", description="Summarize text")
def summarize(
    text: str = Input(description="Text to summarize", min_length=1),
    max_length: int = Input(
        description="Maximum summary length in words",
        default=50,
        ge=10,
        le=500,
    ),
) -> Output(type="text", description="Summarized text"):
    """Summarize input text.

    This stub returns a truncated version. A real implementation
    would use an LLM for summarization.
    """
    words = text.split()
    if len(words) <= max_length:
        return text
    return " ".join(words[:max_length]) + "..."
