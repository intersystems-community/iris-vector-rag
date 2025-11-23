"""
Monitoring and observability module for iris-vector-rag.

This module provides OpenTelemetry integration for production observability,
including query latency tracking, token usage monitoring, and cost tracking.

Classes:
    TelemetryManager: Central telemetry manager with lazy initialization

Functions:
    configure_telemetry: Configure global telemetry settings
    calculate_llm_cost: Calculate LLM API costs based on token usage

Features:
    - Zero overhead when disabled (lazy initialization)
    - <5% overhead when enabled
    - OpenTelemetry GenAI semantic conventions
    - OTLP export support (HTTP/gRPC)
    - Automatic span context propagation
    - Cost tracking for OpenAI, Anthropic models

Example:
    >>> from iris_vector_rag.monitoring import configure_telemetry
    >>>
    >>> # Enable telemetry with OTLP export
    >>> configure_telemetry(
    ...     enabled=True,
    ...     service_name="my-rag-service",
    ...     endpoint="http://localhost:4318",
    ...     sampling_ratio=1.0
    ... )
    >>>
    >>> # Telemetry automatically instruments RAG operations
    >>> pipeline = create_pipeline("basic")
    >>> result = pipeline.query("What is diabetes?")  # Auto-traced
    >>>
    >>> # Manual instrumentation
    >>> from iris_vector_rag.monitoring import telemetry
    >>>
    >>> with telemetry.trace_operation("custom.operation") as span:
    ...     span.set_attribute("custom.param", "value")
    ...     # Your code here
    ...     pass
"""

from iris_vector_rag.monitoring.telemetry import (
    TelemetryManager,
    configure_telemetry,
    telemetry,
)
from iris_vector_rag.monitoring.cost_tracking import calculate_llm_cost

__all__ = [
    "TelemetryManager",
    "configure_telemetry",
    "telemetry",
    "calculate_llm_cost",
]
