"""LLM-powered schema inference for tool completion.

This module provides inference capabilities to complete incomplete tool schemas
using LLM calls to generate missing descriptions, types, and response schemas.
"""

from src.inference.engine import SchemaInferenceEngine

__all__ = [
    "SchemaInferenceEngine",
]
