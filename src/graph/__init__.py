"""Graph module for NetworkX-based tool relationship storage.

This module provides:
- GraphClient: Main wrapper around NetworkX MultiDiGraph
- ToolGraphBuilder: Orchestrates graph construction from ToolRegistry
- Node type constants: TOOL, ENDPOINT, DOMAIN
- Edge type constants: HAS_ENDPOINT, IN_DOMAIN, SAME_DOMAIN, SEMANTICALLY_SIMILAR

Example:
    >>> from src.graph import GraphClient, TOOL, ENDPOINT, HAS_ENDPOINT
    >>> from src.config import GraphConfig
    >>> config = GraphConfig()
    >>> client = GraphClient(config)
    >>> client.add_node("tool_1", TOOL, name="Weather API")
    >>> client.add_node("endpoint_1", ENDPOINT, name="get_weather")
    >>> client.add_edge("tool_1", "endpoint_1", HAS_ENDPOINT)
    >>> client.save_graph()
"""

from .builder import BuildStats, DuplicateNodeError, SimilarityStats, ToolGraphBuilder, sanitize_domain_id
from .client import GraphClient
from .constants import (
    DOMAIN,
    ENDPOINT,
    HAS_ENDPOINT,
    IN_DOMAIN,
    SAME_DOMAIN,
    SCHEMA_VERSION,
    SEMANTICALLY_SIMILAR,
    TOOL,
    TYPE_ATTR,
)
from .embeddings import EmbeddingCache, EmbeddingDimensionError, EmbeddingGenerator

__all__ = [
    # Client
    "GraphClient",
    # Builder
    "ToolGraphBuilder",
    "BuildStats",
    "SimilarityStats",
    "DuplicateNodeError",
    "sanitize_domain_id",
    # Embeddings
    "EmbeddingCache",
    "EmbeddingGenerator",
    "EmbeddingDimensionError",
    # Node types
    "TOOL",
    "ENDPOINT",
    "DOMAIN",
    # Edge types
    "HAS_ENDPOINT",
    "IN_DOMAIN",
    "SAME_DOMAIN",
    "SEMANTICALLY_SIMILAR",
    # Internal constants (for advanced use)
    "TYPE_ATTR",
    "SCHEMA_VERSION",
]
