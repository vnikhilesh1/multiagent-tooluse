"""LLM client module for Anthropic API interactions.

This module provides a unified interface for LLM operations with support for:
- Simple text completions
- JSON and structured output parsing
- Function/tool calling
- Multi-turn conversations
- Automatic retries with exponential backoff
- Hyperspace proxy support for local development
- Two-tier caching (memory + disk) for response caching

Example:
    >>> from src.llm import LLMClient, LLMCache
    >>> cache = LLMCache()
    >>> client = LLMClient(api_key="sk-...")
    >>> response = client.complete("Hello!")
    >>> print(response)
"""

from src.llm.cache import (
    CacheEntry,
    CacheStats,
    LLMCache,
    create_cache_from_config,
)
from src.llm.client import (
    LLMClient,
    LLMResponse,
    create_client_from_config,
    pydantic_to_tool,
)
from src.llm.exceptions import (
    LLMAPIError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMValidationError,
)

__all__ = [
    # Client
    "LLMClient",
    "LLMResponse",
    # Cache
    "LLMCache",
    "CacheEntry",
    "CacheStats",
    # Helpers
    "create_client_from_config",
    "create_cache_from_config",
    "pydantic_to_tool",
    # Exceptions
    "LLMError",
    "LLMRateLimitError",
    "LLMAPIError",
    "LLMValidationError",
    "LLMConnectionError",
]
