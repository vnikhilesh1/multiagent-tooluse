"""LLM Client wrapper around Anthropic SDK.

This module provides a unified interface for interacting with the Anthropic API,
with support for Hyperspace local proxy, multiple completion modes, and robust
error handling with retries.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Type, TypeVar

import anthropic
from pydantic import BaseModel, Field, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.llm.exceptions import (
    LLMAPIError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMValidationError,
)

T = TypeVar("T", bound=BaseModel)


class LLMResponse(BaseModel):
    """Structured response from LLM calls.

    Attributes:
        content: The text content of the response.
        model: The model that generated the response.
        usage: Token usage statistics.
        stop_reason: Why the model stopped generating.
        raw_response: The complete raw response for advanced use cases.
    """

    content: str
    model: str
    usage: Dict[str, int] = Field(default_factory=dict)
    stop_reason: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None


class LLMClient:
    """Wrapper around Anthropic SDK with Hyperspace proxy support.

    Provides multiple completion modes:
    - complete(): Simple text completion
    - complete_json(): JSON parsing
    - complete_structured(): Pydantic model validation
    - complete_with_tools(): Function calling
    - chat(): Multi-turn conversations

    Example:
        >>> client = LLMClient(api_key="sk-...")
        >>> response = client.complete("Hello, world!")
        >>> print(response)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 60.0,
    ):
        """Initialize LLM client.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var).
            base_url: Optional base URL for Hyperspace proxy
                (defaults to LLM_BASE_URL env var).
            default_model: Default model to use
                (defaults to LLM_MODEL env var or claude-sonnet-4-20250514).
            max_retries: Maximum retry attempts for transient errors.
            timeout: Request timeout in seconds.
        """
        from dotenv import load_dotenv

        load_dotenv()

        # Default API key for Hyperspace proxy
        self.api_key = api_key or os.environ.get(
            "ANTHROPIC_API_KEY", "31d1207b-312d-4faf-85b2-ca6d750ed60b"
        )
        self.base_url = base_url or os.environ.get(
            "LLM_BASE_URL", "http://localhost:6655/anthropic"
        )
        self.default_model = default_model or os.environ.get(
            "LLM_MODEL", "claude-sonnet-4-20250514"
        )
        self.max_retries = max_retries
        self.timeout = timeout

        # Build client kwargs
        client_kwargs: Dict[str, Any] = {
            "api_key": self.api_key,
            "base_url": self.base_url,
        }
        if self.timeout:
            client_kwargs["timeout"] = self.timeout

        # Initialize the Anthropic client
        self._client = anthropic.Anthropic(**client_kwargs)

    def complete(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system: Optional[str] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """Simple text completion returning just the response string.

        Args:
            prompt: User prompt.
            model: Model override (uses default if not specified).
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature (0.0-1.0).
            system: Optional system prompt.
            stop_sequences: Optional stop sequences.

        Returns:
            Response text as string.

        Raises:
            LLMError: On API or validation errors.
        """
        messages = [{"role": "user", "content": prompt}]
        response = self._make_request(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            stop_sequences=stop_sequences,
        )
        return response.content

    def complete_json(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        system: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Completion with JSON parsing.

        Automatically adds JSON formatting instructions and parses response.

        Args:
            prompt: User prompt (should describe desired JSON structure).
            model: Model override.
            max_tokens: Maximum tokens.
            temperature: Sampling temperature.
            system: Optional system prompt (JSON instruction will be appended).

        Returns:
            Parsed JSON as dictionary.

        Raises:
            LLMValidationError: If response is not valid JSON.
        """
        # Build system prompt with JSON instruction
        json_instruction = (
            "You must respond with valid JSON only. "
            "Do not include any explanation or markdown formatting."
        )
        if system:
            full_system = f"{system}\n\n{json_instruction}"
        else:
            full_system = json_instruction

        messages = [{"role": "user", "content": prompt}]
        response = self._make_request(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=full_system,
        )

        return self._extract_json(response.content)

    def complete_structured(
        self,
        prompt: str,
        response_model: Type[T],
        *,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        system: Optional[str] = None,
    ) -> T:
        """Completion with Pydantic model validation.

        Uses the model's JSON schema to guide output format.

        Args:
            prompt: User prompt.
            response_model: Pydantic model class for response validation.
            model: Model override.
            max_tokens: Maximum tokens.
            temperature: Sampling temperature.
            system: Optional system prompt.

        Returns:
            Instance of response_model populated with response data.

        Raises:
            LLMValidationError: If response doesn't match model schema.
        """
        # Get JSON schema from Pydantic model
        schema = response_model.model_json_schema()
        schema_str = json.dumps(schema, indent=2)

        # Build system prompt with schema
        schema_instruction = (
            f"You must respond with valid JSON that matches this schema:\n"
            f"{schema_str}\n\n"
            "Respond with JSON only, no explanation or markdown."
        )
        if system:
            full_system = f"{system}\n\n{schema_instruction}"
        else:
            full_system = schema_instruction

        messages = [{"role": "user", "content": prompt}]
        response = self._make_request(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=full_system,
        )

        # Parse JSON and validate against model
        json_data = self._extract_json(response.content)
        try:
            return response_model.model_validate(json_data)
        except ValidationError as e:
            raise LLMValidationError(
                f"Response does not match expected schema: {e}"
            ) from e

    def complete_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        system: Optional[str] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """Function calling with Anthropic tool_use format.

        Args:
            prompt: User prompt.
            tools: List of tool definitions in Anthropic format:
                [{"name": "tool_name", "description": "...",
                  "input_schema": {"type": "object", "properties": {...}}}]
            model: Model override.
            max_tokens: Maximum tokens.
            temperature: Sampling temperature.
            system: Optional system prompt.
            tool_choice: Optional tool choice constraint:
                {"type": "auto"} - Model decides (default)
                {"type": "any"} - Must use a tool
                {"type": "tool", "name": "specific_tool"} - Use specific tool

        Returns:
            LLMResponse with tool_use content blocks in raw_response.

        Raises:
            LLMError: On API errors.
        """
        messages = [{"role": "user", "content": prompt}]
        return self._make_request(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            tools=tools,
            tool_choice=tool_choice,
        )

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Multi-turn conversation support.

        Args:
            messages: List of message dicts in Anthropic format:
                [{"role": "user", "content": "..."},
                 {"role": "assistant", "content": "..."}]
            model: Model override.
            max_tokens: Maximum tokens.
            temperature: Sampling temperature.
            system: Optional system prompt.
            tools: Optional tools for function calling.

        Returns:
            LLMResponse with full response details.

        Raises:
            LLMError: On API errors.
        """
        return self._make_request(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            tools=tools,
        )

    def _make_request(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> LLMResponse:
        """Internal method to make API request with retry logic.

        Handles:
        - Retry with exponential backoff for rate limits
        - Error translation to custom exceptions
        - Response parsing to LLMResponse
        """
        # Create retry decorator with configured max_retries
        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            retry=retry_if_exception_type(
                (anthropic.RateLimitError, anthropic.APIConnectionError)
            ),
            reraise=True,
        )
        def _request_with_retry() -> LLMResponse:
            return self._execute_request(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                tools=tools,
                tool_choice=tool_choice,
                stop_sequences=stop_sequences,
            )

        return _request_with_retry()

    def _execute_request(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> LLMResponse:
        """Execute a single API request without retry logic."""
        try:
            # Build request kwargs
            request_kwargs: Dict[str, Any] = {
                "model": model or self.default_model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            if system:
                request_kwargs["system"] = system
            if tools:
                request_kwargs["tools"] = tools
            if tool_choice:
                request_kwargs["tool_choice"] = tool_choice
            if stop_sequences:
                request_kwargs["stop_sequences"] = stop_sequences

            # Make the API call
            response = self._client.messages.create(**request_kwargs)

            # Parse response
            return self._parse_response(response)

        except anthropic.RateLimitError as e:
            retry_after = getattr(e, "retry_after", None)
            raise LLMRateLimitError(str(e), retry_after=retry_after) from e
        except anthropic.APIConnectionError as e:
            raise LLMConnectionError(f"Connection error: {e}") from e
        except anthropic.APIStatusError as e:
            raise LLMAPIError(str(e), status_code=e.status_code) from e
        except anthropic.APIError as e:
            raise LLMError(f"API error: {e}") from e

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse Anthropic API response into LLMResponse.

        Args:
            response: Raw response from Anthropic API.

        Returns:
            Structured LLMResponse.
        """
        # Extract text content from response
        content_parts = []
        for block in response.content:
            if hasattr(block, "text"):
                content_parts.append(block.text)

        content = "".join(content_parts)

        # Build usage dict
        usage = {}
        if hasattr(response, "usage"):
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }

        # Build raw response dict for tool use cases
        raw_response: Dict[str, Any] = {
            "content": [
                self._content_block_to_dict(block) for block in response.content
            ],
            "model": response.model,
            "stop_reason": response.stop_reason,
        }

        return LLMResponse(
            content=content,
            model=response.model,
            usage=usage,
            stop_reason=response.stop_reason,
            raw_response=raw_response,
        )

    def _content_block_to_dict(self, block: Any) -> Dict[str, Any]:
        """Convert a content block to a dictionary.

        Args:
            block: Content block from Anthropic response.

        Returns:
            Dictionary representation of the block.
        """
        if hasattr(block, "type"):
            if block.type == "text":
                return {"type": "text", "text": block.text}
            elif block.type == "tool_use":
                return {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                }
        # Fallback for unknown block types
        return {"type": "unknown", "data": str(block)}

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from response, handling markdown code blocks.

        Args:
            text: Response text that may contain JSON.

        Returns:
            Parsed JSON as dictionary.

        Raises:
            LLMValidationError: If JSON cannot be extracted or parsed.
        """
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding JSON object/array pattern
        json_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        raise LLMValidationError(
            f"Could not parse JSON from response: {text[:200]}..."
        )


def pydantic_to_tool(
    name: str,
    description: str,
    model: Type[BaseModel],
) -> Dict[str, Any]:
    """Convert a Pydantic model to Anthropic tool definition format.

    Args:
        name: Tool name.
        description: Tool description.
        model: Pydantic model for input schema.

    Returns:
        Tool definition dict for use with complete_with_tools().

    Example:
        >>> class WeatherInput(BaseModel):
        ...     location: str
        ...     units: str = "celsius"
        >>> tool = pydantic_to_tool("get_weather", "Get weather", WeatherInput)
    """
    return {
        "name": name,
        "description": description,
        "input_schema": model.model_json_schema(),
    }


def create_client_from_config(config: Any) -> LLMClient:
    """Create LLMClient from application config.

    Args:
        config: Application Config instance (from src.config.Config).

    Returns:
        Configured LLMClient.

    Example:
        >>> from src.config import load_config
        >>> config = load_config()
        >>> client = create_client_from_config(config)
    """
    return LLMClient(
        default_model=config.models.primary,
        max_retries=config.quality.max_retries,
    )
