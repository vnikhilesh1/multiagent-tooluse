"""Schema inference engine using LLM for completing tool definitions.

Uses Claude API to infer missing fields in tool schemas:
- Descriptions for tools, endpoints, and parameters
- Parameter types (converting UNKNOWN to specific types)
- Response schemas for endpoints
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Set

from anthropic import Anthropic

from src.logging_config import get_logger
from src.models import (
    Endpoint,
    Parameter,
    ParameterType,
    Tool,
    ToolRegistry,
    truncate_description,
)

logger = get_logger(__name__)


class CacheManager(Protocol):
    """Protocol for cache manager interface.

    Any cache manager implementation must provide get and set methods.
    """

    def get(self, key: str, namespace: str = "") -> Optional[Any]:
        """Get a value from cache."""
        ...

    def set(
        self, key: str, value: Any, namespace: str = "", ttl: Optional[int] = None
    ) -> None:
        """Set a value in cache."""
        ...


@dataclass
class InferenceResult:
    """Result of an inference operation.

    Attributes:
        tool_id: ID of the tool that was processed
        endpoint_id: ID of the endpoint (if applicable)
        field_name: Name of the field that was inferred
        original_value: Original value before inference
        inferred_value: New value from LLM inference
        confidence: Confidence score (0.0-1.0) if available
        cached: Whether this result came from cache
    """

    tool_id: str
    endpoint_id: Optional[str] = None
    field_name: str = ""
    original_value: Any = None
    inferred_value: Any = None
    confidence: float = 1.0
    cached: bool = False


@dataclass
class InferenceStats:
    """Statistics from an inference run.

    Attributes:
        tools_processed: Number of tools examined
        tools_updated: Number of tools with at least one inference
        descriptions_inferred: Count of description inferences
        types_inferred: Count of parameter type inferences
        response_schemas_inferred: Count of response schema inferences
        cache_hits: Number of inferences retrieved from cache
        llm_calls: Number of actual LLM API calls made
        errors: List of error messages encountered
    """

    tools_processed: int = 0
    tools_updated: int = 0
    descriptions_inferred: int = 0
    types_inferred: int = 0
    response_schemas_inferred: int = 0
    cache_hits: int = 0
    llm_calls: int = 0
    errors: List[str] = field(default_factory=list)


class SchemaInferenceEngine:
    """LLM-powered engine for completing incomplete tool schemas.

    Identifies tools with missing or incomplete information and uses
    Claude API to infer the missing fields. All inferences are cached
    to avoid redundant API calls.

    Example:
        >>> engine = SchemaInferenceEngine(api_key="sk-...", cache_manager=cache)
        >>> stats = engine.infer_registry(registry, completeness_threshold=0.8)
        >>> print(f"Updated {stats.tools_updated} tools")

        >>> # Infer single tool
        >>> results = engine.infer_tool(tool)

        >>> # Check what was inferred
        >>> for result in results:
        ...     print(f"{result.field_name}: {result.inferred_value}")

    Attributes:
        _inferred_fields: Set of (tool_id, endpoint_id, field_name) tuples
                          tracking which fields were LLM-generated
    """

    # Cache namespace for inference results
    CACHE_NAMESPACE = "schema_inference"

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_manager: Optional[CacheManager] = None,
        model: str = "claude-sonnet-4-20250514",
        base_url: Optional[str] = None,
        max_description_length: int = 500,
        completeness_threshold: float = 0.7,
    ):
        """Initialize the inference engine.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            cache_manager: Cache manager for storing inferences (optional)
            model: Claude model to use for inference
            base_url: Base URL for API (defaults to LLM_BASE_URL env var or Hyperspace proxy)
            max_description_length: Maximum length for generated descriptions
            completeness_threshold: Default threshold for identifying incomplete tools
        """
        import os
        from dotenv import load_dotenv

        load_dotenv()

        # Get API key from param or environment
        self._api_key = api_key or os.getenv(
            "ANTHROPIC_API_KEY", "31d1207b-312d-4faf-85b2-ca6d750ed60b"
        )
        self._base_url = base_url or os.getenv(
            "LLM_BASE_URL", "http://localhost:6655/anthropic"
        )
        self._cache_manager = cache_manager
        self._model = model
        self._max_description_length = max_description_length
        self._completeness_threshold = completeness_threshold

        # Initialize Anthropic client with Hyperspace proxy
        self._client = Anthropic(
            api_key=self._api_key,
            base_url=self._base_url,
        )

        # Track which fields were inferred
        self._inferred_fields: Set[tuple] = set()

    @property
    def inferred_fields(self) -> Set[tuple]:
        """Get set of all fields that were inferred by LLM.

        Returns:
            Set of (tool_id, endpoint_id, field_name) tuples
        """
        return self._inferred_fields.copy()

    def was_inferred(
        self, tool_id: str, field_name: str, endpoint_id: Optional[str] = None
    ) -> bool:
        """Check if a specific field was inferred by LLM.

        Args:
            tool_id: Tool identifier
            field_name: Name of the field to check
            endpoint_id: Endpoint identifier (None for tool-level fields)

        Returns:
            True if the field was LLM-inferred
        """
        return (tool_id, endpoint_id, field_name) in self._inferred_fields

    def find_incomplete_tools(
        self,
        registry: ToolRegistry,
        threshold: Optional[float] = None,
    ) -> List[Tool]:
        """Find tools with completeness score below threshold.

        Args:
            registry: Tool registry to search
            threshold: Completeness threshold (uses default if None)

        Returns:
            List of Tool instances below the threshold
        """
        threshold = threshold if threshold is not None else self._completeness_threshold
        incomplete = []

        for tool in registry.tools.values():
            # Calculate completeness based on description and endpoints
            completeness = self._calculate_tool_completeness(tool)
            if completeness < threshold:
                incomplete.append(tool)

        return incomplete

    def _calculate_tool_completeness(self, tool: Tool) -> float:
        """Calculate completeness score for a tool.

        Factors:
        - Tool has description: 20%
        - Endpoints have descriptions: 30%
        - Parameters have descriptions: 25%
        - Parameters have known types: 25%

        Args:
            tool: Tool to evaluate

        Returns:
            Completeness score between 0.0 and 1.0
        """
        scores = []

        # Tool description (20%)
        has_tool_desc = bool(tool.description and tool.description.strip())
        scores.append(1.0 if has_tool_desc else 0.0)

        if not tool.endpoints:
            # No endpoints, base score on tool description only
            return scores[0] * 0.2

        # Endpoint descriptions (30%)
        ep_desc_count = sum(
            1 for ep in tool.endpoints if ep.description and ep.description.strip()
        )
        ep_desc_score = ep_desc_count / len(tool.endpoints) if tool.endpoints else 0.0
        scores.append(ep_desc_score)

        # Parameter descriptions (25%)
        total_params = sum(len(ep.parameters) for ep in tool.endpoints)
        if total_params > 0:
            param_desc_count = sum(
                1
                for ep in tool.endpoints
                for p in ep.parameters
                if p.description and p.description.strip()
            )
            param_desc_score = param_desc_count / total_params
        else:
            param_desc_score = 1.0  # No params = complete
        scores.append(param_desc_score)

        # Parameter types (25%)
        if total_params > 0:
            known_type_count = sum(
                1
                for ep in tool.endpoints
                for p in ep.parameters
                if p.type != ParameterType.UNKNOWN
            )
            type_score = known_type_count / total_params
        else:
            type_score = 1.0  # No params = complete
        scores.append(type_score)

        # Weighted average
        weights = [0.2, 0.3, 0.25, 0.25]
        return sum(s * w for s, w in zip(scores, weights))

    def infer_registry(
        self,
        registry: ToolRegistry,
        completeness_threshold: Optional[float] = None,
        infer_descriptions: bool = True,
        infer_types: bool = True,
        infer_responses: bool = True,
    ) -> InferenceStats:
        """Infer missing fields for all incomplete tools in registry.

        Args:
            registry: Tool registry to process
            completeness_threshold: Override default threshold
            infer_descriptions: Whether to infer missing descriptions
            infer_types: Whether to infer unknown parameter types
            infer_responses: Whether to infer response schemas

        Returns:
            InferenceStats with counts of all operations
        """
        stats = InferenceStats()

        incomplete_tools = self.find_incomplete_tools(registry, completeness_threshold)
        stats.tools_processed = len(incomplete_tools)

        for tool in incomplete_tools:
            try:
                results = self.infer_tool(
                    tool,
                    infer_descriptions=infer_descriptions,
                    infer_types=infer_types,
                    infer_responses=infer_responses,
                )

                if results:
                    stats.tools_updated += 1

                for result in results:
                    if result.cached:
                        stats.cache_hits += 1
                    else:
                        stats.llm_calls += 1

                    if result.field_name == "description":
                        stats.descriptions_inferred += 1
                    elif result.field_name == "type":
                        stats.types_inferred += 1
                    elif result.field_name == "response_schema":
                        stats.response_schemas_inferred += 1

            except Exception as e:
                error_msg = f"Error processing tool {tool.id}: {e}"
                logger.error(error_msg)
                stats.errors.append(error_msg)

        return stats

    def infer_tool(
        self,
        tool: Tool,
        infer_descriptions: bool = True,
        infer_types: bool = True,
        infer_responses: bool = True,
    ) -> List[InferenceResult]:
        """Infer missing fields for a single tool.

        Args:
            tool: Tool to process
            infer_descriptions: Whether to infer missing descriptions
            infer_types: Whether to infer unknown parameter types
            infer_responses: Whether to infer response schemas

        Returns:
            List of InferenceResult for each inferred field
        """
        results = []

        # Infer tool description
        if infer_descriptions and not (
            tool.description and tool.description.strip()
        ):
            result = self.infer_description(tool)
            if result:
                results.append(result)

        # Process each endpoint
        for endpoint in tool.endpoints:
            # Infer endpoint description
            if infer_descriptions and not (
                endpoint.description and endpoint.description.strip()
            ):
                result = self.infer_description(tool, endpoint)
                if result:
                    results.append(result)

            # Process parameters
            for param in endpoint.parameters:
                # Infer parameter description
                if infer_descriptions and not (
                    param.description and param.description.strip()
                ):
                    result = self.infer_description(tool, endpoint, param)
                    if result:
                        results.append(result)

                # Infer parameter type
                if infer_types and param.type == ParameterType.UNKNOWN:
                    result = self.infer_parameter_type(tool, endpoint, param)
                    if result:
                        results.append(result)

            # Infer response schema
            if infer_responses and endpoint.response_schema is None:
                result = self.infer_response_schema(tool, endpoint)
                if result:
                    results.append(result)

        return results

    def infer_description(
        self,
        tool: Tool,
        endpoint: Optional[Endpoint] = None,
        parameter: Optional[Parameter] = None,
    ) -> Optional[InferenceResult]:
        """Infer a missing description using LLM.

        Can infer descriptions for:
        - Tool (when endpoint and parameter are None)
        - Endpoint (when parameter is None)
        - Parameter (when all provided)

        Args:
            tool: The tool context
            endpoint: The endpoint context (optional)
            parameter: The parameter needing description (optional)

        Returns:
            InferenceResult if inference was made, None otherwise
        """
        # Determine what we're inferring
        if parameter:
            if parameter.description and parameter.description.strip():
                return None
            field_name = "description"
            endpoint_id = endpoint.id if endpoint else None
            cache_key = self._get_cache_key(
                "description", tool.id, endpoint_id, parameter.name
            )
        elif endpoint:
            if endpoint.description and endpoint.description.strip():
                return None
            field_name = "description"
            endpoint_id = endpoint.id
            cache_key = self._get_cache_key("description", tool.id, endpoint_id)
        else:
            if tool.description and tool.description.strip():
                return None
            field_name = "description"
            endpoint_id = None
            cache_key = self._get_cache_key("description", tool.id)

        # Check cache
        cached_value = self._check_cache(cache_key)
        if cached_value is not None:
            self._track_inference(tool.id, field_name, endpoint_id)
            return InferenceResult(
                tool_id=tool.id,
                endpoint_id=endpoint_id,
                field_name=field_name,
                original_value="",
                inferred_value=cached_value,
                cached=True,
            )

        # Build prompt and call LLM
        prompt = self._build_description_prompt(tool, endpoint, parameter)
        response = self._call_llm(prompt)

        if response is None:
            return None

        # Truncate and clean description
        description = truncate_description(response, self._max_description_length)

        # Store in cache
        self._store_cache(cache_key, description)

        # Track inference
        self._track_inference(tool.id, field_name, endpoint_id)

        return InferenceResult(
            tool_id=tool.id,
            endpoint_id=endpoint_id,
            field_name=field_name,
            original_value="",
            inferred_value=description,
            cached=False,
        )

    def infer_parameter_type(
        self,
        tool: Tool,
        endpoint: Endpoint,
        parameter: Parameter,
    ) -> Optional[InferenceResult]:
        """Infer a parameter's type when it's UNKNOWN.

        Args:
            tool: The tool context
            endpoint: The endpoint context
            parameter: The parameter with unknown type

        Returns:
            InferenceResult if type was inferred, None otherwise
        """
        if parameter.type != ParameterType.UNKNOWN:
            return None

        cache_key = self._get_cache_key(
            "type", tool.id, endpoint.id, parameter.name
        )

        # Check cache
        cached_value = self._check_cache(cache_key)
        if cached_value is not None:
            inferred_type = self._parse_type_response(cached_value)
            self._track_inference(tool.id, "type", endpoint.id)
            return InferenceResult(
                tool_id=tool.id,
                endpoint_id=endpoint.id,
                field_name="type",
                original_value=ParameterType.UNKNOWN,
                inferred_value=inferred_type,
                cached=True,
            )

        # Build prompt and call LLM
        prompt = self._build_type_prompt(tool, endpoint, parameter)
        response = self._call_llm(prompt, max_tokens=50)

        if response is None:
            return None

        # Parse type from response
        inferred_type = self._parse_type_response(response)

        # Store raw response in cache
        self._store_cache(cache_key, response)

        # Track inference
        self._track_inference(tool.id, "type", endpoint.id)

        return InferenceResult(
            tool_id=tool.id,
            endpoint_id=endpoint.id,
            field_name="type",
            original_value=ParameterType.UNKNOWN,
            inferred_value=inferred_type,
            cached=False,
        )

    def infer_response_schema(
        self,
        tool: Tool,
        endpoint: Endpoint,
    ) -> Optional[InferenceResult]:
        """Infer the response schema for an endpoint.

        Args:
            tool: The tool context
            endpoint: The endpoint needing response schema

        Returns:
            InferenceResult with response schema dict, None if failed
        """
        if endpoint.response_schema is not None:
            return None

        cache_key = self._get_cache_key("response", tool.id, endpoint.id)

        # Check cache
        cached_value = self._check_cache(cache_key)
        if cached_value is not None:
            schema = self._parse_response_schema(cached_value)
            if schema is not None:
                self._track_inference(tool.id, "response_schema", endpoint.id)
                return InferenceResult(
                    tool_id=tool.id,
                    endpoint_id=endpoint.id,
                    field_name="response_schema",
                    original_value=None,
                    inferred_value=schema,
                    cached=True,
                )

        # Build prompt and call LLM
        prompt = self._build_response_prompt(tool, endpoint)
        response = self._call_llm(prompt, max_tokens=512)

        if response is None:
            return None

        # Parse schema from response
        schema = self._parse_response_schema(response)

        if schema is None:
            return None

        # Store raw response in cache
        self._store_cache(cache_key, response)

        # Track inference
        self._track_inference(tool.id, "response_schema", endpoint.id)

        return InferenceResult(
            tool_id=tool.id,
            endpoint_id=endpoint.id,
            field_name="response_schema",
            original_value=None,
            inferred_value=schema,
            cached=False,
        )

    def _build_description_prompt(
        self,
        tool: Tool,
        endpoint: Optional[Endpoint] = None,
        parameter: Optional[Parameter] = None,
    ) -> str:
        """Build prompt for description inference.

        Args:
            tool: Tool context
            endpoint: Endpoint context (optional)
            parameter: Parameter needing description (optional)

        Returns:
            Formatted prompt string
        """
        if parameter:
            return f"""You are an API documentation expert. Generate a concise description for a parameter.

Tool: {tool.name}
Category: {tool.category}
Endpoint: {endpoint.name if endpoint else 'Unknown'}
Endpoint Description: {endpoint.description if endpoint and endpoint.description else 'Not provided'}
Parameter Name: {parameter.name}
Parameter Type: {parameter.type.value}
Required: {parameter.required}

Generate a clear, concise description (1-2 sentences) for this parameter.
Focus on what the parameter does and any constraints.
Respond with ONLY the description text, no quotes or extra formatting."""

        elif endpoint:
            param_names = ", ".join(p.name for p in endpoint.parameters) or "None"
            return f"""You are an API documentation expert. Generate a concise description for an API endpoint.

Tool: {tool.name}
Tool Description: {tool.description or 'Not provided'}
Category: {tool.category}
Endpoint Name: {endpoint.name}
HTTP Method: {endpoint.method}
Path: {endpoint.path or 'Not provided'}
Parameters: {param_names}

Generate a clear, concise description (1-2 sentences) for this endpoint.
Focus on what the endpoint does and its purpose.
Respond with ONLY the description text, no quotes or extra formatting."""

        else:
            endpoint_names = ", ".join(e.name for e in tool.endpoints[:5])
            if len(tool.endpoints) > 5:
                endpoint_names += f", ... ({len(tool.endpoints) - 5} more)"
            return f"""You are an API documentation expert. Generate a concise description for an API tool.

Tool Name: {tool.name}
Category: {tool.category}
Number of Endpoints: {len(tool.endpoints)}
Endpoint Names: {endpoint_names or 'None'}

Generate a clear, concise description (1-2 sentences) for this API tool.
Focus on the tool's purpose and main capabilities.
Respond with ONLY the description text, no quotes or extra formatting."""

    def _build_type_prompt(
        self,
        tool: Tool,
        endpoint: Endpoint,
        parameter: Parameter,
    ) -> str:
        """Build prompt for parameter type inference.

        Args:
            tool: Tool context
            endpoint: Endpoint context
            parameter: Parameter with unknown type

        Returns:
            Formatted prompt string
        """
        default_str = (
            str(parameter.default) if parameter.default is not None else "None"
        )
        return f"""You are an API expert. Determine the data type for a parameter.

Tool: {tool.name}
Endpoint: {endpoint.name}
Endpoint Description: {endpoint.description or 'Not provided'}
Parameter Name: {parameter.name}
Parameter Description: {parameter.description or 'Not provided'}
Default Value: {default_str}

What is the most likely data type for this parameter?
Choose exactly one from: string, integer, number, boolean, array, object

Respond with ONLY the type name (e.g., "string"), nothing else."""

    def _build_response_prompt(
        self,
        tool: Tool,
        endpoint: Endpoint,
    ) -> str:
        """Build prompt for response schema inference.

        Args:
            tool: Tool context
            endpoint: Endpoint needing response schema

        Returns:
            Formatted prompt string
        """
        params_info = "\n".join(
            f"  - {p.name}: {p.type.value} {'(required)' if p.required else '(optional)'}"
            for p in endpoint.parameters
        )

        return f"""You are an API expert. Generate a likely JSON response schema for an API endpoint.

Tool: {tool.name}
Category: {tool.category}
Endpoint: {endpoint.name}
Description: {endpoint.description or 'Not provided'}
HTTP Method: {endpoint.method}
Path: {endpoint.path or 'Not provided'}
Parameters:
{params_info or '  None'}

Generate a JSON schema for the likely response. Include:
- Likely response fields based on the endpoint purpose
- Appropriate data types for each field
- Brief descriptions for complex fields

Respond with ONLY valid JSON (no markdown, no explanation), for example:
{{"status": "string", "data": {{"id": "integer", "name": "string"}}}}"""

    def _call_llm(self, prompt: str, max_tokens: int = 1024) -> Optional[str]:
        """Make an LLM API call.

        Args:
            prompt: The prompt to send
            max_tokens: Maximum response tokens

        Returns:
            Response text, or None if call failed
        """
        try:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text.strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None

    def _get_cache_key(
        self,
        operation: str,
        tool_id: str,
        endpoint_id: Optional[str] = None,
        param_name: Optional[str] = None,
    ) -> str:
        """Generate a cache key for an inference operation.

        Args:
            operation: Type of inference (description, type, response)
            tool_id: Tool identifier
            endpoint_id: Endpoint identifier (optional)
            param_name: Parameter name (optional)

        Returns:
            Cache key string
        """
        parts = [self.CACHE_NAMESPACE, operation, tool_id]
        if endpoint_id:
            parts.append(endpoint_id)
        if param_name:
            parts.append(param_name)
        return ":".join(parts)

    def _check_cache(self, cache_key: str) -> Optional[Any]:
        """Check cache for existing inference.

        Args:
            cache_key: Key to look up

        Returns:
            Cached value if exists, None otherwise
        """
        try:
            return self._cache_manager.get(cache_key, namespace=self.CACHE_NAMESPACE)
        except Exception as e:
            logger.warning(f"Cache lookup failed: {e}")
            return None

    def _store_cache(self, cache_key: str, value: Any) -> None:
        """Store an inference result in cache.

        Args:
            cache_key: Key to store under
            value: Value to cache
        """
        try:
            self._cache_manager.set(
                cache_key, value, namespace=self.CACHE_NAMESPACE
            )
        except Exception as e:
            logger.warning(f"Cache store failed: {e}")

    def _track_inference(
        self,
        tool_id: str,
        field_name: str,
        endpoint_id: Optional[str] = None,
    ) -> None:
        """Record that a field was inferred by LLM.

        Args:
            tool_id: Tool identifier
            field_name: Name of inferred field
            endpoint_id: Endpoint identifier (optional)
        """
        self._inferred_fields.add((tool_id, endpoint_id, field_name))

    def _parse_type_response(self, response: str) -> ParameterType:
        """Parse LLM response into a ParameterType.

        Args:
            response: Raw LLM response text

        Returns:
            Parsed ParameterType (UNKNOWN if parsing fails)
        """
        response_lower = response.lower().strip()

        # Direct mapping
        type_map = {
            "string": ParameterType.STRING,
            "integer": ParameterType.INTEGER,
            "int": ParameterType.INTEGER,
            "number": ParameterType.NUMBER,
            "float": ParameterType.NUMBER,
            "boolean": ParameterType.BOOLEAN,
            "bool": ParameterType.BOOLEAN,
            "array": ParameterType.ARRAY,
            "list": ParameterType.ARRAY,
            "object": ParameterType.OBJECT,
            "dict": ParameterType.OBJECT,
        }

        for key, value in type_map.items():
            if key in response_lower:
                return value

        return ParameterType.UNKNOWN

    def _parse_response_schema(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response into a response schema dict.

        Args:
            response: Raw LLM response text

        Returns:
            Parsed schema dict, or None if parsing fails
        """
        # Clean up response - remove markdown code blocks if present
        cleaned = response.strip()

        # Remove ```json and ``` markers
        if cleaned.startswith("```"):
            # Find the end of the first line
            first_newline = cleaned.find("\n")
            if first_newline != -1:
                cleaned = cleaned[first_newline + 1 :]

        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        cleaned = cleaned.strip()

        # Try to extract JSON from the response
        # Sometimes LLMs include extra text before/after JSON
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            cleaned = json_match.group(0)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse response schema: {response[:100]}...")
            return None
