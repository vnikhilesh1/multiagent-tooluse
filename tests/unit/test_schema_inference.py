"""Unit tests for schema inference engine."""

import json
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from src.inference import SchemaInferenceEngine
from src.inference.engine import InferenceResult, InferenceStats
from src.models import Endpoint, Parameter, ParameterType, Tool, ToolRegistry


class MockCacheManager:
    """Mock cache manager for testing."""

    def __init__(self):
        self._cache = {}

    def get(self, key: str, namespace: str = "") -> Optional[Any]:
        full_key = f"{namespace}:{key}" if namespace else key
        return self._cache.get(full_key)

    def set(
        self, key: str, value: Any, namespace: str = "", ttl: Optional[int] = None
    ) -> None:
        full_key = f"{namespace}:{key}" if namespace else key
        self._cache[full_key] = value

    def clear(self):
        self._cache.clear()


class TestSchemaInferenceEngineInit:
    """Tests for SchemaInferenceEngine initialization."""

    def test_default_init(self):
        """Test default initialization."""
        cache = MockCacheManager()
        engine = SchemaInferenceEngine(
            api_key="test-key",
            cache_manager=cache,
        )
        assert engine._model == "claude-sonnet-4-20250514"
        assert engine._max_description_length == 500
        assert engine._completeness_threshold == 0.7

    def test_custom_init(self):
        """Test custom initialization."""
        cache = MockCacheManager()
        engine = SchemaInferenceEngine(
            api_key="test-key",
            cache_manager=cache,
            model="claude-opus-4-20250514",
            max_description_length=300,
            completeness_threshold=0.9,
        )
        assert engine._model == "claude-opus-4-20250514"
        assert engine._max_description_length == 300
        assert engine._completeness_threshold == 0.9


class TestFindIncompleteTools:
    """Tests for finding incomplete tools."""

    def test_finds_incomplete_tools(self):
        """Test finding tools below threshold."""
        cache = MockCacheManager()
        engine = SchemaInferenceEngine(
            api_key="test-key",
            cache_manager=cache,
            completeness_threshold=0.8,
        )

        # Create tools with different completeness
        complete_tool = Tool(
            id="complete",
            name="Complete Tool",
            category="Test",
            description="A complete tool",
            endpoints=[
                Endpoint(
                    id="ep1",
                    tool_id="complete",
                    name="endpoint1",
                    method="GET",
                    description="Complete endpoint",
                    parameters=[
                        Parameter(
                            name="param1",
                            type=ParameterType.STRING,
                            description="Complete param",
                            required=True,
                        )
                    ],
                )
            ],
        )

        incomplete_tool = Tool(
            id="incomplete",
            name="Incomplete Tool",
            category="Test",
            description="",  # Missing description
            endpoints=[
                Endpoint(
                    id="ep2",
                    tool_id="incomplete",
                    name="endpoint2",
                    method="GET",
                    description="",  # Missing description
                    parameters=[],
                )
            ],
        )

        registry = ToolRegistry()
        registry.add_tool(complete_tool)
        registry.add_tool(incomplete_tool)

        incomplete = engine.find_incomplete_tools(registry)

        assert len(incomplete) >= 1
        assert any(t.id == "incomplete" for t in incomplete)


class TestInferenceTracking:
    """Tests for tracking inferred fields."""

    def test_track_inference(self):
        """Test tracking that a field was inferred."""
        cache = MockCacheManager()
        engine = SchemaInferenceEngine(
            api_key="test-key",
            cache_manager=cache,
        )

        engine._track_inference("tool1", "description")
        engine._track_inference("tool1", "description", "endpoint1")

        assert engine.was_inferred("tool1", "description")
        assert engine.was_inferred("tool1", "description", "endpoint1")
        assert not engine.was_inferred("tool1", "other_field")

    def test_inferred_fields_property(self):
        """Test getting all inferred fields."""
        cache = MockCacheManager()
        engine = SchemaInferenceEngine(
            api_key="test-key",
            cache_manager=cache,
        )

        engine._track_inference("tool1", "description")
        engine._track_inference("tool2", "response_schema", "ep1")

        fields = engine.inferred_fields
        assert len(fields) == 2
        assert ("tool1", None, "description") in fields
        assert ("tool2", "ep1", "response_schema") in fields


class TestCaching:
    """Tests for inference caching."""

    def test_cache_key_generation(self):
        """Test cache key generation."""
        cache = MockCacheManager()
        engine = SchemaInferenceEngine(
            api_key="test-key",
            cache_manager=cache,
        )

        key1 = engine._get_cache_key("description", "tool1")
        assert "schema_inference" in key1
        assert "description" in key1
        assert "tool1" in key1

        key2 = engine._get_cache_key("type", "tool1", "ep1", "param1")
        assert "ep1" in key2
        assert "param1" in key2

    def test_cache_hit(self):
        """Test cache retrieval."""
        cache = MockCacheManager()
        engine = SchemaInferenceEngine(
            api_key="test-key",
            cache_manager=cache,
        )

        key = engine._get_cache_key("description", "tool1")
        engine._store_cache(key, "Cached description")

        result = engine._check_cache(key)
        assert result == "Cached description"

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = MockCacheManager()
        engine = SchemaInferenceEngine(
            api_key="test-key",
            cache_manager=cache,
        )

        result = engine._check_cache("nonexistent_key")
        assert result is None


class TestPromptBuilding:
    """Tests for prompt building."""

    def test_tool_description_prompt(self):
        """Test building tool description prompt."""
        cache = MockCacheManager()
        engine = SchemaInferenceEngine(
            api_key="test-key",
            cache_manager=cache,
        )

        tool = Tool(
            id="weather",
            name="Weather API",
            category="Weather",
            endpoints=[
                Endpoint(
                    id="forecast",
                    tool_id="weather",
                    name="getForecast",
                    method="GET",
                )
            ],
        )

        prompt = engine._build_description_prompt(tool)

        assert "Weather API" in prompt
        assert "Weather" in prompt
        assert "getForecast" in prompt

    def test_endpoint_description_prompt(self):
        """Test building endpoint description prompt."""
        cache = MockCacheManager()
        engine = SchemaInferenceEngine(
            api_key="test-key",
            cache_manager=cache,
        )

        tool = Tool(
            id="weather",
            name="Weather API",
            category="Weather",
            endpoints=[],
        )
        endpoint = Endpoint(
            id="forecast",
            tool_id="weather",
            name="getForecast",
            method="GET",
            path="/forecast/{city}",
        )

        prompt = engine._build_description_prompt(tool, endpoint)

        assert "getForecast" in prompt
        assert "GET" in prompt
        assert "/forecast/{city}" in prompt

    def test_parameter_description_prompt(self):
        """Test building parameter description prompt."""
        cache = MockCacheManager()
        engine = SchemaInferenceEngine(
            api_key="test-key",
            cache_manager=cache,
        )

        tool = Tool(
            id="weather", name="Weather API", category="Weather", endpoints=[]
        )
        endpoint = Endpoint(
            id="forecast", tool_id="weather", name="getForecast", method="GET"
        )
        parameter = Parameter(name="city", type=ParameterType.STRING, required=True)

        prompt = engine._build_description_prompt(tool, endpoint, parameter)

        assert "city" in prompt
        assert "string" in prompt.lower()
        assert "required" in prompt.lower() or "Required" in prompt

    def test_type_inference_prompt(self):
        """Test building type inference prompt."""
        cache = MockCacheManager()
        engine = SchemaInferenceEngine(
            api_key="test-key",
            cache_manager=cache,
        )

        tool = Tool(
            id="weather", name="Weather API", category="Weather", endpoints=[]
        )
        endpoint = Endpoint(
            id="forecast",
            tool_id="weather",
            name="getForecast",
            method="GET",
            description="Get weather forecast",
        )
        parameter = Parameter(name="days", type=ParameterType.UNKNOWN, default=7)

        prompt = engine._build_type_prompt(tool, endpoint, parameter)

        assert "days" in prompt
        assert "7" in prompt or "None" in prompt
        assert "string" in prompt.lower()
        assert "integer" in prompt.lower()

    def test_response_schema_prompt(self):
        """Test building response schema prompt."""
        cache = MockCacheManager()
        engine = SchemaInferenceEngine(
            api_key="test-key",
            cache_manager=cache,
        )

        tool = Tool(
            id="weather", name="Weather API", category="Weather", endpoints=[]
        )
        endpoint = Endpoint(
            id="forecast",
            tool_id="weather",
            name="getForecast",
            method="GET",
            description="Get weather forecast",
            parameters=[
                Parameter(name="city", type=ParameterType.STRING, required=True),
            ],
        )

        prompt = engine._build_response_prompt(tool, endpoint)

        assert "getForecast" in prompt
        assert "city" in prompt
        assert "JSON" in prompt


class TestTypeResponseParsing:
    """Tests for parsing LLM type responses."""

    def test_parse_string_type(self):
        """Test parsing string type."""
        cache = MockCacheManager()
        engine = SchemaInferenceEngine(api_key="test", cache_manager=cache)

        assert engine._parse_type_response("string") == ParameterType.STRING
        assert engine._parse_type_response("STRING") == ParameterType.STRING
        assert engine._parse_type_response("  string  ") == ParameterType.STRING

    def test_parse_integer_type(self):
        """Test parsing integer type."""
        cache = MockCacheManager()
        engine = SchemaInferenceEngine(api_key="test", cache_manager=cache)

        assert engine._parse_type_response("integer") == ParameterType.INTEGER
        assert engine._parse_type_response("int") == ParameterType.INTEGER

    def test_parse_number_type(self):
        """Test parsing number type."""
        cache = MockCacheManager()
        engine = SchemaInferenceEngine(api_key="test", cache_manager=cache)

        assert engine._parse_type_response("number") == ParameterType.NUMBER
        assert engine._parse_type_response("float") == ParameterType.NUMBER

    def test_parse_boolean_type(self):
        """Test parsing boolean type."""
        cache = MockCacheManager()
        engine = SchemaInferenceEngine(api_key="test", cache_manager=cache)

        assert engine._parse_type_response("boolean") == ParameterType.BOOLEAN
        assert engine._parse_type_response("bool") == ParameterType.BOOLEAN

    def test_parse_array_type(self):
        """Test parsing array type."""
        cache = MockCacheManager()
        engine = SchemaInferenceEngine(api_key="test", cache_manager=cache)

        assert engine._parse_type_response("array") == ParameterType.ARRAY
        assert engine._parse_type_response("list") == ParameterType.ARRAY

    def test_parse_object_type(self):
        """Test parsing object type."""
        cache = MockCacheManager()
        engine = SchemaInferenceEngine(api_key="test", cache_manager=cache)

        assert engine._parse_type_response("object") == ParameterType.OBJECT
        assert engine._parse_type_response("dict") == ParameterType.OBJECT

    def test_parse_unknown_returns_unknown(self):
        """Test unparseable response returns UNKNOWN."""
        cache = MockCacheManager()
        engine = SchemaInferenceEngine(api_key="test", cache_manager=cache)

        assert engine._parse_type_response("xyz") == ParameterType.UNKNOWN
        assert engine._parse_type_response("") == ParameterType.UNKNOWN


class TestResponseSchemaParsing:
    """Tests for parsing response schema."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON response."""
        cache = MockCacheManager()
        engine = SchemaInferenceEngine(api_key="test", cache_manager=cache)

        response = '{"status": "string", "data": {"id": "integer"}}'
        result = engine._parse_response_schema(response)

        assert result is not None
        assert result["status"] == "string"
        assert result["data"]["id"] == "integer"

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON returns None."""
        cache = MockCacheManager()
        engine = SchemaInferenceEngine(api_key="test", cache_manager=cache)

        result = engine._parse_response_schema("not json {{{")
        assert result is None

    def test_parse_json_with_markdown(self):
        """Test parsing JSON wrapped in markdown code blocks."""
        cache = MockCacheManager()
        engine = SchemaInferenceEngine(api_key="test", cache_manager=cache)

        response = '```json\n{"status": "string"}\n```'
        result = engine._parse_response_schema(response)

        # Should handle markdown wrapping
        assert result is not None
        assert result["status"] == "string"


class TestInferDescription:
    """Tests for description inference."""

    @patch.object(SchemaInferenceEngine, "_call_llm")
    def test_infer_tool_description(self, mock_llm):
        """Test inferring tool description."""
        mock_llm.return_value = "A weather forecasting API"

        cache = MockCacheManager()
        engine = SchemaInferenceEngine(api_key="test", cache_manager=cache)

        tool = Tool(
            id="weather",
            name="Weather API",
            category="Weather",
            description="",
            endpoints=[],
        )

        result = engine.infer_description(tool)

        assert result is not None
        assert result.field_name == "description"
        assert "weather" in result.inferred_value.lower()
        assert engine.was_inferred("weather", "description")

    @patch.object(SchemaInferenceEngine, "_call_llm")
    def test_skips_existing_description(self, mock_llm):
        """Test that existing descriptions are not overwritten."""
        cache = MockCacheManager()
        engine = SchemaInferenceEngine(api_key="test", cache_manager=cache)

        tool = Tool(
            id="weather",
            name="Weather API",
            category="Weather",
            description="Existing description",
            endpoints=[],
        )

        result = engine.infer_description(tool)

        assert result is None
        mock_llm.assert_not_called()


class TestInferParameterType:
    """Tests for parameter type inference."""

    @patch.object(SchemaInferenceEngine, "_call_llm")
    def test_infer_type(self, mock_llm):
        """Test inferring parameter type."""
        mock_llm.return_value = "integer"

        cache = MockCacheManager()
        engine = SchemaInferenceEngine(api_key="test", cache_manager=cache)

        tool = Tool(
            id="weather", name="Weather API", category="Weather", endpoints=[]
        )
        endpoint = Endpoint(
            id="forecast", tool_id="weather", name="getForecast", method="GET"
        )
        parameter = Parameter(name="days", type=ParameterType.UNKNOWN)

        result = engine.infer_parameter_type(tool, endpoint, parameter)

        assert result is not None
        assert result.inferred_value == ParameterType.INTEGER

    @patch.object(SchemaInferenceEngine, "_call_llm")
    def test_skips_known_type(self, mock_llm):
        """Test that known types are not re-inferred."""
        cache = MockCacheManager()
        engine = SchemaInferenceEngine(api_key="test", cache_manager=cache)

        tool = Tool(
            id="weather", name="Weather API", category="Weather", endpoints=[]
        )
        endpoint = Endpoint(
            id="forecast", tool_id="weather", name="getForecast", method="GET"
        )
        parameter = Parameter(name="city", type=ParameterType.STRING)

        result = engine.infer_parameter_type(tool, endpoint, parameter)

        assert result is None
        mock_llm.assert_not_called()


class TestInferResponseSchema:
    """Tests for response schema inference."""

    @patch.object(SchemaInferenceEngine, "_call_llm")
    def test_infer_response_schema(self, mock_llm):
        """Test inferring response schema."""
        mock_llm.return_value = '{"temperature": "number", "conditions": "string"}'

        cache = MockCacheManager()
        engine = SchemaInferenceEngine(api_key="test", cache_manager=cache)

        tool = Tool(
            id="weather", name="Weather API", category="Weather", endpoints=[]
        )
        endpoint = Endpoint(
            id="forecast",
            tool_id="weather",
            name="getForecast",
            method="GET",
            description="Get forecast",
        )

        result = engine.infer_response_schema(tool, endpoint)

        assert result is not None
        assert result.field_name == "response_schema"
        assert isinstance(result.inferred_value, dict)
        assert "temperature" in result.inferred_value


class TestInferTool:
    """Tests for full tool inference."""

    @patch.object(SchemaInferenceEngine, "_call_llm")
    def test_infer_tool_all_fields(self, mock_llm):
        """Test inferring all missing fields for a tool."""
        # Mock returns for each LLM call in order:
        # 1. Tool description
        # 2. Endpoint description
        # 3. city param description
        # 4. days param description (UNKNOWN type has no description)
        # 5. days param type inference
        # 6. Response schema
        mock_llm.side_effect = [
            "A weather API",  # Tool description
            "Get weather forecast",  # Endpoint description
            "City name for forecast",  # city Parameter description
            "Number of forecast days",  # days Parameter description
            "integer",  # days type
            '{"temp": "number"}',  # Response schema
        ]

        cache = MockCacheManager()
        engine = SchemaInferenceEngine(api_key="test", cache_manager=cache)

        tool = Tool(
            id="weather",
            name="Weather API",
            category="Weather",
            description="",
            endpoints=[
                Endpoint(
                    id="forecast",
                    tool_id="weather",
                    name="getForecast",
                    method="GET",
                    description="",
                    parameters=[
                        Parameter(
                            name="city", type=ParameterType.STRING, description=""
                        ),
                        Parameter(name="days", type=ParameterType.UNKNOWN),
                    ],
                )
            ],
        )

        results = engine.infer_tool(tool)

        assert len(results) >= 1
        assert any(r.field_name == "description" for r in results)


class TestInferRegistry:
    """Tests for registry-wide inference."""

    @patch.object(SchemaInferenceEngine, "_call_llm")
    def test_infer_registry(self, mock_llm):
        """Test inferring fields for entire registry."""
        mock_llm.return_value = "Inferred description"

        cache = MockCacheManager()
        engine = SchemaInferenceEngine(
            api_key="test",
            cache_manager=cache,
            completeness_threshold=0.9,
        )

        tool = Tool(
            id="incomplete",
            name="Test Tool",
            category="Test",
            description="",
            endpoints=[
                Endpoint(
                    id="ep1",
                    tool_id="incomplete",
                    name="test",
                    method="GET",
                    description="",
                )
            ],
        )

        registry = ToolRegistry()
        registry.add_tool(tool)

        stats = engine.infer_registry(registry)

        assert stats.tools_processed >= 1
        assert stats.llm_calls >= 0


class TestInferenceStats:
    """Tests for inference statistics."""

    def test_stats_default_values(self):
        """Test InferenceStats default values."""
        stats = InferenceStats()

        assert stats.tools_processed == 0
        assert stats.tools_updated == 0
        assert stats.descriptions_inferred == 0
        assert stats.types_inferred == 0
        assert stats.response_schemas_inferred == 0
        assert stats.cache_hits == 0
        assert stats.llm_calls == 0
        assert stats.errors == []


class TestInferenceResult:
    """Tests for InferenceResult dataclass."""

    def test_result_creation(self):
        """Test creating InferenceResult."""
        result = InferenceResult(
            tool_id="weather",
            endpoint_id="forecast",
            field_name="description",
            original_value="",
            inferred_value="Weather forecast endpoint",
            confidence=0.95,
            cached=False,
        )

        assert result.tool_id == "weather"
        assert result.endpoint_id == "forecast"
        assert result.field_name == "description"
        assert result.confidence == 0.95
        assert result.cached is False


class TestCompletenessCalculation:
    """Tests for tool completeness calculation."""

    def test_complete_tool_scores_high(self):
        """Test that a complete tool has high completeness score."""
        cache = MockCacheManager()
        engine = SchemaInferenceEngine(api_key="test", cache_manager=cache)

        tool = Tool(
            id="complete",
            name="Complete Tool",
            category="Test",
            description="A complete tool with everything filled in",
            endpoints=[
                Endpoint(
                    id="ep1",
                    tool_id="complete",
                    name="endpoint1",
                    method="GET",
                    description="A complete endpoint",
                    parameters=[
                        Parameter(
                            name="param1",
                            type=ParameterType.STRING,
                            description="A complete parameter",
                            required=True,
                        )
                    ],
                )
            ],
        )

        score = engine._calculate_tool_completeness(tool)
        assert score >= 0.9

    def test_incomplete_tool_scores_low(self):
        """Test that an incomplete tool has low completeness score."""
        cache = MockCacheManager()
        engine = SchemaInferenceEngine(api_key="test", cache_manager=cache)

        tool = Tool(
            id="incomplete",
            name="Incomplete Tool",
            category="Test",
            description="",  # Missing
            endpoints=[
                Endpoint(
                    id="ep1",
                    tool_id="incomplete",
                    name="endpoint1",
                    method="GET",
                    description="",  # Missing
                    parameters=[
                        Parameter(
                            name="param1",
                            type=ParameterType.UNKNOWN,  # Unknown
                            description="",  # Missing
                        )
                    ],
                )
            ],
        )

        score = engine._calculate_tool_completeness(tool)
        assert score < 0.5
