"""Unit tests for data models."""

import pytest

from src.models import (
    Endpoint,
    Parameter,
    ParameterType,
    Tool,
    ToolRegistry,
)


class TestParameterType:
    """Tests for ParameterType enum."""

    def test_enum_values(self):
        """Test all enum values exist."""
        assert ParameterType.STRING.value == "string"
        assert ParameterType.INTEGER.value == "integer"
        assert ParameterType.NUMBER.value == "number"
        assert ParameterType.BOOLEAN.value == "boolean"
        assert ParameterType.ARRAY.value == "array"
        assert ParameterType.OBJECT.value == "object"
        assert ParameterType.UNKNOWN.value == "unknown"

    def test_from_string_exact(self):
        """Test from_string with exact matches."""
        assert ParameterType.from_string("string") == ParameterType.STRING
        assert ParameterType.from_string("integer") == ParameterType.INTEGER
        assert ParameterType.from_string("boolean") == ParameterType.BOOLEAN

    def test_from_string_aliases(self):
        """Test from_string with aliases."""
        assert ParameterType.from_string("str") == ParameterType.STRING
        assert ParameterType.from_string("int") == ParameterType.INTEGER
        assert ParameterType.from_string("float") == ParameterType.NUMBER
        assert ParameterType.from_string("bool") == ParameterType.BOOLEAN
        assert ParameterType.from_string("list") == ParameterType.ARRAY
        assert ParameterType.from_string("dict") == ParameterType.OBJECT

    def test_from_string_case_insensitive(self):
        """Test from_string is case insensitive."""
        assert ParameterType.from_string("STRING") == ParameterType.STRING
        assert ParameterType.from_string("String") == ParameterType.STRING

    def test_from_string_unknown(self):
        """Test from_string returns UNKNOWN for unrecognized types."""
        assert ParameterType.from_string("foo") == ParameterType.UNKNOWN
        assert ParameterType.from_string(None) == ParameterType.UNKNOWN
        assert ParameterType.from_string("") == ParameterType.UNKNOWN


class TestParameter:
    """Tests for Parameter model."""

    def test_minimal_parameter(self):
        """Test parameter with just name."""
        param = Parameter(name="query")
        assert param.name == "query"
        assert param.type == ParameterType.UNKNOWN
        assert param.description == ""
        assert param.required == False
        assert param.default is None
        assert param.enum is None

    def test_full_parameter(self):
        """Test parameter with all fields."""
        param = Parameter(
            name="limit",
            type=ParameterType.INTEGER,
            description="Maximum results",
            required=True,
            default=10,
            enum=[10, 25, 50, 100],
        )
        assert param.name == "limit"
        assert param.type == ParameterType.INTEGER
        assert param.description == "Maximum results"
        assert param.required == True
        assert param.default == 10
        assert param.enum == [10, 25, 50, 100]

    def test_empty_name_rejected(self):
        """Test empty name is rejected."""
        with pytest.raises(ValueError):
            Parameter(name="")

    def test_whitespace_name_rejected(self):
        """Test whitespace-only name is rejected."""
        with pytest.raises(ValueError):
            Parameter(name="   ")

    def test_serialization(self):
        """Test parameter serializes to JSON."""
        param = Parameter(name="test", type=ParameterType.STRING)
        data = param.model_dump()
        assert data["name"] == "test"
        assert data["type"] == "string"


class TestEndpoint:
    """Tests for Endpoint model."""

    def test_minimal_endpoint(self):
        """Test endpoint with required fields only."""
        endpoint = Endpoint(
            id="tool1_get_data",
            tool_id="tool1",
            name="get_data",
        )
        assert endpoint.id == "tool1_get_data"
        assert endpoint.tool_id == "tool1"
        assert endpoint.name == "get_data"
        assert endpoint.method == "GET"
        assert endpoint.parameters == []
        # Minimal endpoint: no description, no path, no schema = low score
        # Only gets param_types (0.3) + name_quality (0.0 - too short)
        assert 0.0 <= endpoint.completeness_score <= 0.5

    def test_method_normalized(self):
        """Test HTTP method is normalized to uppercase."""
        endpoint = Endpoint(
            id="test",
            tool_id="test",
            name="test",
            method="post",
        )
        assert endpoint.method == "POST"

    def test_method_mixed_case(self):
        """Test HTTP method normalization with mixed case."""
        endpoint = Endpoint(
            id="test",
            tool_id="test",
            name="test",
            method="PaTcH",
        )
        assert endpoint.method == "PATCH"

    def test_required_parameters(self):
        """Test required_parameters computed field."""
        endpoint = Endpoint(
            id="test",
            tool_id="test",
            name="test",
            parameters=[
                Parameter(name="required_param", required=True),
                Parameter(name="optional_param", required=False),
            ],
        )
        required = endpoint.required_parameters
        assert len(required) == 1
        assert required[0].name == "required_param"

    def test_optional_parameters(self):
        """Test optional_parameters computed field."""
        endpoint = Endpoint(
            id="test",
            tool_id="test",
            name="test",
            parameters=[
                Parameter(name="required_param", required=True),
                Parameter(name="optional_param", required=False),
            ],
        )
        optional = endpoint.optional_parameters
        assert len(optional) == 1
        assert optional[0].name == "optional_param"

    def test_completeness_score_is_computed(self):
        """Test completeness_score is a computed property based on content."""
        # Minimal endpoint should have low score
        endpoint_min = Endpoint(id="t", tool_id="t", name="t")
        assert 0.0 <= endpoint_min.completeness_score <= 1.0

        # Complete endpoint should have high score
        endpoint_full = Endpoint(
            id="t",
            tool_id="t",
            name="getUserProfile",
            path="/api/users/{id}",
            description="Retrieves user profile by ID",
            response_schema={"user": "object"},
        )
        assert endpoint_full.completeness_score >= 0.9

    def test_completeness_score_valid_range(self):
        """Test completeness_score is always in valid 0-1 range."""
        # Minimal endpoint
        endpoint_min = Endpoint(id="t", tool_id="t", name="t")
        assert 0.0 <= endpoint_min.completeness_score <= 1.0

        # Full endpoint
        endpoint_max = Endpoint(
            id="t",
            tool_id="t",
            name="getUserProfile",
            path="/api/users",
            description="Gets user profile information",
            response_schema={"data": "object"},
        )
        assert 0.0 <= endpoint_max.completeness_score <= 1.0

    def test_full_endpoint(self):
        """Test endpoint with all fields."""
        endpoint = Endpoint(
            id="weather_get_forecast",
            tool_id="weather_api",
            name="get_forecast",
            method="GET",
            path="/forecast/{city}",
            description="Get weather forecast for a city",
            parameters=[
                Parameter(name="city", type=ParameterType.STRING, required=True),
                Parameter(name="days", type=ParameterType.INTEGER, default=7),
            ],
            response_schema={"type": "object"},
            domain="weather",
            raw_schema={"operationId": "getForecast"},
            inferred_fields=["response_schema"],
        )
        assert endpoint.domain == "weather"
        # Full endpoint with all fields should score high (computed property)
        assert endpoint.completeness_score >= 0.9
        assert endpoint.inferred_fields == ["response_schema"]


class TestTool:
    """Tests for Tool model."""

    def test_minimal_tool(self):
        """Test tool with required fields only."""
        tool = Tool(id="weather_api", name="Weather API")
        assert tool.id == "weather_api"
        assert tool.name == "Weather API"
        assert tool.category == "Uncategorized"
        assert tool.endpoints == []
        assert tool.endpoint_count == 0

    def test_empty_id_rejected(self):
        """Test empty id is rejected."""
        with pytest.raises(ValueError):
            Tool(id="", name="Test")

    def test_empty_name_rejected(self):
        """Test empty name is rejected."""
        with pytest.raises(ValueError):
            Tool(id="test", name="")

    def test_whitespace_id_rejected(self):
        """Test whitespace-only id is rejected."""
        with pytest.raises(ValueError):
            Tool(id="   ", name="Test")

    def test_endpoint_count(self):
        """Test endpoint_count computed field."""
        tool = Tool(
            id="test",
            name="Test",
            endpoints=[
                Endpoint(id="e1", tool_id="test", name="ep1"),
                Endpoint(id="e2", tool_id="test", name="ep2"),
            ],
        )
        assert tool.endpoint_count == 2

    def test_get_endpoint(self):
        """Test get_endpoint method."""
        endpoint = Endpoint(id="e1", tool_id="test", name="ep1")
        tool = Tool(id="test", name="Test", endpoints=[endpoint])

        assert tool.get_endpoint("e1") == endpoint
        assert tool.get_endpoint("nonexistent") is None

    def test_full_tool(self):
        """Test tool with all fields."""
        tool = Tool(
            id="weather_api",
            name="Weather API",
            category="Data",
            description="Provides weather information",
            api_host="https://api.weather.com",
            endpoints=[
                Endpoint(
                    id="e1",
                    tool_id="weather_api",
                    name="get_current",
                    path="/current",
                    description="Get current weather conditions",
                ),
            ],
            raw_schema={"openapi": "3.0.0"},
        )
        assert tool.category == "Data"
        assert tool.api_host == "https://api.weather.com"
        # Completeness is computed based on content
        assert 0.0 <= tool.completeness_score <= 1.0


class TestToolRegistry:
    """Tests for ToolRegistry model."""

    def test_empty_registry(self):
        """Test empty registry."""
        registry = ToolRegistry()
        assert registry.tool_count == 0
        assert registry.endpoint_count == 0
        assert registry.domains == []

    def test_add_tool(self):
        """Test adding a tool."""
        registry = ToolRegistry()
        tool = Tool(
            id="weather",
            name="Weather API",
            endpoints=[
                Endpoint(id="w1", tool_id="weather", name="get_weather", domain="weather"),
            ],
        )

        registry.add_tool(tool)

        assert registry.tool_count == 1
        assert registry.endpoint_count == 1
        assert "weather" in registry.domains

    def test_add_duplicate_tool_raises(self):
        """Test adding duplicate tool raises ValueError."""
        registry = ToolRegistry()
        tool = Tool(id="test", name="Test")

        registry.add_tool(tool)

        with pytest.raises(ValueError):
            registry.add_tool(tool)

    def test_get_tool(self):
        """Test getting a tool by ID."""
        registry = ToolRegistry()
        tool = Tool(id="test", name="Test")
        registry.add_tool(tool)

        assert registry.get_tool("test") == tool
        assert registry.get_tool("nonexistent") is None

    def test_get_endpoint(self):
        """Test getting an endpoint by ID."""
        registry = ToolRegistry()
        endpoint = Endpoint(id="e1", tool_id="test", name="ep1")
        tool = Tool(id="test", name="Test", endpoints=[endpoint])
        registry.add_tool(tool)

        assert registry.get_endpoint("e1") == endpoint
        assert registry.get_endpoint("nonexistent") is None

    def test_get_endpoints_by_domain(self):
        """Test filtering endpoints by domain."""
        registry = ToolRegistry()

        tool1 = Tool(
            id="weather",
            name="Weather",
            endpoints=[
                Endpoint(id="w1", tool_id="weather", name="forecast", domain="weather"),
                Endpoint(id="w2", tool_id="weather", name="current", domain="weather"),
            ],
        )
        tool2 = Tool(
            id="finance",
            name="Finance",
            endpoints=[
                Endpoint(id="f1", tool_id="finance", name="stocks", domain="finance"),
            ],
        )

        registry.add_tool(tool1)
        registry.add_tool(tool2)

        weather_endpoints = registry.get_endpoints_by_domain("weather")
        assert len(weather_endpoints) == 2

        finance_endpoints = registry.get_endpoints_by_domain("finance")
        assert len(finance_endpoints) == 1

        empty = registry.get_endpoints_by_domain("nonexistent")
        assert len(empty) == 0

    def test_domains_sorted(self):
        """Test domains property returns sorted list."""
        registry = ToolRegistry()

        tool = Tool(
            id="test",
            name="Test",
            endpoints=[
                Endpoint(id="e1", tool_id="test", name="ep1", domain="zebra"),
                Endpoint(id="e2", tool_id="test", name="ep2", domain="alpha"),
                Endpoint(id="e3", tool_id="test", name="ep3", domain="middle"),
            ],
        )
        registry.add_tool(tool)

        assert registry.domains == ["alpha", "middle", "zebra"]

    def test_registry_with_initial_tools(self):
        """Test creating registry with tools in constructor."""
        tool = Tool(
            id="test",
            name="Test",
            endpoints=[
                Endpoint(id="e1", tool_id="test", name="ep1", domain="weather"),
            ],
        )
        registry = ToolRegistry(tools={"test": tool})

        assert registry.tool_count == 1
        assert registry.endpoint_count == 1
        assert registry.get_endpoint("e1") is not None
        assert "weather" in registry.domains

    def test_multiple_tools_same_domain(self):
        """Test multiple tools contributing to same domain."""
        registry = ToolRegistry()

        tool1 = Tool(
            id="weather1",
            name="Weather 1",
            endpoints=[
                Endpoint(id="w1", tool_id="weather1", name="ep1", domain="weather"),
            ],
        )
        tool2 = Tool(
            id="weather2",
            name="Weather 2",
            endpoints=[
                Endpoint(id="w2", tool_id="weather2", name="ep2", domain="weather"),
            ],
        )

        registry.add_tool(tool1)
        registry.add_tool(tool2)

        weather_endpoints = registry.get_endpoints_by_domain("weather")
        assert len(weather_endpoints) == 2
        assert registry.domains == ["weather"]
