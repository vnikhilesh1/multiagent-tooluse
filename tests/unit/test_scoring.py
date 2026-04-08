"""Unit tests for completeness scoring."""

import pytest

from src.models import (
    Endpoint,
    Parameter,
    ParameterType,
    Tool,
)
from src.models.scoring import (
    GENERIC_NAMES,
    get_endpoint_score_breakdown,
    get_tool_score_breakdown,
    score_description,
    score_endpoint,
    score_name_quality,
    score_parameter_types,
    score_path_validity,
    score_response_schema,
    score_tool,
)


class TestScoreDescription:
    """Tests for description scoring."""

    def test_good_description(self):
        """Test that a good description scores 1.0."""
        assert score_description("This is a valid description") == 1.0

    def test_empty_description(self):
        """Test that empty description scores 0.0."""
        assert score_description("") == 0.0
        assert score_description(None) == 0.0

    def test_short_description(self):
        """Test that short description scores 0.0."""
        assert score_description("Short") == 0.0
        assert score_description("abc") == 0.0

    def test_placeholder_description(self):
        """Test that placeholder text scores 0.0."""
        assert score_description("TODO") == 0.0
        assert score_description("TBD") == 0.0
        assert score_description("N/A") == 0.0
        assert score_description("description") == 0.0

    def test_whitespace_only(self):
        """Test that whitespace-only scores 0.0."""
        assert score_description("   ") == 0.0
        assert score_description("\n\t") == 0.0

    def test_custom_min_length(self):
        """Test custom minimum length."""
        assert score_description("Short", min_length=3) == 1.0
        assert score_description("Hi", min_length=3) == 0.0


class TestScoreParameterTypes:
    """Tests for parameter type scoring."""

    def test_all_known_types(self):
        """Test that all known types score 1.0."""
        params = [
            Parameter(name="a", type=ParameterType.STRING),
            Parameter(name="b", type=ParameterType.INTEGER),
            Parameter(name="c", type=ParameterType.BOOLEAN),
        ]
        assert score_parameter_types(params) == 1.0

    def test_all_unknown_types(self):
        """Test that all unknown types score 0.0."""
        params = [
            Parameter(name="a", type=ParameterType.UNKNOWN),
            Parameter(name="b", type=ParameterType.UNKNOWN),
        ]
        assert score_parameter_types(params) == 0.0

    def test_mixed_types(self):
        """Test mixed known/unknown types."""
        params = [
            Parameter(name="a", type=ParameterType.STRING),
            Parameter(name="b", type=ParameterType.UNKNOWN),
        ]
        assert score_parameter_types(params) == 0.5

    def test_empty_parameters(self):
        """Test that empty parameters score 1.0."""
        assert score_parameter_types([]) == 1.0

    def test_partial_known(self):
        """Test partial known types."""
        params = [
            Parameter(name="a", type=ParameterType.STRING),
            Parameter(name="b", type=ParameterType.INTEGER),
            Parameter(name="c", type=ParameterType.UNKNOWN),
            Parameter(name="d", type=ParameterType.UNKNOWN),
        ]
        assert score_parameter_types(params) == 0.5


class TestScoreResponseSchema:
    """Tests for response schema scoring."""

    def test_has_schema(self):
        """Test that having schema scores 1.0."""
        endpoint = Endpoint(
            id="test",
            tool_id="tool",
            name="test",
            method="GET",
            response_schema={"status": "string"},
        )
        assert score_response_schema(endpoint) == 1.0

    def test_no_schema(self):
        """Test that no schema scores 0.0."""
        endpoint = Endpoint(
            id="test",
            tool_id="tool",
            name="test",
            method="GET",
        )
        assert score_response_schema(endpoint) == 0.0

    def test_empty_schema(self):
        """Test that empty schema scores 1.0 (still defined)."""
        endpoint = Endpoint(
            id="test",
            tool_id="tool",
            name="test",
            method="GET",
            response_schema={},
        )
        assert score_response_schema(endpoint) == 1.0


class TestScoreNameQuality:
    """Tests for name quality scoring."""

    def test_good_name(self):
        """Test that descriptive names score 1.0."""
        assert score_name_quality("getUserProfile") == 1.0
        assert score_name_quality("weather_forecast") == 1.0
        assert score_name_quality("createOrder") == 1.0

    def test_generic_name(self):
        """Test that generic names score 0.0."""
        assert score_name_quality("api") == 0.0
        assert score_name_quality("endpoint") == 0.0
        assert score_name_quality("test") == 0.0
        assert score_name_quality("data") == 0.0

    def test_short_name(self):
        """Test that very short names score 0.0."""
        assert score_name_quality("a") == 0.0
        assert score_name_quality("ab") == 0.0
        assert score_name_quality("abc") == 0.0

    def test_empty_name(self):
        """Test that empty name scores 0.0."""
        assert score_name_quality("") == 0.0

    def test_camel_case_generic(self):
        """Test camelCase with all generic words."""
        assert score_name_quality("getData") == 0.0
        assert score_name_quality("testApi") == 0.0

    def test_snake_case_generic(self):
        """Test snake_case with all generic words."""
        assert score_name_quality("get_data") == 0.0
        assert score_name_quality("test_api") == 0.0

    def test_mixed_meaningful(self):
        """Test names with some meaningful content."""
        assert score_name_quality("getWeather") == 1.0
        assert score_name_quality("user_profile") == 1.0


class TestScorePathValidity:
    """Tests for path validity scoring."""

    def test_valid_absolute_path(self):
        """Test that absolute paths score 1.0."""
        assert score_path_validity("/api/users") == 1.0
        assert score_path_validity("/v1/weather/forecast") == 1.0

    def test_valid_url(self):
        """Test that URLs score 1.0."""
        assert score_path_validity("https://api.example.com/users") == 1.0
        assert score_path_validity("http://localhost/api") == 1.0

    def test_relative_path_with_slash(self):
        """Test that relative paths with slash score 1.0."""
        assert score_path_validity("api/v1/users") == 1.0

    def test_empty_path(self):
        """Test that empty path scores 0.0."""
        assert score_path_validity("") == 0.0
        assert score_path_validity(None) == 0.0
        assert score_path_validity("   ") == 0.0

    def test_no_path_structure(self):
        """Test that non-path strings score 0.0."""
        assert score_path_validity("users") == 0.0
        assert score_path_validity("getUsers") == 0.0


class TestScoreEndpoint:
    """Tests for endpoint scoring."""

    def test_complete_endpoint(self):
        """Test that a complete endpoint scores high."""
        endpoint = Endpoint(
            id="test",
            tool_id="tool",
            name="getUserProfile",
            method="GET",
            path="/api/users/{id}",
            description="Retrieves user profile information by ID",
            parameters=[
                Parameter(name="id", type=ParameterType.STRING, required=True),
            ],
            response_schema={"user": {"id": "string", "name": "string"}},
        )
        score = score_endpoint(endpoint)
        assert score >= 0.9  # Should be near perfect

    def test_minimal_endpoint(self):
        """Test that a minimal endpoint scores low."""
        endpoint = Endpoint(
            id="test",
            tool_id="tool",
            name="api",
            method="GET",
        )
        score = score_endpoint(endpoint)
        assert score <= 0.4  # Should be quite low

    def test_partial_endpoint(self):
        """Test a partially complete endpoint."""
        endpoint = Endpoint(
            id="test",
            tool_id="tool",
            name="getWeather",
            method="GET",
            path="/weather",
            description="Gets weather",  # Short but valid (11 chars)
        )
        score = score_endpoint(endpoint)
        # Has: description(0.2) + params(0.3, empty=1.0) + name(0.1) + path(0.2) = 0.8
        # Missing: response_schema(0.2)
        assert 0.7 <= score <= 0.9

    def test_score_is_bounded(self):
        """Test that score is between 0 and 1."""
        endpoint = Endpoint(
            id="test",
            tool_id="tool",
            name="test",
            method="GET",
        )
        score = score_endpoint(endpoint)
        assert 0.0 <= score <= 1.0


class TestScoreTool:
    """Tests for tool scoring."""

    def test_complete_tool(self):
        """Test that a complete tool scores high."""
        tool = Tool(
            id="weather_api",
            name="Weather API Service",
            category="Weather",
            description="A comprehensive weather forecasting API",
            endpoints=[
                Endpoint(
                    id="forecast",
                    tool_id="weather_api",
                    name="getForecast",
                    method="GET",
                    path="/forecast/{city}",
                    description="Get weather forecast for a city",
                    parameters=[
                        Parameter(name="city", type=ParameterType.STRING),
                    ],
                    response_schema={"temp": "number"},
                ),
            ],
        )
        score = score_tool(tool)
        assert score >= 0.8

    def test_minimal_tool(self):
        """Test that a minimal tool scores low."""
        tool = Tool(
            id="api",
            name="API",
            category="Test",
        )
        score = score_tool(tool)
        assert score <= 0.2

    def test_tool_without_endpoints(self):
        """Test tool without endpoints."""
        tool = Tool(
            id="empty",
            name="Empty Tool",
            category="Test",
            description="A tool with no endpoints defined",
        )
        score = score_tool(tool)
        # Should have description (+0.2) and name (+0.1) but no endpoints
        assert 0.2 <= score <= 0.4

    def test_tool_with_incomplete_endpoints(self):
        """Test tool with incomplete endpoints."""
        tool = Tool(
            id="partial",
            name="Partial Tool",
            category="Test",
            description="Tool with incomplete endpoints",
            endpoints=[
                Endpoint(
                    id="ep1",
                    tool_id="partial",
                    name="api",  # Generic name
                    method="GET",
                    # No path, description, schema
                ),
            ],
        )
        score = score_tool(tool)
        # Has: description(0.2) + name(0.1) + has_endpoints(0.2) + endpoints_avg(0.5 * 0.3) = 0.65
        assert 0.5 <= score <= 0.7


class TestGetEndpointScoreBreakdown:
    """Tests for endpoint score breakdown."""

    def test_breakdown_structure(self):
        """Test that breakdown has expected structure."""
        endpoint = Endpoint(
            id="test",
            tool_id="tool",
            name="getWeather",
            method="GET",
            path="/weather",
            description="Get current weather conditions",
        )
        breakdown = get_endpoint_score_breakdown(endpoint)

        assert "description" in breakdown
        assert "parameter_types" in breakdown
        assert "response_schema" in breakdown
        assert "name_quality" in breakdown
        assert "path_validity" in breakdown
        assert "total" in breakdown

        # Check each component has expected fields
        for key in [
            "description",
            "parameter_types",
            "response_schema",
            "name_quality",
            "path_validity",
        ]:
            assert "score" in breakdown[key]
            assert "weight" in breakdown[key]
            assert "weighted" in breakdown[key]

    def test_breakdown_weights_sum_to_one(self):
        """Test that weights sum to 1.0."""
        endpoint = Endpoint(
            id="test",
            tool_id="tool",
            name="test",
            method="GET",
        )
        breakdown = get_endpoint_score_breakdown(endpoint)

        total_weight = sum(
            breakdown[k]["weight"]
            for k in [
                "description",
                "parameter_types",
                "response_schema",
                "name_quality",
                "path_validity",
            ]
        )
        assert abs(total_weight - 1.0) < 0.001

    def test_breakdown_total_matches_score(self):
        """Test that breakdown total matches score_endpoint."""
        endpoint = Endpoint(
            id="test",
            tool_id="tool",
            name="getWeather",
            method="GET",
            path="/weather",
            description="Get current weather conditions",
            parameters=[
                Parameter(name="city", type=ParameterType.STRING),
            ],
        )
        breakdown = get_endpoint_score_breakdown(endpoint)
        direct_score = score_endpoint(endpoint)

        assert abs(breakdown["total"] - direct_score) < 0.01


class TestGetToolScoreBreakdown:
    """Tests for tool score breakdown."""

    def test_breakdown_structure(self):
        """Test that breakdown has expected structure."""
        tool = Tool(
            id="test",
            name="Test Tool",
            category="Test",
            description="A test tool for testing",
            endpoints=[
                Endpoint(
                    id="ep1",
                    tool_id="test",
                    name="endpoint1",
                    method="GET",
                ),
            ],
        )
        breakdown = get_tool_score_breakdown(tool)

        assert "description" in breakdown
        assert "name_quality" in breakdown
        assert "has_endpoints" in breakdown
        assert "endpoints_completeness" in breakdown
        assert "total" in breakdown

    def test_breakdown_total_matches_score(self):
        """Test that breakdown total matches score_tool."""
        tool = Tool(
            id="weather",
            name="Weather Service",
            category="Weather",
            description="A weather forecasting service",
            endpoints=[
                Endpoint(
                    id="forecast",
                    tool_id="weather",
                    name="getForecast",
                    method="GET",
                    path="/forecast",
                ),
            ],
        )
        breakdown = get_tool_score_breakdown(tool)
        direct_score = score_tool(tool)

        assert abs(breakdown["total"] - direct_score) < 0.01


class TestGenericNames:
    """Tests for GENERIC_NAMES constant."""

    def test_common_generics_included(self):
        """Test that common generic names are in the set."""
        assert "api" in GENERIC_NAMES
        assert "endpoint" in GENERIC_NAMES
        assert "test" in GENERIC_NAMES
        assert "data" in GENERIC_NAMES
        assert "get" in GENERIC_NAMES
        assert "post" in GENERIC_NAMES

    def test_is_set(self):
        """Test that GENERIC_NAMES is a set for O(1) lookup."""
        assert isinstance(GENERIC_NAMES, set)


class TestIntegrationWithModels:
    """Tests for integration with model classes."""

    def test_endpoint_has_completeness_score(self):
        """Test that Endpoint model exposes completeness_score."""
        endpoint = Endpoint(
            id="test",
            tool_id="tool",
            name="getWeather",
            method="GET",
            path="/weather",
            description="Get weather data",
        )
        # Should be accessible as property
        assert hasattr(endpoint, "completeness_score")
        assert 0.0 <= endpoint.completeness_score <= 1.0

    def test_tool_has_completeness_score(self):
        """Test that Tool model exposes completeness_score."""
        tool = Tool(
            id="weather",
            name="Weather API",
            category="Weather",
            description="Weather forecasting API",
            endpoints=[],
        )
        assert hasattr(tool, "completeness_score")
        assert 0.0 <= tool.completeness_score <= 1.0

    def test_endpoint_score_matches_function(self):
        """Test that Endpoint.completeness_score matches score_endpoint()."""
        endpoint = Endpoint(
            id="test",
            tool_id="tool",
            name="getWeather",
            method="GET",
            path="/weather",
            description="Get weather data for a location",
        )
        assert endpoint.completeness_score == score_endpoint(endpoint)

    def test_tool_score_matches_function(self):
        """Test that Tool.completeness_score matches score_tool()."""
        tool = Tool(
            id="weather",
            name="Weather API",
            category="Weather",
            description="Weather forecasting API service",
            endpoints=[
                Endpoint(
                    id="forecast",
                    tool_id="weather",
                    name="getForecast",
                    method="GET",
                    path="/forecast",
                ),
            ],
        )
        assert tool.completeness_score == score_tool(tool)
