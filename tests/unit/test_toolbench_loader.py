"""Unit tests for ToolBench loader."""

import json
import tempfile
from pathlib import Path

import pytest

from src.loaders import ToolBenchLoader
from src.models import ParameterType, ToolRegistry


class TestToolBenchLoaderInit:
    """Tests for ToolBenchLoader initialization."""

    def test_default_init(self):
        """Test default initialization."""
        loader = ToolBenchLoader()
        assert loader._max_description_length == 500
        assert loader._default_method == "GET"
        assert loader._skip_malformed == True

    def test_custom_init(self):
        """Test custom initialization."""
        loader = ToolBenchLoader(
            max_description_length=200,
            default_method="POST",
            skip_malformed=False,
        )
        assert loader._max_description_length == 200
        assert loader._default_method == "POST"
        assert loader._skip_malformed == False


class TestApiListFormat:
    """Tests for api_list format parsing."""

    def test_parse_api_list_format(self):
        """Test parsing api_list format."""
        loader = ToolBenchLoader()
        data = {
            "tool_name": "Weather API",
            "category_name": "Weather",
            "api_list": [
                {
                    "name": "get_forecast",
                    "url": "https://api.weather.com/forecast",
                    "method": "GET",
                    "description": "Get weather forecast",
                    "required_parameters": [
                        {"name": "city", "type": "string", "description": "City name"}
                    ],
                    "optional_parameters": [
                        {
                            "name": "days",
                            "type": "integer",
                            "description": "Forecast days",
                            "default": 7,
                        }
                    ],
                }
            ],
        }

        tool = loader._parse_tool(data)

        assert tool is not None
        assert tool.name == "Weather API"
        assert tool.category == "Weather"
        assert len(tool.endpoints) == 1

        endpoint = tool.endpoints[0]
        assert endpoint.name == "get_forecast"
        assert endpoint.method == "GET"
        assert len(endpoint.parameters) == 2

        # Check required parameter
        city_param = next(p for p in endpoint.parameters if p.name == "city")
        assert city_param.required == True
        assert city_param.type == ParameterType.STRING

        # Check optional parameter
        days_param = next(p for p in endpoint.parameters if p.name == "days")
        assert days_param.required == False
        assert days_param.default == 7

    def test_api_list_with_tool_id(self):
        """Test api_list with explicit tool_id."""
        loader = ToolBenchLoader()
        data = {"tool_id": "custom_weather_id", "tool_name": "Weather API", "api_list": []}

        tool = loader._parse_tool(data)
        assert tool.id == "custom_weather_id"

    def test_api_list_empty(self):
        """Test api_list with empty array."""
        loader = ToolBenchLoader()
        data = {"tool_name": "Empty API", "api_list": []}

        tool = loader._parse_tool(data)
        assert tool is not None
        assert tool.name == "Empty API"
        assert len(tool.endpoints) == 0


class TestEndpointsFormat:
    """Tests for endpoints format parsing."""

    def test_parse_endpoints_format(self):
        """Test parsing endpoints format."""
        loader = ToolBenchLoader()
        data = {
            "id": "weather_api",
            "name": "Weather API",
            "category": "Weather",
            "endpoints": [
                {
                    "name": "getForecast",
                    "path": "/forecast/{city}",
                    "method": "GET",
                    "description": "Get forecast",
                    "parameters": {
                        "city": {"type": "string", "required": True},
                        "days": {"type": "int", "required": False, "default": 7},
                    },
                }
            ],
        }

        tool = loader._parse_tool(data)

        assert tool is not None
        assert tool.id == "weather_api"
        assert tool.name == "Weather API"
        assert len(tool.endpoints) == 1

        endpoint = tool.endpoints[0]
        assert endpoint.name == "getForecast"
        assert endpoint.path == "/forecast/{city}"

        # Check parameters parsed from dict
        assert len(endpoint.parameters) == 2
        city_param = next(p for p in endpoint.parameters if p.name == "city")
        assert city_param.required == True

    def test_endpoints_with_list_parameters(self):
        """Test endpoints format with list-style parameters."""
        loader = ToolBenchLoader()
        data = {
            "name": "Test API",
            "endpoints": [
                {
                    "name": "test",
                    "parameters": [
                        {"name": "param1", "type": "string", "required": True}
                    ],
                }
            ],
        }

        tool = loader._parse_tool(data)
        assert len(tool.endpoints[0].parameters) == 1


class TestApisFormat:
    """Tests for apis format parsing."""

    def test_parse_apis_format(self):
        """Test parsing apis format."""
        loader = ToolBenchLoader()
        data = {
            "tool_name": "Weather",
            "apis": [
                {
                    "api_name": "forecast",
                    "api_description": "Get forecast",
                    "method": "get",
                    "parameters": [
                        {
                            "parameter_name": "city",
                            "parameter_type": "string",
                            "required": True,
                        }
                    ],
                }
            ],
        }

        tool = loader._parse_tool(data)

        assert tool is not None
        assert tool.name == "Weather"
        assert len(tool.endpoints) == 1

        endpoint = tool.endpoints[0]
        assert endpoint.name == "forecast"
        assert endpoint.method == "GET"  # Normalized to uppercase

    def test_apis_with_dict_parameters(self):
        """Test apis format with dict-style parameters."""
        loader = ToolBenchLoader()
        data = {
            "tool_name": "Test",
            "apis": [
                {
                    "api_name": "test",
                    "parameters": {"param1": {"type": "string", "required": True}},
                }
            ],
        }

        tool = loader._parse_tool(data)
        assert len(tool.endpoints[0].parameters) == 1


class TestMethodNormalization:
    """Tests for HTTP method normalization."""

    def test_lowercase_method_normalized(self):
        """Test lowercase method is uppercased."""
        loader = ToolBenchLoader()
        assert loader._normalize_method("get") == "GET"
        assert loader._normalize_method("post") == "POST"
        assert loader._normalize_method("Put") == "PUT"

    def test_none_method_uses_default(self):
        """Test None method uses default."""
        loader = ToolBenchLoader(default_method="POST")
        assert loader._normalize_method(None) == "POST"
        assert loader._normalize_method("") == "POST"

    def test_whitespace_method(self):
        """Test method with whitespace."""
        loader = ToolBenchLoader()
        assert loader._normalize_method("  get  ") == "GET"


class TestParameterParsing:
    """Tests for parameter parsing."""

    def test_parse_parameters_from_list(self):
        """Test parsing parameters from list format."""
        loader = ToolBenchLoader()
        params = [
            {"name": "city", "type": "string", "description": "City name"},
            {"name": "count", "type": "integer", "default": 10},
        ]

        result = loader._parse_parameters_from_list(params, required=True)

        assert len(result) == 2
        assert result[0].name == "city"
        assert result[0].type == ParameterType.STRING
        assert result[0].required == True
        assert result[1].default == 10

    def test_parse_parameters_from_dict(self):
        """Test parsing parameters from dict format."""
        loader = ToolBenchLoader()
        params = {
            "city": {"type": "string", "required": True},
            "days": {"type": "int", "required": False, "default": 7},
        }

        result = loader._parse_parameters_from_dict(params)

        assert len(result) == 2
        city_param = next(p for p in result if p.name == "city")
        assert city_param.required == True
        assert city_param.type == ParameterType.STRING

    def test_parameter_type_aliases(self):
        """Test parameter type aliases are normalized."""
        loader = ToolBenchLoader()
        params = [
            {"name": "p1", "type": "str"},
            {"name": "p2", "type": "int"},
            {"name": "p3", "type": "bool"},
        ]

        result = loader._parse_parameters_from_list(params)

        assert result[0].type == ParameterType.STRING
        assert result[1].type == ParameterType.INTEGER
        assert result[2].type == ParameterType.BOOLEAN

    def test_parse_parameters_with_alternate_names(self):
        """Test parsing with alternate field names."""
        loader = ToolBenchLoader()
        params = [
            {"parameter_name": "city", "parameter_type": "string"},
            {"param_name": "count", "param_type": "integer"},
        ]

        result = loader._parse_parameters_from_list(params)

        assert len(result) == 2
        assert result[0].name == "city"
        assert result[1].name == "count"

    def test_parse_parameters_skips_invalid(self):
        """Test parsing skips parameters without names."""
        loader = ToolBenchLoader()
        params = [
            {"name": "valid", "type": "string"},
            {"type": "string"},  # No name
            {},  # Empty
        ]

        result = loader._parse_parameters_from_list(params)

        assert len(result) == 1
        assert result[0].name == "valid"

    def test_parse_parameters_string_required(self):
        """Test parsing handles string 'true'/'false' for required."""
        loader = ToolBenchLoader()
        params = [
            {"name": "p1", "required": "true"},
            {"name": "p2", "required": "false"},
        ]

        result = loader._parse_parameters_from_list(params)

        assert result[0].required == True
        assert result[1].required == False


class TestToolIdExtraction:
    """Tests for tool ID extraction."""

    def test_extract_from_id_field(self):
        """Test extracting tool ID from 'id' field."""
        loader = ToolBenchLoader()
        data = {"id": "my_tool", "name": "My Tool"}
        assert loader._extract_tool_id(data) == "my_tool"

    def test_extract_from_tool_id_field(self):
        """Test extracting tool ID from 'tool_id' field."""
        loader = ToolBenchLoader()
        data = {"tool_id": "my_tool", "name": "My Tool"}
        assert loader._extract_tool_id(data) == "my_tool"

    def test_extract_from_tool_name(self):
        """Test extracting tool ID from tool_name."""
        loader = ToolBenchLoader()
        data = {"tool_name": "My Weather API"}
        assert loader._extract_tool_id(data) == "my_weather_api"

    def test_extract_from_name(self):
        """Test extracting tool ID from name field."""
        loader = ToolBenchLoader()
        data = {"name": "My API"}
        assert loader._extract_tool_id(data) == "my_api"

    def test_extract_from_filename(self):
        """Test extracting tool ID from source filename."""
        loader = ToolBenchLoader()
        data = {}
        assert loader._extract_tool_id(data, "weather_api.json") == "weather_api"

    def test_extract_generates_hash(self):
        """Test hash generation as fallback."""
        loader = ToolBenchLoader()
        data = {"some_field": "value"}
        tool_id = loader._extract_tool_id(data)
        assert len(tool_id) == 16  # MD5 hash truncated to 16 chars


class TestEndpointIdGeneration:
    """Tests for endpoint ID generation."""

    def test_generate_endpoint_id(self):
        """Test endpoint ID generation."""
        loader = ToolBenchLoader()
        endpoint_id = loader._generate_endpoint_id("tool1", "get_data")
        assert endpoint_id == "tool1_get_data"

    def test_generate_endpoint_id_sanitizes(self):
        """Test endpoint ID sanitizes the name."""
        loader = ToolBenchLoader()
        endpoint_id = loader._generate_endpoint_id("tool1", "Get Data")
        assert endpoint_id == "tool1_get_data"


class TestDomainInference:
    """Tests for domain inference."""

    def test_domain_from_category(self):
        """Test domain is inferred from category."""
        loader = ToolBenchLoader()
        data = {
            "tool_name": "Test API",
            "category_name": "Weather",
            "api_list": [{"name": "test", "description": "test"}],
        }

        tool = loader._parse_tool(data)
        endpoint = tool.endpoints[0]
        assert endpoint.domain == "weather"

    def test_domain_from_path(self):
        """Test domain is inferred from path."""
        loader = ToolBenchLoader()
        data = {
            "name": "Test API",
            "endpoints": [{"name": "test", "path": "/api/finance/stocks"}],
        }

        tool = loader._parse_tool(data)
        endpoint = tool.endpoints[0]
        assert endpoint.domain == "finance"

    def test_domain_default(self):
        """Test default domain when none inferred."""
        loader = ToolBenchLoader()
        data = {
            "tool_name": "Test",
            "api_list": [{"name": "test"}],
        }

        tool = loader._parse_tool(data)
        endpoint = tool.endpoints[0]
        assert endpoint.domain == "general"


class TestLoadFile:
    """Tests for loading single files."""

    def test_load_valid_file(self):
        """Test loading a valid JSON file."""
        loader = ToolBenchLoader()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "tool_name": "Test API",
                    "api_list": [{"name": "test_endpoint", "description": "Test"}],
                },
                f,
            )
            f.flush()

            tool = loader.load_file(Path(f.name))

            assert tool is not None
            assert tool.name == "Test API"

            Path(f.name).unlink()

    def test_load_missing_file(self):
        """Test loading a missing file raises error."""
        loader = ToolBenchLoader()

        with pytest.raises(FileNotFoundError):
            loader.load_file(Path("/nonexistent/file.json"))

    def test_load_invalid_json(self):
        """Test loading invalid JSON returns None with skip_malformed."""
        loader = ToolBenchLoader(skip_malformed=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json {{{")
            f.flush()

            tool = loader.load_file(Path(f.name))
            assert tool is None

            Path(f.name).unlink()

    def test_load_invalid_json_raises_when_not_skipping(self):
        """Test loading invalid JSON raises when skip_malformed=False."""
        loader = ToolBenchLoader(skip_malformed=False)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json {{{")
            f.flush()

            with pytest.raises(json.JSONDecodeError):
                loader.load_file(Path(f.name))

            Path(f.name).unlink()


class TestLoadDirectory:
    """Tests for loading directories."""

    def test_load_directory(self):
        """Test loading a directory of JSON files."""
        loader = ToolBenchLoader()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            for i in range(3):
                with open(Path(tmpdir) / f"tool_{i}.json", "w") as f:
                    json.dump(
                        {"tool_name": f"Tool {i}", "api_list": [{"name": "test"}]}, f
                    )

            registry = loader.load_directory(Path(tmpdir), show_progress=False)

            assert registry.tool_count == 3

    def test_load_directory_with_limit(self):
        """Test loading with limit parameter."""
        loader = ToolBenchLoader()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            for i in range(10):
                with open(Path(tmpdir) / f"tool_{i}.json", "w") as f:
                    json.dump(
                        {"tool_name": f"Tool {i}", "api_list": [{"name": "test"}]}, f
                    )

            registry = loader.load_directory(Path(tmpdir), limit=3, show_progress=False)

            assert registry.tool_count == 3

    def test_load_directory_not_found(self):
        """Test loading nonexistent directory raises error."""
        loader = ToolBenchLoader()

        with pytest.raises(FileNotFoundError):
            loader.load_directory(Path("/nonexistent/directory"))

    def test_load_directory_skips_malformed(self):
        """Test malformed files are skipped when skip_malformed=True."""
        loader = ToolBenchLoader(skip_malformed=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Valid file
            with open(Path(tmpdir) / "valid.json", "w") as f:
                json.dump({"tool_name": "Valid", "api_list": [{"name": "test"}]}, f)

            # Invalid file
            with open(Path(tmpdir) / "invalid.json", "w") as f:
                f.write("not json")

            registry = loader.load_directory(Path(tmpdir), show_progress=False)

            assert registry.tool_count == 1

    def test_load_directory_custom_pattern(self):
        """Test loading with custom glob pattern."""
        loader = ToolBenchLoader()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create .json file
            with open(Path(tmpdir) / "tool.json", "w") as f:
                json.dump({"tool_name": "JSON Tool", "api_list": [{"name": "test"}]}, f)

            # Create .txt file (should be ignored)
            with open(Path(tmpdir) / "tool.txt", "w") as f:
                f.write("not a json file")

            registry = loader.load_directory(
                Path(tmpdir), pattern="*.json", show_progress=False
            )

            assert registry.tool_count == 1


class TestMalformedHandling:
    """Tests for malformed data handling."""

    def test_skip_malformed_entries(self):
        """Test malformed entries are skipped."""
        loader = ToolBenchLoader(skip_malformed=True)
        data = {
            "tool_name": "Test",
            "api_list": [
                {"name": "valid", "description": "Valid endpoint"},
                {},  # Malformed - no name
                {"name": "also_valid"},
            ],
        }

        tool = loader._parse_tool(data)

        # Should have 2 valid endpoints, malformed one skipped
        assert len(tool.endpoints) == 2

    def test_missing_name_skipped(self):
        """Test endpoints without names are skipped."""
        loader = ToolBenchLoader()
        data = {"tool_name": "Test", "api_list": [{"description": "No name field"}]}

        tool = loader._parse_tool(data)
        assert len(tool.endpoints) == 0


class TestDescriptionTruncation:
    """Tests for description truncation."""

    def test_long_description_truncated(self):
        """Test long descriptions are truncated."""
        loader = ToolBenchLoader(max_description_length=50)
        data = {
            "tool_name": "Test",
            "api_list": [{"name": "test", "description": "x" * 100}],
        }

        tool = loader._parse_tool(data)
        assert len(tool.endpoints[0].description) <= 50

    def test_tool_description_truncated(self):
        """Test tool descriptions are truncated."""
        loader = ToolBenchLoader(max_description_length=50)
        data = {
            "tool_name": "Test",
            "description": "x" * 100,
            "api_list": [],
        }

        tool = loader._parse_tool(data)
        assert len(tool.description) <= 50
