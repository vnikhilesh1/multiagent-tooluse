"""Unit tests for LLM client."""

import os
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from src.llm import LLMClient, LLMError, LLMResponse, LLMValidationError
from src.llm.exceptions import LLMAPIError, LLMRateLimitError


class SampleResponse(BaseModel):
    """Sample response model for testing."""

    name: str
    value: int


class TestLLMClientInit:
    """Tests for LLMClient initialization."""

    def test_init_with_api_key(self):
        """Test client initializes with explicit API key."""
        client = LLMClient(api_key="test-key")
        assert client.api_key == "test-key"
        assert client.default_model == "claude-sonnet-4-20250514"

    def test_init_with_defaults(self):
        """Test client initializes with default values."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
            client = LLMClient()
            assert client.default_model == "claude-sonnet-4-20250514"

    def test_init_with_custom_model(self):
        """Test client accepts custom default model."""
        client = LLMClient(api_key="test-key", default_model="claude-opus-4-20250514")
        assert client.default_model == "claude-opus-4-20250514"

    def test_init_with_base_url(self):
        """Test client accepts Hyperspace proxy URL."""
        client = LLMClient(
            api_key="test-key", base_url="http://localhost:6655/anthropic"
        )
        assert client.base_url == "http://localhost:6655/anthropic"

    def test_init_reads_env_vars(self):
        """Test client reads from environment variables."""
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "env-key",
                "LLM_BASE_URL": "http://proxy:6655",
                "LLM_MODEL": "claude-haiku-4-5-20251001",
            },
            clear=False,
        ):
            client = LLMClient()
            assert client.api_key == "env-key"
            assert client.base_url == "http://proxy:6655"
            assert client.default_model == "claude-haiku-4-5-20251001"

    def test_init_explicit_overrides_env(self):
        """Test explicit parameters override environment variables."""
        with patch.dict(
            os.environ,
            {"ANTHROPIC_API_KEY": "env-key", "LLM_MODEL": "env-model"},
            clear=False,
        ):
            client = LLMClient(api_key="explicit-key", default_model="explicit-model")
            assert client.api_key == "explicit-key"
            assert client.default_model == "explicit-model"


class TestComplete:
    """Tests for complete() method."""

    @patch("src.llm.client.anthropic.Anthropic")
    def test_complete_returns_string(self, mock_anthropic):
        """Test complete() returns response text."""
        # Setup mock
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_response = MagicMock()
        mock_content_block = MagicMock()
        mock_content_block.text = "Hello, world!"
        mock_content_block.type = "text"
        mock_response.content = [mock_content_block]
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response

        client = LLMClient(api_key="test-key")
        result = client.complete("Say hello")

        assert result == "Hello, world!"
        assert mock_client.messages.create.called

    @patch("src.llm.client.anthropic.Anthropic")
    def test_complete_with_system_prompt(self, mock_anthropic):
        """Test complete() passes system prompt correctly."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_response = MagicMock()
        mock_content_block = MagicMock()
        mock_content_block.text = "Response"
        mock_content_block.type = "text"
        mock_response.content = [mock_content_block]
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response

        client = LLMClient(api_key="test-key")
        client.complete("Hello", system="You are helpful")

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are helpful"


class TestCompleteJSON:
    """Tests for complete_json() method."""

    def test_extract_json_direct(self):
        """Test JSON extraction from direct JSON response."""
        client = LLMClient(api_key="test-key")
        result = client._extract_json('{"name": "test", "value": 42}')
        assert result == {"name": "test", "value": 42}

    def test_extract_json_from_markdown(self):
        """Test JSON extraction from markdown code block."""
        client = LLMClient(api_key="test-key")
        text = """Here's the JSON:
```json
{"name": "test", "value": 42}
```"""
        result = client._extract_json(text)
        assert result == {"name": "test", "value": 42}

    def test_extract_json_from_markdown_no_language(self):
        """Test JSON extraction from markdown code block without language tag."""
        client = LLMClient(api_key="test-key")
        text = """Result:
```
{"name": "test", "value": 42}
```"""
        result = client._extract_json(text)
        assert result == {"name": "test", "value": 42}

    def test_extract_json_embedded(self):
        """Test JSON extraction from text with embedded JSON."""
        client = LLMClient(api_key="test-key")
        text = 'Here is the result: {"name": "test", "value": 42} and more text'
        result = client._extract_json(text)
        assert result == {"name": "test", "value": 42}

    def test_extract_json_array(self):
        """Test JSON extraction for array."""
        client = LLMClient(api_key="test-key")
        result = client._extract_json('[{"a": 1}, {"b": 2}]')
        assert result == [{"a": 1}, {"b": 2}]

    def test_extract_json_invalid(self):
        """Test JSON extraction raises error for invalid JSON."""
        client = LLMClient(api_key="test-key")
        with pytest.raises(LLMValidationError):
            client._extract_json("This is not JSON at all")


class TestCompleteStructured:
    """Tests for complete_structured() method."""

    @patch("src.llm.client.anthropic.Anthropic")
    def test_structured_returns_model(self, mock_anthropic):
        """Test complete_structured() returns Pydantic model instance."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_response = MagicMock()
        mock_content_block = MagicMock()
        mock_content_block.text = '{"name": "test", "value": 42}'
        mock_content_block.type = "text"
        mock_response.content = [mock_content_block]
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response

        client = LLMClient(api_key="test-key")
        result = client.complete_structured("Get data", SampleResponse)

        assert isinstance(result, SampleResponse)
        assert result.name == "test"
        assert result.value == 42

    @patch("src.llm.client.anthropic.Anthropic")
    def test_structured_validation_error(self, mock_anthropic):
        """Test complete_structured() raises error for invalid data."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_response = MagicMock()
        mock_content_block = MagicMock()
        # Missing required 'value' field
        mock_content_block.text = '{"name": "test"}'
        mock_content_block.type = "text"
        mock_response.content = [mock_content_block]
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response

        client = LLMClient(api_key="test-key")
        with pytest.raises(LLMValidationError):
            client.complete_structured("Get data", SampleResponse)


class TestCompleteWithTools:
    """Tests for complete_with_tools() method."""

    @patch("src.llm.client.anthropic.Anthropic")
    def test_tools_passed_to_api(self, mock_anthropic):
        """Test tools are passed correctly to API."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_response = MagicMock()
        mock_content_block = MagicMock()
        mock_content_block.type = "text"
        mock_content_block.text = "Using tool"
        mock_response.content = [mock_content_block]
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.stop_reason = "tool_use"
        mock_client.messages.create.return_value = mock_response

        tools = [
            {
                "name": "get_weather",
                "description": "Get weather for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            }
        ]

        client = LLMClient(api_key="test-key")
        result = client.complete_with_tools("What's the weather?", tools)

        assert isinstance(result, LLMResponse)
        call_kwargs = mock_client.messages.create.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] == tools

    @patch("src.llm.client.anthropic.Anthropic")
    def test_tool_choice_passed(self, mock_anthropic):
        """Test tool_choice is passed correctly."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_response = MagicMock()
        mock_content_block = MagicMock()
        mock_content_block.type = "text"
        mock_content_block.text = "Using tool"
        mock_response.content = [mock_content_block]
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.stop_reason = "tool_use"
        mock_client.messages.create.return_value = mock_response

        tools = [{"name": "test", "description": "test", "input_schema": {"type": "object"}}]
        tool_choice = {"type": "tool", "name": "test"}

        client = LLMClient(api_key="test-key")
        client.complete_with_tools("Test", tools, tool_choice=tool_choice)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["tool_choice"] == tool_choice


class TestChat:
    """Tests for chat() method."""

    @patch("src.llm.client.anthropic.Anthropic")
    def test_chat_with_messages(self, mock_anthropic):
        """Test chat() handles message history."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_response = MagicMock()
        mock_content_block = MagicMock()
        mock_content_block.text = "I'm doing well!"
        mock_content_block.type = "text"
        mock_response.content = [mock_content_block]
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 20
        mock_response.usage.output_tokens = 10
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        client = LLMClient(api_key="test-key")
        result = client.chat(messages)

        assert isinstance(result, LLMResponse)
        assert result.content == "I'm doing well!"
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["messages"] == messages


class TestLLMResponse:
    """Tests for LLMResponse model."""

    def test_response_creation(self):
        """Test LLMResponse can be created with all fields."""
        response = LLMResponse(
            content="test content",
            model="claude-sonnet-4-20250514",
            usage={"input_tokens": 10, "output_tokens": 5},
            stop_reason="end_turn",
            raw_response={"content": [{"type": "text", "text": "test"}]},
        )
        assert response.content == "test content"
        assert response.model == "claude-sonnet-4-20250514"
        assert response.usage["input_tokens"] == 10
        assert response.stop_reason == "end_turn"

    def test_response_minimal(self):
        """Test LLMResponse with minimal fields."""
        response = LLMResponse(
            content="test",
            model="test-model",
            usage={},
        )
        assert response.content == "test"
        assert response.stop_reason is None
        assert response.raw_response is None


class TestExceptions:
    """Tests for exception classes."""

    def test_llm_error_base(self):
        """Test LLMError is the base exception."""
        assert issubclass(LLMRateLimitError, LLMError)
        assert issubclass(LLMAPIError, LLMError)
        assert issubclass(LLMValidationError, LLMError)

    def test_rate_limit_error_retry_after(self):
        """Test LLMRateLimitError stores retry_after."""
        exc = LLMRateLimitError("Rate limited", retry_after=5.0)
        assert exc.retry_after == 5.0
        assert str(exc) == "Rate limited"

    def test_api_error_status_code(self):
        """Test LLMAPIError stores status_code."""
        exc = LLMAPIError("Server error", status_code=500)
        assert exc.status_code == 500
        assert str(exc) == "Server error"


class TestPydanticToTool:
    """Tests for pydantic_to_tool() helper."""

    def test_converts_model_to_tool(self):
        """Test Pydantic model converts to tool definition."""
        from src.llm.client import pydantic_to_tool

        class WeatherInput(BaseModel):
            location: str
            units: str = "celsius"

        tool = pydantic_to_tool(
            name="get_weather",
            description="Get weather for a location",
            model=WeatherInput,
        )

        assert tool["name"] == "get_weather"
        assert tool["description"] == "Get weather for a location"
        assert "input_schema" in tool
        assert tool["input_schema"]["type"] == "object"
        assert "location" in tool["input_schema"]["properties"]

    def test_preserves_field_descriptions(self):
        """Test field descriptions are preserved in schema."""
        from src.llm.client import pydantic_to_tool
        from pydantic import Field

        class InputWithDescriptions(BaseModel):
            query: str = Field(description="The search query")
            limit: int = Field(default=10, description="Max results")

        tool = pydantic_to_tool("search", "Search things", InputWithDescriptions)
        props = tool["input_schema"]["properties"]
        assert "description" in props["query"]
        assert props["query"]["description"] == "The search query"


class TestCreateClientFromConfig:
    """Tests for create_client_from_config() helper."""

    def test_creates_client_from_config(self):
        """Test client creation from config object."""
        from src.llm.client import create_client_from_config

        # Create a mock config
        mock_config = MagicMock()
        mock_config.models.primary = "claude-opus-4-20250514"
        mock_config.quality.max_retries = 5

        client = create_client_from_config(mock_config)

        assert client.default_model == "claude-opus-4-20250514"
        assert client.max_retries == 5
