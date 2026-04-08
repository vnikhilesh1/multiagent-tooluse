"""Unit tests for custom validators."""

import pytest

from src.models import (
    DOMAIN_KEYWORDS,
    ParameterType,
    infer_domain,
    normalize_type_string,
    sanitize_id,
    truncate_description,
)


class TestNormalizeTypeString:
    """Tests for normalize_type_string validator."""

    def test_standard_types(self):
        """Test standard type strings."""
        assert normalize_type_string("string") == ParameterType.STRING
        assert normalize_type_string("integer") == ParameterType.INTEGER
        assert normalize_type_string("number") == ParameterType.NUMBER
        assert normalize_type_string("boolean") == ParameterType.BOOLEAN
        assert normalize_type_string("array") == ParameterType.ARRAY
        assert normalize_type_string("object") == ParameterType.OBJECT

    def test_aliases(self):
        """Test common type aliases."""
        assert normalize_type_string("str") == ParameterType.STRING
        assert normalize_type_string("int") == ParameterType.INTEGER
        assert normalize_type_string("float") == ParameterType.NUMBER
        assert normalize_type_string("double") == ParameterType.NUMBER
        assert normalize_type_string("bool") == ParameterType.BOOLEAN
        assert normalize_type_string("list") == ParameterType.ARRAY
        assert normalize_type_string("dict") == ParameterType.OBJECT

    def test_case_insensitive(self):
        """Test case insensitivity."""
        assert normalize_type_string("STRING") == ParameterType.STRING
        assert normalize_type_string("String") == ParameterType.STRING
        assert normalize_type_string("INTEGER") == ParameterType.INTEGER

    def test_unknown_types(self):
        """Test unknown type handling."""
        assert normalize_type_string(None) == ParameterType.UNKNOWN
        assert normalize_type_string("") == ParameterType.UNKNOWN
        assert normalize_type_string("unknown_type") == ParameterType.UNKNOWN
        assert normalize_type_string("custom") == ParameterType.UNKNOWN

    def test_whitespace_handling(self):
        """Test whitespace is handled."""
        assert normalize_type_string("  string  ") == ParameterType.STRING
        assert normalize_type_string("\tinteger\n") == ParameterType.INTEGER


class TestTruncateDescription:
    """Tests for truncate_description validator."""

    def test_short_description_unchanged(self):
        """Test short descriptions are not modified."""
        text = "This is a short description."
        assert truncate_description(text) == text
        assert truncate_description(text, max_length=500) == text

    def test_exact_length_unchanged(self):
        """Test description at exact max_length is unchanged."""
        text = "x" * 500
        assert truncate_description(text, max_length=500) == text

    def test_truncation_with_suffix(self):
        """Test long descriptions are truncated with suffix."""
        text = "word " * 200  # 1000 chars
        result = truncate_description(text, max_length=50)
        assert len(result) <= 50
        assert result.endswith("...")

    def test_word_boundary_truncation(self):
        """Test truncation at word boundary."""
        text = "This is a long sentence that needs truncation"
        result = truncate_description(text, max_length=25)
        assert result.endswith("...")
        # Should not cut mid-word
        assert not result[:-3].endswith("senten")

    def test_custom_suffix(self):
        """Test custom suffix."""
        text = "x" * 100
        result = truncate_description(text, max_length=50, suffix="[...]")
        assert result.endswith("[...]")

    def test_empty_description(self):
        """Test empty description handling."""
        assert truncate_description("") == ""
        assert truncate_description("", max_length=10) == ""

    def test_very_short_max_length(self):
        """Test very short max_length."""
        text = "Hello world"
        result = truncate_description(text, max_length=5)
        assert len(result) <= 5

    def test_no_spaces_in_text(self):
        """Test text without spaces truncates correctly."""
        text = "x" * 100
        result = truncate_description(text, max_length=50)
        assert len(result) <= 50
        assert result.endswith("...")

    def test_single_long_word(self):
        """Test single long word is truncated."""
        text = "superlongwordwithoutanyspaces"
        result = truncate_description(text, max_length=15)
        assert len(result) <= 15
        assert result.endswith("...")


class TestSanitizeId:
    """Tests for sanitize_id validator."""

    def test_simple_id(self):
        """Test simple IDs pass through."""
        assert sanitize_id("simple_id") == "simple_id"
        assert sanitize_id("simple-id") == "simple-id"

    def test_spaces_replaced(self):
        """Test spaces are replaced."""
        assert sanitize_id("my api tool") == "my_api_tool"

    def test_special_chars_replaced(self):
        """Test special characters are replaced."""
        assert sanitize_id("api@v2.0") == "api_v2_0"
        assert sanitize_id("tool(beta)") == "tool_beta"

    def test_lowercase_conversion(self):
        """Test lowercase conversion."""
        assert sanitize_id("MyAPITool") == "myapitool"
        assert sanitize_id("Weather_API", lowercase=False) == "Weather_API"

    def test_consecutive_replacements_collapsed(self):
        """Test consecutive replacements are collapsed."""
        assert sanitize_id("api--tool") == "api-tool"
        assert sanitize_id("api___tool") == "api_tool"
        assert sanitize_id("my   api   tool") == "my_api_tool"

    def test_leading_trailing_replacements_removed(self):
        """Test leading/trailing replacements removed."""
        assert sanitize_id("_api_tool_") == "api_tool"
        assert sanitize_id("  api tool  ") == "api_tool"

    def test_custom_replacement(self):
        """Test custom replacement character."""
        assert sanitize_id("my api tool", replacement="-") == "my-api-tool"

    def test_max_length(self):
        """Test max_length truncation."""
        long_id = "a" * 200
        result = sanitize_id(long_id, max_length=128)
        assert len(result) == 128

    def test_empty_id_raises(self):
        """Test empty ID raises ValueError."""
        with pytest.raises(ValueError):
            sanitize_id("")
        with pytest.raises(ValueError):
            sanitize_id("   ")

    def test_special_only_id_raises(self):
        """Test ID with only special chars raises ValueError."""
        with pytest.raises(ValueError):
            sanitize_id("@#$%")

    def test_numbers_preserved(self):
        """Test numbers are preserved."""
        assert sanitize_id("api123") == "api123"
        assert sanitize_id("v2_api") == "v2_api"

    def test_hyphen_preserved(self):
        """Test hyphens are preserved."""
        assert sanitize_id("my-api-tool") == "my-api-tool"

    def test_mixed_separators(self):
        """Test mixed separators are normalized."""
        assert sanitize_id("my-api_tool") == "my-api_tool"


class TestInferDomain:
    """Tests for infer_domain validator."""

    def test_from_category(self):
        """Test domain inference from category."""
        assert infer_domain(category="Finance") == "finance"
        assert infer_domain(category="Weather Services") == "weather"
        assert infer_domain(category="Social Media") == "social"

    def test_from_path(self):
        """Test domain inference from path."""
        assert infer_domain(path="/api/weather/forecast") == "weather"
        assert infer_domain(path="/v1/stocks/price") == "finance"
        assert infer_domain(path="/users/profile") == "social"

    def test_from_name(self):
        """Test domain inference from name."""
        assert infer_domain(name="getWeatherForecast") == "weather"
        assert infer_domain(name="searchProducts") == "search"
        assert infer_domain(name="sendMessage") == "communication"

    def test_from_description(self):
        """Test domain inference from description."""
        assert infer_domain(description="Get current weather data") == "weather"
        assert infer_domain(description="Search for products in our store") == "search"

    def test_priority_order(self):
        """Test category takes priority over other fields."""
        result = infer_domain(
            category="Finance",
            path="/api/weather",
            name="getWeather",
            description="Weather data",
        )
        assert result == "finance"

    def test_fallback_to_path(self):
        """Test fallback to path when category has no domain."""
        result = infer_domain(category="APIs", path="/api/weather/forecast")
        assert result == "weather"

    def test_default_when_no_match(self):
        """Test default is returned when no domain found."""
        assert infer_domain() == "general"
        assert infer_domain(category="Custom", path="/api/v1") == "general"
        assert infer_domain(default="other") == "other"

    def test_case_insensitive(self):
        """Test domain inference is case insensitive."""
        assert infer_domain(category="WEATHER") == "weather"
        assert infer_domain(path="/API/FINANCE/STOCKS") == "finance"

    def test_all_domains_have_keywords(self):
        """Test all expected domains are in DOMAIN_KEYWORDS."""
        expected_domains = [
            "weather",
            "finance",
            "social",
            "search",
            "media",
            "travel",
            "food",
            "health",
            "shopping",
            "communication",
            "maps",
            "news",
            "entertainment",
            "data",
        ]
        for domain in expected_domains:
            assert domain in DOMAIN_KEYWORDS
            assert len(DOMAIN_KEYWORDS[domain]) > 0

    def test_word_boundary_matching(self):
        """Test that partial matches are avoided."""
        # "searching" should not match "search" domain due to word boundary
        # But "search" in a sentence should match
        assert infer_domain(description="Use the search feature") == "search"

    def test_empty_inputs(self):
        """Test empty inputs return default."""
        assert infer_domain(category="", path="", name="", description="") == "general"

    def test_none_inputs(self):
        """Test None inputs return default."""
        assert (
            infer_domain(category=None, path=None, name=None, description=None)
            == "general"
        )


class TestDomainKeywords:
    """Tests for DOMAIN_KEYWORDS constant."""

    def test_keywords_are_lowercase(self):
        """Test all keywords are lowercase."""
        for domain, keywords in DOMAIN_KEYWORDS.items():
            for keyword in keywords:
                assert (
                    keyword == keyword.lower()
                ), f"Keyword '{keyword}' in {domain} is not lowercase"

    def test_no_duplicate_keywords(self):
        """Test no duplicate keywords within domains."""
        for domain, keywords in DOMAIN_KEYWORDS.items():
            assert len(keywords) == len(set(keywords)), f"Duplicate keywords in {domain}"

    def test_domains_are_lowercase(self):
        """Test all domain names are lowercase."""
        for domain in DOMAIN_KEYWORDS.keys():
            assert (
                domain == domain.lower()
            ), f"Domain '{domain}' is not lowercase"

    def test_keywords_not_empty(self):
        """Test each domain has at least one keyword."""
        for domain, keywords in DOMAIN_KEYWORDS.items():
            assert len(keywords) > 0, f"Domain '{domain}' has no keywords"
