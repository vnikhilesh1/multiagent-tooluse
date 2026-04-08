"""Custom validators for tool registry models.

Provides reusable validation functions for:
- Type string normalization (str → string, int → integer)
- Description truncation with ellipsis
- ID sanitization for safe storage/URLs
- Domain inference from category or path
"""

import re
from typing import Optional

from src.models.registry import ParameterType


# Domain keyword mapping for inference
DOMAIN_KEYWORDS = {
    "weather": ["weather", "forecast", "climate", "temperature", "humidity", "wind", "rain"],
    "finance": ["finance", "stock", "trading", "market", "banking", "payment", "currency", "money", "price", "invest"],
    "social": ["social", "user", "profile", "friend", "follow", "post", "comment", "like", "share"],
    "search": ["search", "query", "find", "lookup", "discover"],
    "media": ["media", "video", "audio", "image", "photo", "music", "stream", "upload"],
    "travel": ["travel", "flight", "hotel", "booking", "destination", "trip", "vacation"],
    "food": ["food", "restaurant", "recipe", "menu", "delivery", "cuisine", "meal"],
    "health": ["health", "medical", "fitness", "doctor", "hospital", "medicine", "wellness"],
    "shopping": ["shop", "product", "cart", "order", "ecommerce", "store", "buy", "purchase"],
    "communication": ["email", "message", "sms", "notification", "chat", "send", "inbox"],
    "maps": ["map", "location", "geo", "direction", "place", "address", "coordinate", "route"],
    "news": ["news", "article", "headline", "feed", "press", "publish"],
    "entertainment": ["entertainment", "movie", "game", "sport", "event", "ticket", "show"],
    "data": ["data", "analytics", "statistics", "report", "metric", "dashboard"],
}


def normalize_type_string(type_str: Optional[str]) -> ParameterType:
    """Normalize a type string to a ParameterType enum.

    Handles common variations and aliases:
    - "str", "String", "STRING" → ParameterType.STRING
    - "int", "Int", "INTEGER" → ParameterType.INTEGER
    - "float", "double", "Number" → ParameterType.NUMBER
    - "bool", "Bool", "BOOLEAN" → ParameterType.BOOLEAN
    - "list", "List", "ARRAY" → ParameterType.ARRAY
    - "dict", "Dict", "OBJECT" → ParameterType.OBJECT
    - None, empty, or unrecognized → ParameterType.UNKNOWN

    Args:
        type_str: Raw type string from API specification

    Returns:
        Normalized ParameterType enum value

    Example:
        >>> normalize_type_string("str")
        ParameterType.STRING
        >>> normalize_type_string("INTEGER")
        ParameterType.INTEGER
        >>> normalize_type_string(None)
        ParameterType.UNKNOWN
    """
    return ParameterType.from_string(type_str)


def truncate_description(
    description: str,
    max_length: int = 500,
    suffix: str = "...",
) -> str:
    """Truncate a description to a maximum length.

    If the description exceeds max_length, truncates at a word boundary
    and appends the suffix. If no word boundary is found, truncates at
    max_length - len(suffix).

    Args:
        description: The description text to truncate
        max_length: Maximum allowed length (default 500)
        suffix: String to append when truncating (default "...")

    Returns:
        Truncated description with suffix if needed, or original if short enough

    Example:
        >>> truncate_description("Short text", max_length=500)
        'Short text'
        >>> truncate_description("A " * 300, max_length=20)
        'A A A A A A A A...'
    """
    if not description:
        return ""

    if len(description) <= max_length:
        return description

    # Calculate truncation point
    truncate_at = max_length - len(suffix)

    if truncate_at <= 0:
        return suffix[:max_length]

    # Try to truncate at word boundary
    truncated = description[:truncate_at]

    # Find last space for word boundary
    last_space = truncated.rfind(" ")

    if last_space > truncate_at // 2:  # Only use word boundary if reasonable
        truncated = truncated[:last_space]

    return truncated.rstrip() + suffix


def sanitize_id(
    raw_id: str,
    replacement: str = "_",
    lowercase: bool = True,
    max_length: int = 128,
) -> str:
    """Sanitize an ID string for safe storage and URLs.

    Performs the following transformations:
    1. Strip leading/trailing whitespace
    2. Optionally convert to lowercase
    3. Replace spaces and special characters with replacement
    4. Collapse multiple consecutive replacements/separators to single
    5. Remove leading/trailing replacements
    6. Truncate to max_length

    Valid characters after sanitization: a-z, A-Z, 0-9, underscore, hyphen

    Args:
        raw_id: The raw ID string to sanitize
        replacement: Character to replace invalid chars with (default "_")
        lowercase: Whether to convert to lowercase (default True)
        max_length: Maximum length of the result (default 128)

    Returns:
        Sanitized ID string

    Raises:
        ValueError: If raw_id is empty or results in empty string

    Example:
        >>> sanitize_id("My API Tool")
        'my_api_tool'
        >>> sanitize_id("Weather API (v2)")
        'weather_api_v2'
        >>> sanitize_id("test--id", replacement="-")
        'test-id'
    """
    if not raw_id or not raw_id.strip():
        raise ValueError("ID cannot be empty")

    result = raw_id.strip()

    if lowercase:
        result = result.lower()

    # Replace invalid characters
    result = re.sub(r"[^a-zA-Z0-9_\-]", replacement, result)

    # Collapse multiple consecutive replacements (handle both _ and -)
    if replacement:
        result = re.sub(re.escape(replacement) + "+", replacement, result)

    # Also collapse consecutive hyphens and underscores regardless of replacement
    result = re.sub(r"-+", "-", result)
    result = re.sub(r"_+", "_", result)

    # Remove leading/trailing replacements
    result = result.strip(replacement)
    result = result.strip("-_")

    # Truncate
    result = result[:max_length]

    # Remove trailing replacement after truncation
    result = result.rstrip(replacement)
    result = result.rstrip("-_")

    if not result:
        raise ValueError(f"ID '{raw_id}' results in empty string after sanitization")

    return result


def infer_domain(
    category: Optional[str] = None,
    path: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    default: str = "general",
) -> str:
    """Infer a semantic domain from available context.

    Uses the following priority order:
    1. Category (if provided and recognized)
    2. Path segments (looks for domain keywords)
    3. Name (looks for domain keywords)
    4. Description (looks for domain keywords)
    5. Default value

    Args:
        category: Tool category (e.g., "Finance", "Weather")
        path: API path (e.g., "/api/v1/weather/forecast")
        name: Endpoint or tool name
        description: Description text
        default: Default domain if none inferred (default "general")

    Returns:
        Inferred domain string (lowercase)

    Example:
        >>> infer_domain(category="Finance")
        'finance'
        >>> infer_domain(path="/api/weather/forecast")
        'weather'
        >>> infer_domain(name="getStockPrice")
        'finance'
    """
    # Try category first (highest priority)
    if category:
        domain = _find_domain_in_text(category)
        if domain:
            return domain

    # Try path
    if path:
        domain = _find_domain_in_text(path)
        if domain:
            return domain

    # Try name
    if name:
        domain = _find_domain_in_text(name)
        if domain:
            return domain

    # Try description (lowest priority due to potential noise)
    if description:
        domain = _find_domain_in_text(description)
        if domain:
            return domain

    return default


def _find_domain_in_text(text: str) -> Optional[str]:
    """Find a domain by searching for keywords in text.

    Args:
        text: Text to search for domain keywords

    Returns:
        Domain name if keywords found, None otherwise
    """
    if not text:
        return None

    # Convert camelCase/PascalCase to spaces for better matching
    # Insert space before uppercase letters that follow lowercase letters
    text_expanded = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text_lower = text_expanded.lower()

    # Check each domain's keywords
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for keyword in keywords:
            # Use word boundary matching to avoid partial matches
            if re.search(r"\b" + re.escape(keyword) + r"\b", text_lower):
                return domain

    return None
