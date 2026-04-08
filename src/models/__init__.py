"""Data models for tool registry.

This module provides Pydantic models for representing:
- API parameters with type information
- API endpoints with full specifications
- Tools containing multiple endpoints
- A registry for managing tools and endpoints

Also provides validation utilities:
- Type normalization
- Description truncation
- ID sanitization
- Domain inference

And scoring utilities:
- Completeness scoring for tools and endpoints
"""

from src.models.registry import (
    Endpoint,
    Parameter,
    ParameterType,
    Tool,
    ToolRegistry,
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
from src.models.validators import (
    DOMAIN_KEYWORDS,
    infer_domain,
    normalize_type_string,
    sanitize_id,
    truncate_description,
)

__all__ = [
    # Models
    "ParameterType",
    "Parameter",
    "Endpoint",
    "Tool",
    "ToolRegistry",
    # Validators
    "normalize_type_string",
    "truncate_description",
    "sanitize_id",
    "infer_domain",
    "DOMAIN_KEYWORDS",
    # Scoring
    "score_endpoint",
    "score_tool",
    "score_description",
    "score_parameter_types",
    "score_response_schema",
    "score_name_quality",
    "score_path_validity",
    "get_endpoint_score_breakdown",
    "get_tool_score_breakdown",
    "GENERIC_NAMES",
]
