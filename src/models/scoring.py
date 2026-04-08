"""Completeness scoring for tools and endpoints.

Provides functions to calculate 0-1 completeness scores based on:
- Description presence and quality
- Parameter type coverage
- Response schema presence
- Name meaningfulness
- Path validity
"""

import re
from typing import List, Optional

from src.models.registry import Endpoint, Parameter, ParameterType, Tool


# Generic/non-descriptive names to penalize
GENERIC_NAMES = {
    "api",
    "endpoint",
    "method",
    "function",
    "call",
    "request",
    "response",
    "data",
    "get",
    "post",
    "put",
    "delete",
    "patch",
    "test",
    "temp",
    "tmp",
    "foo",
    "bar",
    "baz",
    "example",
    "sample",
    "demo",
}


def score_description(description: Optional[str], min_length: int = 10) -> float:
    """Score the quality of a description.

    Args:
        description: The description text
        min_length: Minimum length for a "good" description

    Returns:
        1.0 if description meets criteria, 0.0 otherwise
    """
    if not description:
        return 0.0

    cleaned = description.strip()
    if len(cleaned) < min_length:
        return 0.0

    # Check it's not just placeholder text
    placeholders = {"todo", "tbd", "n/a", "none", "null", "undefined", "description"}
    if cleaned.lower() in placeholders:
        return 0.0

    return 1.0


def score_parameter_types(parameters: List[Parameter]) -> float:
    """Score parameter type coverage.

    Args:
        parameters: List of parameters to evaluate

    Returns:
        Fraction of parameters with known types (0.0-1.0)
        Returns 1.0 if no parameters (nothing to score)
    """
    if not parameters:
        return 1.0  # No parameters = nothing to score, consider complete

    known_count = sum(1 for p in parameters if p.type != ParameterType.UNKNOWN)

    return known_count / len(parameters)


def score_response_schema(endpoint: Endpoint) -> float:
    """Score presence of response schema.

    Args:
        endpoint: Endpoint to check

    Returns:
        1.0 if has response_schema, 0.0 otherwise
    """
    if endpoint.response_schema is not None:
        return 1.0
    return 0.0


def score_name_quality(name: str) -> float:
    """Score the quality/meaningfulness of a name.

    A good name:
    - Is longer than 3 characters
    - Is not a generic term (api, endpoint, test, etc.)
    - Contains meaningful words (not just abbreviations)

    Args:
        name: The name to evaluate

    Returns:
        1.0 if name is meaningful, 0.0 otherwise
    """
    if not name:
        return 0.0

    cleaned = name.strip().lower()

    # Too short
    if len(cleaned) <= 3:
        return 0.0

    # Check against generic names
    # Split camelCase and snake_case for checking
    words = re.split(r"[_\s]+|(?<=[a-z])(?=[A-Z])", name)
    words = [w.lower() for w in words if w]

    # If all words are generic, it's not descriptive
    if all(w in GENERIC_NAMES for w in words):
        return 0.0

    # If the entire name (without separators) is generic
    if cleaned.replace("_", "").replace("-", "") in GENERIC_NAMES:
        return 0.0

    return 1.0


def score_path_validity(path: Optional[str]) -> float:
    """Score the validity of an API path.

    A valid path:
    - Starts with "/" or is a full URL
    - Contains path segments
    - Is not empty or just whitespace

    Args:
        path: The API path to evaluate

    Returns:
        1.0 if path is valid, 0.0 otherwise
    """
    if not path:
        return 0.0

    cleaned = path.strip()
    if not cleaned:
        return 0.0

    # Valid if starts with / or is a URL
    if cleaned.startswith("/"):
        return 1.0

    # Check for URL patterns
    if cleaned.startswith(("http://", "https://", "www.")):
        return 1.0

    # Check for path-like patterns (contains /)
    if "/" in cleaned:
        return 1.0

    return 0.0


def score_endpoint(endpoint: Endpoint) -> float:
    """Calculate completeness score for an endpoint.

    Scoring breakdown:
    - Description: +0.2 if has meaningful description (>10 chars)
    - Parameter types: +0.3 if all parameters have known types
    - Response schema: +0.2 if has response_schema defined
    - Meaningful name: +0.1 if name is descriptive
    - Valid path: +0.2 if has valid API path

    Args:
        endpoint: Endpoint to score

    Returns:
        Completeness score between 0.0 and 1.0
    """
    score = 0.0

    # Description: +0.2
    score += 0.2 * score_description(endpoint.description)

    # Parameter types: +0.3
    score += 0.3 * score_parameter_types(endpoint.parameters)

    # Response schema: +0.2
    score += 0.2 * score_response_schema(endpoint)

    # Meaningful name: +0.1
    score += 0.1 * score_name_quality(endpoint.name)

    # Valid path: +0.2
    score += 0.2 * score_path_validity(endpoint.path)

    return round(score, 2)


def score_tool(tool: Tool) -> float:
    """Calculate completeness score for a tool.

    Scoring breakdown:
    - Description: +0.2 if has meaningful description (>10 chars)
    - Endpoints complete: +0.5 weighted by average endpoint completeness
    - Meaningful name: +0.1 if name is descriptive
    - Has endpoints: +0.2 if has at least one endpoint

    Args:
        tool: Tool to score

    Returns:
        Completeness score between 0.0 and 1.0
    """
    score = 0.0

    # Description: +0.2
    score += 0.2 * score_description(tool.description)

    # Meaningful name: +0.1
    score += 0.1 * score_name_quality(tool.name)

    # Has endpoints: +0.2
    if tool.endpoints:
        score += 0.2

        # Endpoints complete: +0.5 weighted by average
        avg_endpoint_score = sum(
            score_endpoint(ep) for ep in tool.endpoints
        ) / len(tool.endpoints)
        score += 0.5 * avg_endpoint_score

    return round(score, 2)


def get_endpoint_score_breakdown(endpoint: Endpoint) -> dict:
    """Get detailed breakdown of endpoint scoring.

    Args:
        endpoint: Endpoint to analyze

    Returns:
        Dictionary with individual component scores and total
    """
    desc_score = score_description(endpoint.description)
    param_score = score_parameter_types(endpoint.parameters)
    response_score = score_response_schema(endpoint)
    name_score = score_name_quality(endpoint.name)
    path_score = score_path_validity(endpoint.path)

    return {
        "description": {
            "score": desc_score,
            "weight": 0.2,
            "weighted": round(0.2 * desc_score, 3),
        },
        "parameter_types": {
            "score": param_score,
            "weight": 0.3,
            "weighted": round(0.3 * param_score, 3),
        },
        "response_schema": {
            "score": response_score,
            "weight": 0.2,
            "weighted": round(0.2 * response_score, 3),
        },
        "name_quality": {
            "score": name_score,
            "weight": 0.1,
            "weighted": round(0.1 * name_score, 3),
        },
        "path_validity": {
            "score": path_score,
            "weight": 0.2,
            "weighted": round(0.2 * path_score, 3),
        },
        "total": score_endpoint(endpoint),
    }


def get_tool_score_breakdown(tool: Tool) -> dict:
    """Get detailed breakdown of tool scoring.

    Args:
        tool: Tool to analyze

    Returns:
        Dictionary with individual component scores and total
    """
    desc_score = score_description(tool.description)
    name_score = score_name_quality(tool.name)
    has_endpoints = 1.0 if tool.endpoints else 0.0

    if tool.endpoints:
        avg_endpoint_score = sum(
            score_endpoint(ep) for ep in tool.endpoints
        ) / len(tool.endpoints)
    else:
        avg_endpoint_score = 0.0

    return {
        "description": {
            "score": desc_score,
            "weight": 0.2,
            "weighted": round(0.2 * desc_score, 3),
        },
        "name_quality": {
            "score": name_score,
            "weight": 0.1,
            "weighted": round(0.1 * name_score, 3),
        },
        "has_endpoints": {
            "score": has_endpoints,
            "weight": 0.2,
            "weighted": round(0.2 * has_endpoints, 3),
        },
        "endpoints_completeness": {
            "score": avg_endpoint_score,
            "weight": 0.5,
            "weighted": round(0.5 * avg_endpoint_score, 3) if tool.endpoints else 0.0,
        },
        "total": score_tool(tool),
    }
