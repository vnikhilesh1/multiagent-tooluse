"""Data models for the Tool Registry.

This module provides Pydantic models for representing tools, endpoints,
parameters, and the overall registry. These models are used throughout
the system for data validation and serialization.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ParameterType(str, Enum):
    """Enumeration of supported parameter types."""

    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    UNKNOWN = "unknown"


class Parameter(BaseModel):
    """Model representing an API parameter.

    Attributes:
        name: Parameter name
        type: Parameter type (string, integer, etc.)
        description: Human-readable description
        required: Whether the parameter is required
        default: Default value if not provided
        enum: List of allowed values (if constrained)
    """

    name: str
    type: ParameterType = ParameterType.STRING
    description: str = ""
    required: bool = False
    default: Optional[Any] = None
    enum: Optional[List[str]] = None

    @field_validator("type", mode="before")
    @classmethod
    def normalize_type(cls, v: Any) -> ParameterType:
        """Normalize type strings to ParameterType enum."""
        if isinstance(v, ParameterType):
            return v
        if isinstance(v, str):
            type_map = {
                "str": ParameterType.STRING,
                "int": ParameterType.INTEGER,
                "float": ParameterType.NUMBER,
                "bool": ParameterType.BOOLEAN,
                "list": ParameterType.ARRAY,
                "dict": ParameterType.OBJECT,
            }
            return type_map.get(v.lower(), ParameterType(v.lower()) if v.lower() in [e.value for e in ParameterType] else ParameterType.UNKNOWN)
        return ParameterType.UNKNOWN


class Endpoint(BaseModel):
    """Model representing an API endpoint.

    Attributes:
        id: Unique identifier for the endpoint
        tool_id: ID of the parent tool
        name: Human-readable name
        method: HTTP method (GET, POST, etc.)
        path: URL path
        description: Human-readable description
        parameters: List of parameters
        response_schema: Expected response structure
        domain: Domain/category of the endpoint
        raw_schema: Original schema from source data
        completeness_score: Score indicating how complete the definition is (0-1)
        inferred_fields: List of fields that were LLM-inferred
    """

    id: str
    tool_id: str
    name: str
    method: str = "GET"
    path: str = ""
    description: str = ""
    parameters: List[Parameter] = Field(default_factory=list)
    response_schema: Optional[Dict[str, Any]] = None
    domain: str = ""
    raw_schema: Optional[Dict[str, Any]] = None
    completeness_score: float = 0.0
    inferred_fields: List[str] = Field(default_factory=list)

    @field_validator("method", mode="before")
    @classmethod
    def normalize_method(cls, v: Any) -> str:
        """Normalize HTTP method to uppercase."""
        if isinstance(v, str):
            return v.upper()
        return "GET"


class Tool(BaseModel):
    """Model representing an API tool (collection of endpoints).

    Attributes:
        id: Unique identifier for the tool
        name: Human-readable name
        category: Tool category
        description: Human-readable description
        api_host: Base URL for the API
        endpoints: List of endpoints
        raw_schema: Original schema from source data
        completeness_score: Score indicating how complete the definition is (0-1)
    """

    id: str
    name: str
    category: str = ""
    description: str = ""
    api_host: str = ""
    endpoints: List[Endpoint] = Field(default_factory=list)
    raw_schema: Optional[Dict[str, Any]] = None
    completeness_score: float = 0.0


class ToolRegistry(BaseModel):
    """Model representing a collection of tools and endpoints.

    Attributes:
        tools: Dictionary mapping tool ID to Tool
        endpoints: Dictionary mapping endpoint ID to Endpoint
        domains: List of unique domain names
    """

    tools: Dict[str, Tool] = Field(default_factory=dict)
    endpoints: Dict[str, Endpoint] = Field(default_factory=dict)
    domains: List[str] = Field(default_factory=list)

    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the registry and index its endpoints.

        Args:
            tool: The Tool to add
        """
        self.tools[tool.id] = tool
        for endpoint in tool.endpoints:
            self.endpoints[endpoint.id] = endpoint
            if endpoint.domain and endpoint.domain not in self.domains:
                self.domains.append(endpoint.domain)

    def get_endpoint(self, endpoint_id: str) -> Optional[Endpoint]:
        """Retrieve an endpoint by ID.

        Args:
            endpoint_id: The endpoint ID to look up

        Returns:
            The Endpoint if found, None otherwise
        """
        return self.endpoints.get(endpoint_id)

    def get_endpoints_by_domain(self, domain: str) -> List[Endpoint]:
        """Filter endpoints by domain.

        Args:
            domain: The domain to filter by

        Returns:
            List of endpoints in the specified domain
        """
        return [ep for ep in self.endpoints.values() if ep.domain == domain]
