"""Tool registry models.

Provides data models for API tools and endpoints with:
- Strong typing via Pydantic
- Validation of parameter types and constraints
- Registry for efficient tool/endpoint lookup
- Domain-based filtering for endpoint discovery
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, PrivateAttr, computed_field, field_validator


class ParameterType(str, Enum):
    """Supported parameter types for API endpoints.

    Maps to JSON Schema / OpenAPI types with an 'unknown' fallback
    for incomplete specifications.
    """

    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, type_str: Optional[str]) -> "ParameterType":
        """Convert a string to ParameterType, defaulting to UNKNOWN.

        Args:
            type_str: Type string from API spec (e.g., "string", "int")

        Returns:
            Corresponding ParameterType enum value
        """
        if type_str is None:
            return cls.UNKNOWN

        type_lower = type_str.lower().strip()

        # Handle common aliases
        type_mapping = {
            "string": cls.STRING,
            "str": cls.STRING,
            "integer": cls.INTEGER,
            "int": cls.INTEGER,
            "number": cls.NUMBER,
            "float": cls.NUMBER,
            "double": cls.NUMBER,
            "boolean": cls.BOOLEAN,
            "bool": cls.BOOLEAN,
            "array": cls.ARRAY,
            "list": cls.ARRAY,
            "object": cls.OBJECT,
            "dict": cls.OBJECT,
        }

        return type_mapping.get(type_lower, cls.UNKNOWN)


class Parameter(BaseModel):
    """API endpoint parameter specification.

    Represents a single parameter for an API endpoint with full
    type information and constraints.

    Attributes:
        name: Parameter name as used in the API
        type: Parameter type (string, integer, etc.)
        description: Human-readable description
        required: Whether the parameter is required
        default: Default value if not provided
        enum: List of allowed values (if constrained)
    """

    name: str
    type: ParameterType = ParameterType.UNKNOWN
    description: str = ""
    required: bool = False
    default: Optional[Any] = None
    enum: Optional[List[Any]] = None

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        """Ensure parameter name is not empty."""
        if not v or not v.strip():
            raise ValueError("Parameter name cannot be empty")
        return v


class Endpoint(BaseModel):
    """API endpoint specification.

    Represents a single API endpoint with full specification including
    parameters, response schema, and metadata for completeness tracking.

    Attributes:
        id: Unique endpoint identifier (tool_id + endpoint_name hash)
        tool_id: Parent tool identifier
        name: Endpoint name/operation ID
        method: HTTP method (GET, POST, etc.)
        path: URL path template
        description: Human-readable description
        parameters: List of endpoint parameters
        response_schema: Expected response structure
        domain: Semantic domain (e.g., "weather", "finance")
        raw_schema: Original OpenAPI/spec schema
        completeness_score: 0.0-1.0 indicating spec completeness
        inferred_fields: List of fields that were inferred (not in original spec)
    """

    id: str
    tool_id: str
    name: str
    method: str = "GET"
    path: str = ""
    description: str = ""
    parameters: List[Parameter] = Field(default_factory=list)
    response_schema: Optional[Dict[str, Any]] = None
    domain: str = "general"
    raw_schema: Optional[Dict[str, Any]] = None
    inferred_fields: List[str] = Field(default_factory=list)

    @field_validator("method")
    @classmethod
    def normalize_method(cls, v: str) -> str:
        """Normalize HTTP method to uppercase."""
        return v.upper()

    @computed_field
    @property
    def completeness_score(self) -> float:
        """Calculate completeness score for this endpoint."""
        from src.models.scoring import score_endpoint

        return score_endpoint(self)

    @computed_field
    @property
    def required_parameters(self) -> List[Parameter]:
        """Get list of required parameters."""
        return [p for p in self.parameters if p.required]

    @computed_field
    @property
    def optional_parameters(self) -> List[Parameter]:
        """Get list of optional parameters."""
        return [p for p in self.parameters if not p.required]


class Tool(BaseModel):
    """API tool specification.

    Represents an API tool (service) containing multiple endpoints.

    Attributes:
        id: Unique tool identifier
        name: Tool/API name
        category: Tool category (e.g., "Data", "Finance")
        description: Human-readable description
        api_host: Base URL for API calls
        endpoints: List of endpoints in this tool
        raw_schema: Original OpenAPI/spec schema
        completeness_score: Average completeness across endpoints
    """

    id: str
    name: str
    category: str = "Uncategorized"
    description: str = ""
    api_host: str = ""
    endpoints: List[Endpoint] = Field(default_factory=list)
    raw_schema: Optional[Dict[str, Any]] = None

    @field_validator("id", "name")
    @classmethod
    def not_empty(cls, v: str) -> str:
        """Ensure id and name are not empty."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v

    @computed_field
    @property
    def completeness_score(self) -> float:
        """Calculate completeness score for this tool."""
        from src.models.scoring import score_tool

        return score_tool(self)

    @computed_field
    @property
    def endpoint_count(self) -> int:
        """Get number of endpoints."""
        return len(self.endpoints)

    def get_endpoint(self, endpoint_id: str) -> Optional[Endpoint]:
        """Get endpoint by ID.

        Args:
            endpoint_id: Endpoint identifier

        Returns:
            Endpoint if found, None otherwise
        """
        for endpoint in self.endpoints:
            if endpoint.id == endpoint_id:
                return endpoint
        return None


class ToolRegistry(BaseModel):
    """Registry for managing tools and endpoints.

    Provides efficient lookup and filtering of tools and endpoints
    with domain-based organization.

    Attributes:
        tools: Dictionary mapping tool_id to Tool
    """

    tools: Dict[str, Tool] = Field(default_factory=dict)

    # Private attributes for indexing
    _endpoint_index: Dict[str, Endpoint] = PrivateAttr(default_factory=dict)
    _domains: Set[str] = PrivateAttr(default_factory=set)

    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, __context: Any) -> None:
        """Initialize indexes after model creation."""
        # Rebuild indexes from any tools passed during construction
        for tool in self.tools.values():
            for endpoint in tool.endpoints:
                self._endpoint_index[endpoint.id] = endpoint
                self._domains.add(endpoint.domain)

    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the registry and index its endpoints.

        Args:
            tool: Tool to add

        Raises:
            ValueError: If tool with same ID already exists
        """
        if tool.id in self.tools:
            raise ValueError(f"Tool with ID '{tool.id}' already exists in registry")

        self.tools[tool.id] = tool

        # Index endpoints
        for endpoint in tool.endpoints:
            self._endpoint_index[endpoint.id] = endpoint
            self._domains.add(endpoint.domain)

    def get_tool(self, tool_id: str) -> Optional[Tool]:
        """Get tool by ID.

        Args:
            tool_id: Tool identifier

        Returns:
            Tool if found, None otherwise
        """
        return self.tools.get(tool_id)

    def get_endpoint(self, endpoint_id: str) -> Optional[Endpoint]:
        """Get endpoint by ID from the index.

        Args:
            endpoint_id: Endpoint identifier

        Returns:
            Endpoint if found, None otherwise
        """
        return self._endpoint_index.get(endpoint_id)

    def get_endpoints_by_domain(self, domain: str) -> List[Endpoint]:
        """Get all endpoints in a domain.

        Args:
            domain: Domain name to filter by

        Returns:
            List of endpoints in the domain
        """
        return [
            endpoint
            for endpoint in self._endpoint_index.values()
            if endpoint.domain == domain
        ]

    @property
    def domains(self) -> List[str]:
        """Get sorted list of all domains."""
        return sorted(self._domains)

    @property
    def endpoint_count(self) -> int:
        """Get total number of endpoints across all tools."""
        return len(self._endpoint_index)

    @property
    def tool_count(self) -> int:
        """Get total number of tools."""
        return len(self.tools)
