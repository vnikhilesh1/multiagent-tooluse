"""Sampling Constraints Model for tool chain validation.

This module provides the SamplingConstraints Pydantic model that defines
constraints for valid tool chains, and methods to validate chains against
those constraints.
"""

from enum import Enum
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator


class ChainPattern(str, Enum):
    """Enumeration of supported chain patterns."""

    SEQUENTIAL = "sequential"  # A → B → C (linear)
    PARALLEL = "parallel"      # A → (B, C) → D (concurrent calls)
    BRANCHING = "branching"    # A → B or A → C (conditional)
    ITERATIVE = "iterative"    # A → B → A (loops)


class SamplingConstraints(BaseModel):
    """Pydantic model defining constraints for valid tool chains.

    Attributes:
        min_steps: Minimum chain length (default 2, must be >= 1)
        max_steps: Maximum chain length (default 5, must be >= min_steps)
        required_domains: Domains that must be included in the chain
        excluded_domains: Domains to avoid in the chain
        required_tools: Tools that must appear in the chain
        excluded_tools: Tools to exclude from the chain
        pattern: Chain execution pattern (default SEQUENTIAL)
        min_completeness: Minimum endpoint completeness score (0.0-1.0, default 0.5)
        require_multi_tool: Whether chain must use >= 2 distinct tools (default False)

    Example:
        >>> constraints = SamplingConstraints(
        ...     min_steps=3,
        ...     max_steps=6,
        ...     required_domains=["weather"],
        ...     min_completeness=0.7,
        ... )
        >>> is_valid, errors = constraints.validate_chain(chain)
    """

    min_steps: int = Field(
        default=2,
        ge=1,
        description="Minimum chain length (must be >= 1)"
    )
    max_steps: int = Field(
        default=5,
        description="Maximum chain length"
    )
    required_domains: List[str] = Field(
        default_factory=list,
        description="Domains that must be included in the chain"
    )
    excluded_domains: List[str] = Field(
        default_factory=list,
        description="Domains to avoid in the chain"
    )
    required_tools: List[str] = Field(
        default_factory=list,
        description="Tools that must appear in the chain"
    )
    excluded_tools: List[str] = Field(
        default_factory=list,
        description="Tools to exclude from the chain"
    )
    pattern: ChainPattern = Field(
        default=ChainPattern.SEQUENTIAL,
        description="Chain execution pattern"
    )
    min_completeness: float = Field(
        default=0.5,
        description="Minimum endpoint completeness score (0.0-1.0)"
    )
    require_multi_tool: bool = Field(
        default=False,
        description="Whether chain must use >= 2 distinct tools"
    )

    @field_validator('min_completeness')
    @classmethod
    def validate_completeness_range(cls, v: float) -> float:
        """Validate that min_completeness is between 0.0 and 1.0."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("min_completeness must be between 0.0 and 1.0")
        return v

    @model_validator(mode='after')
    def validate_step_range(self) -> 'SamplingConstraints':
        """Validate that max_steps >= min_steps."""
        if self.max_steps < self.min_steps:
            raise ValueError(
                f"max_steps ({self.max_steps}) must be >= min_steps ({self.min_steps})"
            )
        return self

    def validate_chain(self, chain: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """Check if a chain satisfies all constraints.

        Args:
            chain: List of endpoint dictionaries, each containing:
                - endpoint_id: str
                - tool_id: str
                - domain: str
                - completeness_score: float

        Returns:
            Tuple of (is_valid: bool, errors: List[str])
            - is_valid: True if chain satisfies all constraints
            - errors: List of human-readable error messages (empty if valid)
        """
        errors: List[str] = []

        # 1. Check chain length
        chain_length = len(chain)
        if chain_length < self.min_steps:
            errors.append(
                f"Chain length {chain_length} is below min_steps {self.min_steps}"
            )
        if chain_length > self.max_steps:
            errors.append(
                f"Chain length {chain_length} exceeds max_steps {self.max_steps}"
            )

        # Extract domains and tools from chain
        chain_domains = set()
        chain_tools = set()
        for endpoint in chain:
            domain = endpoint.get("domain", "")
            tool_id = endpoint.get("tool_id", "")
            if domain:
                chain_domains.add(domain)
            if tool_id:
                chain_tools.add(tool_id)

        # 2. Check required domains
        for required_domain in self.required_domains:
            if required_domain not in chain_domains:
                errors.append(
                    f"Required domain '{required_domain}' not found in chain"
                )

        # 3. Check excluded domains
        for endpoint in chain:
            domain = endpoint.get("domain", "")
            if domain in self.excluded_domains:
                errors.append(
                    f"Endpoint '{endpoint.get('endpoint_id', 'unknown')}' uses "
                    f"excluded domain '{domain}'"
                )

        # 4. Check required tools
        for required_tool in self.required_tools:
            if required_tool not in chain_tools:
                errors.append(
                    f"Required tool '{required_tool}' not found in chain"
                )

        # 5. Check excluded tools
        for endpoint in chain:
            tool_id = endpoint.get("tool_id", "")
            if tool_id in self.excluded_tools:
                errors.append(
                    f"Endpoint '{endpoint.get('endpoint_id', 'unknown')}' uses "
                    f"excluded tool '{tool_id}'"
                )

        # 6. Check completeness scores
        for endpoint in chain:
            completeness = endpoint.get("completeness_score", 0.0)
            if completeness < self.min_completeness:
                errors.append(
                    f"Endpoint '{endpoint.get('endpoint_id', 'unknown')}' has "
                    f"completeness {completeness} below minimum {self.min_completeness}"
                )

        # 7. Check multi-tool requirement
        if self.require_multi_tool and len(chain_tools) < 2:
            errors.append(
                f"Chain uses only {len(chain_tools)} tool(s), "
                f"but require_multi_tool requires >= 2 distinct tools"
            )

        return (len(errors) == 0, errors)
