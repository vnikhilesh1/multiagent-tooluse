"""Diversity Steering Agent for dataset coverage.

This agent tracks tool and domain usage across conversations
and steers sampling toward underrepresented areas.
"""

import hashlib
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.context import ConversationContext
    from src.sampling.constraints import SamplingConstraints


@dataclass
class DiversityMetrics:
    """Computed diversity statistics.

    Attributes:
        tool_entropy: Shannon entropy of tool distribution (-Σ p_i * log(p_i))
        domain_entropy: Shannon entropy of domain distribution
        unique_tools: Number of unique tools used
        unique_domains: Number of unique domains used
        unique_pairs: Number of unique tool pairs
        unique_patterns: Number of unique chain patterns
        total_conversations: Total conversations recorded
        pair_ratio: Ratio of unique pairs to possible pairs
    """
    tool_entropy: float
    domain_entropy: float
    unique_tools: int
    unique_domains: int
    unique_pairs: int
    unique_patterns: int
    total_conversations: int
    pair_ratio: float


@dataclass
class DiversityTracker:
    """Tracks usage statistics for diversity steering.

    Maintains counters for tools, domains, and tool pairs
    to enable inverse-frequency weighting during sampling.

    Attributes:
        tool_counts: Counter mapping tool_id to usage count
        domain_counts: Counter mapping domain to usage count
        tool_pair_counts: Counter mapping (tool1, tool2) to co-occurrence count
        pattern_hashes: Set of unique chain pattern hashes

    Example:
        >>> tracker = DiversityTracker()
        >>> tracker.record_tool("weather_api")
        >>> tracker.record_domain("weather")
        >>> tracker.tool_counts["weather_api"]
        1
    """
    tool_counts: Counter = field(default_factory=Counter)
    domain_counts: Counter = field(default_factory=Counter)
    tool_pair_counts: Counter = field(default_factory=Counter)
    pattern_hashes: Set[str] = field(default_factory=set)
    _total_conversations: int = field(default=0, repr=False)

    def record_tool(self, tool_id: str) -> None:
        """Record a tool usage.

        Args:
            tool_id: The tool identifier to record
        """
        self.tool_counts[tool_id] += 1

    def record_domain(self, domain: str) -> None:
        """Record a domain usage.

        Args:
            domain: The domain to record
        """
        self.domain_counts[domain] += 1

    def record_tool_pair(self, tool1: str, tool2: str) -> None:
        """Record a tool pair co-occurrence.

        Args:
            tool1: First tool identifier
            tool2: Second tool identifier
        """
        # Normalize order for consistent counting
        pair = tuple(sorted([tool1, tool2]))
        self.tool_pair_counts[pair] += 1

    def record_pattern(self, tool_ids: List[str]) -> None:
        """Record a chain pattern by its hash.

        Args:
            tool_ids: List of tool IDs in the chain
        """
        pattern_hash = self._compute_pattern_hash(tool_ids)
        self.pattern_hashes.add(pattern_hash)

    def _compute_pattern_hash(self, tool_ids: List[str]) -> str:
        """Compute hash of sorted tool list.

        Args:
            tool_ids: List of tool IDs

        Returns:
            MD5 hash of sorted, joined tool IDs
        """
        sorted_ids = sorted(tool_ids)
        pattern_str = ",".join(sorted_ids)
        return hashlib.md5(pattern_str.encode()).hexdigest()[:16]

    def increment_conversations(self) -> None:
        """Increment the total conversation count."""
        self._total_conversations += 1

    @property
    def total_conversations(self) -> int:
        """Get total number of recorded conversations."""
        return self._total_conversations


class DiversitySteeringAgent:
    """Agent that steers sampling toward underrepresented tools/domains.

    Tracks usage patterns across generated conversations and provides:
    - Sampling constraints that favor underrepresented domains
    - Diversity weights for endpoint selection
    - Metrics for evaluating dataset diversity

    Attributes:
        tracker: DiversityTracker instance
        endpoint_to_tool: Mapping from endpoint_id to tool_id
        endpoint_to_domain: Mapping from endpoint_id to domain

    Example:
        >>> agent = DiversitySteeringAgent()
        >>> agent.record(context, endpoints)
        >>> weights = agent.get_diversity_weights()
        >>> constraints = agent.suggest_constraints()
    """

    def __init__(self) -> None:
        """Initialize the diversity steering agent."""
        self.tracker = DiversityTracker()
        self.endpoint_to_tool: Dict[str, str] = {}
        self.endpoint_to_domain: Dict[str, str] = {}

    def register_endpoint(
        self,
        endpoint_id: str,
        tool_id: str,
        domain: Optional[str] = None,
    ) -> None:
        """Register an endpoint's tool and domain mappings.

        Args:
            endpoint_id: The endpoint identifier
            tool_id: The parent tool identifier
            domain: Optional domain category
        """
        self.endpoint_to_tool[endpoint_id] = tool_id
        if domain:
            self.endpoint_to_domain[endpoint_id] = domain

    def record(
        self,
        context: "ConversationContext",
        endpoints: Optional[Dict[str, any]] = None,
    ) -> None:
        """Record a completed conversation for diversity tracking.

        Updates all counters based on the tools and domains used
        in the conversation.

        Args:
            context: Completed conversation context
            endpoints: Optional dict of endpoint_id -> Endpoint for lookups
        """
        # Get tool chain from context
        tool_chain = context.tool_chain or []

        # Collect tools and domains used
        tools_used: Set[str] = set()
        domains_used: Set[str] = set()

        for endpoint_id in tool_chain:
            # Get tool_id
            tool_id = self.endpoint_to_tool.get(endpoint_id)
            if tool_id:
                tools_used.add(tool_id)
                self.tracker.record_tool(tool_id)

            # Get domain
            domain = self.endpoint_to_domain.get(endpoint_id)
            if domain:
                domains_used.add(domain)
                self.tracker.record_domain(domain)

        # Record tool pairs
        tools_list = list(tools_used)
        for i in range(len(tools_list)):
            for j in range(i + 1, len(tools_list)):
                self.tracker.record_tool_pair(tools_list[i], tools_list[j])

        # Record pattern hash
        self.tracker.record_pattern(list(tools_used))

        # Increment conversation count
        self.tracker.increment_conversations()

    def suggest_constraints(self) -> "SamplingConstraints":
        """Suggest sampling constraints to improve diversity.

        Finds underrepresented domains and returns constraints
        that steer toward them.

        Returns:
            SamplingConstraints with required_domains set
        """
        from src.sampling.constraints import SamplingConstraints

        constraints = SamplingConstraints()

        # Get all known domains (registered + recorded)
        all_domains = set(self.endpoint_to_domain.values())

        if all_domains:
            # Find domain with lowest count (0 for unrecorded domains)
            min_domain = min(
                all_domains,
                key=lambda d: self.tracker.domain_counts.get(d, 0)
            )
            constraints.required_domains = [min_domain]

        # Require multi-tool if we have low pair diversity
        if self.tracker.total_conversations > 5:
            metrics = self.compute_metrics()
            if metrics.pair_ratio < 0.3:
                constraints.require_multi_tool = True

        return constraints

    def get_diversity_weights(self) -> Dict[str, float]:
        """Get inverse-frequency weights for all known endpoints.

        Endpoints from underused tools get higher weights.
        Weight formula: max_count / (count + 1)

        Returns:
            Dict mapping endpoint_id to weight (higher = prioritize)
        """
        weights: Dict[str, float] = {}

        if not self.tracker.tool_counts:
            # No data yet, equal weights
            return {eid: 1.0 for eid in self.endpoint_to_tool}

        max_count = max(self.tracker.tool_counts.values())

        for endpoint_id, tool_id in self.endpoint_to_tool.items():
            tool_count = self.tracker.tool_counts.get(tool_id, 0)
            # Inverse frequency: higher weight for less-used tools
            weight = max_count / (tool_count + 1)
            weights[endpoint_id] = weight

        return weights

    def compute_metrics(self) -> DiversityMetrics:
        """Compute diversity statistics.

        Returns:
            DiversityMetrics with entropy and coverage stats
        """
        # Tool entropy
        tool_entropy = self._compute_entropy(self.tracker.tool_counts)

        # Domain entropy
        domain_entropy = self._compute_entropy(self.tracker.domain_counts)

        # Unique counts
        unique_tools = len(self.tracker.tool_counts)
        unique_domains = len(self.tracker.domain_counts)
        unique_pairs = len(self.tracker.tool_pair_counts)
        unique_patterns = len(self.tracker.pattern_hashes)

        # Pair ratio: unique pairs / possible pairs
        possible_pairs = unique_tools * (unique_tools - 1) // 2 if unique_tools > 1 else 1
        pair_ratio = unique_pairs / possible_pairs if possible_pairs > 0 else 0.0

        return DiversityMetrics(
            tool_entropy=tool_entropy,
            domain_entropy=domain_entropy,
            unique_tools=unique_tools,
            unique_domains=unique_domains,
            unique_pairs=unique_pairs,
            unique_patterns=unique_patterns,
            total_conversations=self.tracker.total_conversations,
            pair_ratio=pair_ratio,
        )

    def _compute_entropy(self, counts: Counter) -> float:
        """Compute Shannon entropy from counts.

        Formula: H = -Σ(p_i * log(p_i))

        Args:
            counts: Counter of item frequencies

        Returns:
            Entropy value (higher = more even distribution)
        """
        if not counts:
            return 0.0

        total = sum(counts.values())
        if total == 0:
            return 0.0

        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log(p)

        return entropy

    def reset(self) -> None:
        """Reset all tracking counters."""
        self.tracker = DiversityTracker()

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"DiversitySteeringAgent("
            f"tools={len(self.tracker.tool_counts)}, "
            f"domains={len(self.tracker.domain_counts)}, "
            f"conversations={self.tracker.total_conversations})"
        )
