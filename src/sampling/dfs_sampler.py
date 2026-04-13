"""DFS-based tool chain sampler.

This module provides a depth-first search based sampler for generating
valid tool chains from the graph, with support for backtracking and
diversity weighting.
"""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from src.graph import GraphClient, SAME_DOMAIN, SEMANTICALLY_SIMILAR
from src.sampling.constraints import ChainPattern, SamplingConstraints
from src.sampling.patterns import (
    BaseChainPattern,
    BranchingChain,
    IterativeChain,
    ParallelChain,
    SequentialChain,
)


@dataclass
class DFSState:
    """State for DFS traversal.

    Attributes:
        current: Current endpoint ID
        path: List of endpoint IDs in current path
        visited: Set of visited endpoint IDs
        depth: Current depth in search
    """

    current: str
    path: List[str]
    visited: Set[str]
    depth: int


class DFSSampler:
    """DFS-based tool chain sampler.

    Uses depth-first search with backtracking to find valid tool chains
    that satisfy the given constraints. Supports diversity weighting to
    prioritize variety in tool selection.

    Attributes:
        client: GraphClient instance for graph queries
        constraints: SamplingConstraints defining valid chains
        max_start_candidates: Maximum starting points to try
        max_neighbors: Maximum neighbors to consider per step

    Example:
        >>> sampler = DFSSampler(client, constraints)
        >>> chain = sampler.sample()
        >>> if chain:
        ...     print(f"Found chain with {len(chain)} endpoints")
    """

    def __init__(
        self,
        client: GraphClient,
        constraints: SamplingConstraints,
        max_start_candidates: int = 10,
        max_neighbors: int = 10,
        diversity_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """Initialize the DFS sampler.

        Args:
            client: GraphClient instance for graph queries
            constraints: SamplingConstraints defining valid chains
            max_start_candidates: Maximum starting points to try (default 10)
            max_neighbors: Maximum neighbors to consider per step (default 10)
            diversity_weights: Optional dict mapping endpoint_id to weight.
                              Higher weights increase selection probability.
        """
        self.client = client
        self.constraints = constraints
        self.max_start_candidates = max_start_candidates
        self.max_neighbors = max_neighbors
        self.diversity_weights = diversity_weights or {}

    def sample(self, max_attempts: int = 100) -> Optional[List[Dict[str, Any]]]:
        """Sample a valid tool chain using DFS.

        Args:
            max_attempts: Maximum DFS iterations before giving up

        Returns:
            List of endpoint dictionaries if successful, None if no valid chain found.
            Each dict contains: endpoint_id, tool_id, domain, completeness_score, name
        """
        # Get starting candidates
        start_candidates = self._get_start_candidates()

        if not start_candidates:
            return None

        attempts = 0

        # Try each starting point
        for start in start_candidates:
            if attempts >= max_attempts:
                break

            # Initialize DFS stack
            stack = [
                DFSState(
                    current=start,
                    path=[start],
                    visited={start},
                    depth=1,
                )
            ]

            while stack and attempts < max_attempts:
                attempts += 1
                state = stack.pop()

                # Check if we have a valid chain (at or above min_steps)
                if state.depth >= self.constraints.min_steps:
                    if self._is_valid_chain(state.path):
                        return self._build_chain_data(state.path)

                # Expand if not at max depth
                if state.depth < self.constraints.max_steps:
                    neighbors = self._get_neighbors(state.current, state.visited)

                    for neighbor in neighbors:
                        new_state = DFSState(
                            current=neighbor,
                            path=state.path + [neighbor],
                            visited=state.visited | {neighbor},
                            depth=state.depth + 1,
                        )
                        stack.append(new_state)

        return None

    def _get_start_candidates(self) -> List[str]:
        """Get valid starting endpoints for DFS.

        Returns:
            List of endpoint IDs that meet constraints, sorted by diversity weight.
        """
        # Get all endpoints meeting completeness threshold
        candidates = self.client.filter_by_completeness(self.constraints.min_completeness)

        # Filter by excluded domains
        if self.constraints.excluded_domains:
            filtered = []
            for ep_id in candidates:
                ep_data = self.client.get_endpoint_by_id(ep_id)
                if ep_data:
                    domain = ep_data.get("domain", "")
                    if domain not in self.constraints.excluded_domains:
                        filtered.append(ep_id)
            candidates = filtered

        # Filter by excluded tools
        if self.constraints.excluded_tools:
            filtered = []
            for ep_id in candidates:
                ep_data = self.client.get_endpoint_by_id(ep_id)
                if ep_data:
                    tool_id = ep_data.get("tool_id", "")
                    if tool_id not in self.constraints.excluded_tools:
                        filtered.append(ep_id)
            candidates = filtered

        # Apply diversity weighting and limit
        candidates = self._apply_diversity_weight(candidates)

        return candidates[: self.max_start_candidates]

    def _get_neighbors(self, endpoint_id: str, visited: Set[str]) -> List[str]:
        """Get valid neighbor endpoints for DFS expansion.

        Args:
            endpoint_id: Current endpoint ID
            visited: Set of already-visited endpoint IDs

        Returns:
            List of neighbor endpoint IDs, filtered and weighted.
        """
        # Get neighbors via both edge types
        same_domain_neighbors = self.client.get_neighbors(endpoint_id, edge_type=SAME_DOMAIN)
        semantic_neighbors = self.client.get_neighbors(endpoint_id, edge_type=SEMANTICALLY_SIMILAR)

        # Combine and deduplicate
        all_neighbors = list(set(same_domain_neighbors + semantic_neighbors))

        # Filter out visited
        candidates = [n for n in all_neighbors if n not in visited]

        # Filter by completeness
        candidates = [
            n for n in candidates
            if self._meets_completeness(n)
        ]

        # Filter by excluded domains
        if self.constraints.excluded_domains:
            candidates = [
                n for n in candidates
                if not self._in_excluded_domain(n)
            ]

        # Filter by excluded tools
        if self.constraints.excluded_tools:
            candidates = [
                n for n in candidates
                if not self._uses_excluded_tool(n)
            ]

        # Apply diversity weighting
        candidates = self._apply_diversity_weight(candidates)

        return candidates[: self.max_neighbors]

    def _is_valid_chain(self, path: List[str]) -> bool:
        """Check if current path forms a valid chain.

        Args:
            path: List of endpoint IDs

        Returns:
            True if chain satisfies all constraints
        """
        chain_data = self._build_chain_data(path)

        if not chain_data:
            return False

        is_valid, _ = self.constraints.validate_chain(chain_data)
        return is_valid

    def _build_chain_data(self, path: List[str]) -> List[Dict[str, Any]]:
        """Convert path of endpoint IDs to chain data.

        Args:
            path: List of endpoint IDs

        Returns:
            List of endpoint dictionaries with full attributes
        """
        chain = []
        for ep_id in path:
            ep_data = self.client.get_endpoint_by_id(ep_id)
            if ep_data:
                chain.append({
                    "endpoint_id": ep_id,
                    "tool_id": ep_data.get("tool_id", ""),
                    "domain": ep_data.get("domain", ""),
                    "completeness_score": ep_data.get("completeness_score", 0.0),
                    "name": ep_data.get("name", ""),
                })
        return chain

    def _apply_diversity_weight(self, candidates: List[str]) -> List[str]:
        """Weight candidates to prioritize diversity.

        Combines external diversity_weights (from DiversitySteeringAgent) with
        internal tool-frequency weighting. Higher total weight = higher priority.

        Args:
            candidates: List of endpoint IDs

        Returns:
            List of endpoint IDs sorted by combined weight (highest first)
        """
        if not candidates:
            return candidates

        # Count tools among candidates (for internal diversity)
        tool_counts: Dict[str, int] = {}
        for ep_id in candidates:
            ep_data = self.client.get_endpoint_by_id(ep_id)
            if ep_data:
                tool_id = ep_data.get("tool_id", "")
                tool_counts[tool_id] = tool_counts.get(tool_id, 0) + 1

        def weight(ep_id: str) -> float:
            # External weight from diversity_weights (default 1.0)
            external_weight = self.diversity_weights.get(ep_id, 1.0)

            # Internal weight: inverse tool frequency
            ep_data = self.client.get_endpoint_by_id(ep_id)
            tool_id = ep_data.get("tool_id", "") if ep_data else ""
            internal_weight = 1.0 / (tool_counts.get(tool_id, 1) + 1)

            # Combine: multiply external weight by internal weight
            return external_weight * internal_weight

        # Shuffle first for randomness, then sort by weight (highest first)
        shuffled = list(candidates)
        random.shuffle(shuffled)

        return sorted(shuffled, key=weight, reverse=True)

    def _meets_completeness(self, endpoint_id: str) -> bool:
        """Check if endpoint meets completeness threshold.

        Args:
            endpoint_id: Endpoint ID to check

        Returns:
            True if completeness >= min_completeness
        """
        ep_data = self.client.get_endpoint_by_id(endpoint_id)
        if not ep_data:
            return False
        completeness = ep_data.get("completeness_score", 0.0)
        return completeness >= self.constraints.min_completeness

    def _in_excluded_domain(self, endpoint_id: str) -> bool:
        """Check if endpoint is in an excluded domain.

        Args:
            endpoint_id: Endpoint ID to check

        Returns:
            True if endpoint's domain is in excluded_domains
        """
        ep_data = self.client.get_endpoint_by_id(endpoint_id)
        if not ep_data:
            return False
        domain = ep_data.get("domain", "")
        return domain in self.constraints.excluded_domains

    def _uses_excluded_tool(self, endpoint_id: str) -> bool:
        """Check if endpoint uses an excluded tool.

        Args:
            endpoint_id: Endpoint ID to check

        Returns:
            True if endpoint's tool_id is in excluded_tools
        """
        ep_data = self.client.get_endpoint_by_id(endpoint_id)
        if not ep_data:
            return False
        tool_id = ep_data.get("tool_id", "")
        return tool_id in self.constraints.excluded_tools

    def _build_pattern(self, chain: List[Dict[str, Any]]) -> BaseChainPattern:
        """Convert chain to pattern based on constraints.pattern.

        Args:
            chain: List of endpoint dictionaries with 'endpoint_id' key

        Returns:
            Appropriate pattern instance based on constraints.pattern
        """
        # Extract endpoint IDs
        endpoint_ids = [ep["endpoint_id"] for ep in chain]

        # Handle empty chain
        if len(endpoint_ids) == 0:
            return SequentialChain(steps=[])

        # Chains with fewer than 3 endpoints fall back to sequential
        if len(endpoint_ids) < 3:
            return SequentialChain(steps=endpoint_ids)

        # Get pattern type (default to SEQUENTIAL)
        pattern_type = self.constraints.pattern

        if pattern_type == ChainPattern.PARALLEL:
            # First 2 are parallel, rest are then_steps
            return ParallelChain(
                parallel_steps=endpoint_ids[:2],
                then_steps=endpoint_ids[2:],
            )

        elif pattern_type == ChainPattern.BRANCHING:
            # First is start, last is merge, middle is branch
            return BranchingChain(
                start=endpoint_ids[0],
                branches={"main": endpoint_ids[1:-1]},
                merge=endpoint_ids[-1],
            )

        elif pattern_type == ChainPattern.ITERATIVE:
            # First is start, last is end, middle determines loop
            middle = endpoint_ids[1:-1]
            return IterativeChain(
                start=endpoint_ids[0],
                loop_step=middle[0],  # Use first middle endpoint as loop step
                loop_count=len(middle),  # Number of middle endpoints = iterations
                end=endpoint_ids[-1],
            )

        # Default: SEQUENTIAL or None
        return SequentialChain(steps=endpoint_ids)
