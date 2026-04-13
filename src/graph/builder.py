"""Graph Builder - Orchestrates graph construction from ToolRegistry.

This module provides the ToolGraphBuilder class for building the tool graph
from a ToolRegistry, including tools, endpoints, and their relationships.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from src.registry.models import Endpoint, Tool, ToolRegistry

from .client import GraphClient
from .constants import DOMAIN, ENDPOINT, HAS_ENDPOINT, IN_DOMAIN, SAME_DOMAIN, SEMANTICALLY_SIMILAR, TOOL

logger = logging.getLogger(__name__)


class DuplicateNodeError(Exception):
    """Exception raised when attempting to add a duplicate node.

    Attributes:
        node_id: The ID of the duplicate node
        node_type: The type of the node (e.g., "Tool", "Endpoint")
    """

    def __init__(self, node_id: str, node_type: str) -> None:
        self.node_id = node_id
        self.node_type = node_type
        super().__init__(f"Duplicate {node_type} node: {node_id}")


@dataclass
class BuildStats:
    """Statistics from a graph build operation.

    Attributes:
        tools_added: Number of tool nodes successfully added
        tools_skipped: Number of tool nodes skipped (already existed)
        endpoints_added: Number of endpoint nodes successfully added
        endpoints_skipped: Number of endpoint nodes skipped (already existed)
        edges_added: Number of HAS_ENDPOINT edges created
        domains_added: Number of domain nodes successfully added
        domains_skipped: Number of domain nodes skipped (already existed)
        in_domain_edges_added: Number of IN_DOMAIN edges created
        duration_seconds: Total time taken for the build operation
    """

    tools_added: int = 0
    tools_skipped: int = 0
    endpoints_added: int = 0
    endpoints_skipped: int = 0
    edges_added: int = 0
    domains_added: int = 0
    domains_skipped: int = 0
    in_domain_edges_added: int = 0
    duration_seconds: float = 0.0


@dataclass
class SimilarityStats:
    """Statistics from semantic similarity edge creation.

    Attributes:
        edges_created: Number of SEMANTICALLY_SIMILAR edges created
        edges_skipped_threshold: Number of pairs skipped due to low similarity
        edges_skipped_same_domain: Number of pairs skipped due to existing SAME_DOMAIN edge
        mean_similarity: Mean similarity score of created edges
        max_similarity: Maximum similarity score of created edges
        min_similarity: Minimum similarity score of created edges
    """

    edges_created: int = 0
    edges_skipped_threshold: int = 0
    edges_skipped_same_domain: int = 0
    mean_similarity: float = 0.0
    max_similarity: float = 0.0
    min_similarity: float = 0.0


class PydanticJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles Pydantic models."""

    def default(self, obj: Any) -> Any:
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "dict"):
            return obj.dict()
        if hasattr(obj, "value"):  # For enums
            return obj.value
        return super().default(obj)


def sanitize_domain_id(domain_name: str) -> str:
    """Convert a domain name to a valid node ID.

    Converts the domain name to lowercase, replaces non-alphanumeric
    characters with underscores, and prepends "domain_".

    Args:
        domain_name: The domain name to sanitize

    Returns:
        A sanitized node ID in the format "domain_{sanitized_name}"

    Example:
        >>> sanitize_domain_id("Weather API")
        'domain_weather_api'
        >>> sanitize_domain_id("Finance & Banking")
        'domain_finance___banking'
    """
    # Convert to lowercase
    sanitized = domain_name.lower()
    # Replace non-alphanumeric characters with underscores
    sanitized = re.sub(r'[^a-z0-9]', '_', sanitized)
    return f"domain_{sanitized}"


def compute_pairwise_similarity(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity between embedding vectors.

    Uses vectorized numpy operations for efficient computation.
    Normalizes each embedding to unit length, then computes the
    dot product matrix.

    Args:
        embeddings: Matrix of shape (n, dim) where each row is an embedding

    Returns:
        Similarity matrix of shape (n, n) where element (i, j) is the
        cosine similarity between embeddings[i] and embeddings[j]

    Example:
        >>> embeddings = np.array([[1, 0], [1, 0], [0, 1]])
        >>> sim = compute_pairwise_similarity(embeddings)
        >>> sim[0, 1]  # Identical vectors
        1.0
        >>> sim[0, 2]  # Orthogonal vectors
        0.0
    """
    # Normalize each row to unit length
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-10)  # Avoid division by zero

    # Similarity matrix via dot product of normalized vectors
    return normalized @ normalized.T


class ToolGraphBuilder:
    """Orchestrates graph construction from ToolRegistry.

    This class provides methods to build the tool graph from a ToolRegistry,
    including adding tools, endpoints, and their relationships. It supports
    both full builds (clearing the graph first) and incremental builds
    (adding to existing graph).

    Attributes:
        client: The GraphClient instance to build into
        show_progress: Whether to display tqdm progress bars

    Example:
        >>> client = GraphClient(GraphConfig())
        >>> builder = ToolGraphBuilder(client, show_progress=True)
        >>> stats = builder.build_from_registry(registry)
        >>> print(f"Added {stats.tools_added} tools")
    """

    def __init__(self, client: GraphClient, show_progress: bool = True) -> None:
        """Initialize the ToolGraphBuilder.

        Args:
            client: GraphClient instance to build the graph into
            show_progress: Whether to display tqdm progress bars (default True)
        """
        self.client = client
        self.show_progress = show_progress

    def build_from_registry(
        self, registry: ToolRegistry, incremental: bool = False, build_domains: bool = True
    ) -> BuildStats:
        """Build the graph from a ToolRegistry.

        This is the main entry point for building the graph. It adds all tools
        and endpoints from the registry, creating HAS_ENDPOINT edges between them.
        Optionally also creates Domain nodes and IN_DOMAIN edges.

        Args:
            registry: The ToolRegistry containing tools and endpoints to add
            incremental: If False (default), clears the graph before building.
                        If True, adds to existing graph, skipping duplicates.
            build_domains: If True (default), also build Domain nodes and IN_DOMAIN edges.

        Returns:
            BuildStats with counts of added/skipped nodes and edges
        """
        start_time = time.time()
        stats = BuildStats()

        # Clear graph if not incremental
        if not incremental:
            self.client.clear_graph()

        # Add all tools
        tools_iter = registry.tools.values()
        if self.show_progress:
            tools_iter = tqdm(
                list(tools_iter), desc="Adding tools...", unit="tool"
            )

        for tool in tools_iter:
            if self.add_tool(tool, skip_if_exists=True):
                stats.tools_added += 1
            else:
                stats.tools_skipped += 1

        # Add all endpoints (from tool.endpoints, not registry.endpoints)
        # This ensures we add endpoints that belong to the tools we just added
        all_endpoints: List[Endpoint] = []
        for tool in registry.tools.values():
            all_endpoints.extend(tool.endpoints)

        endpoints_iter = all_endpoints
        if self.show_progress:
            endpoints_iter = tqdm(
                list(endpoints_iter), desc="Adding endpoints...", unit="endpoint"
            )

        for endpoint in endpoints_iter:
            if self.add_endpoint(endpoint, skip_if_exists=True):
                stats.endpoints_added += 1
            else:
                stats.endpoints_skipped += 1

        # Create HAS_ENDPOINT edges
        edges_to_create = [
            (endpoint.tool_id, endpoint.id) for endpoint in all_endpoints
        ]

        edges_iter = edges_to_create
        if self.show_progress:
            edges_iter = tqdm(
                list(edges_iter), desc="Creating HAS_ENDPOINT edges...", unit="edge"
            )

        for tool_id, endpoint_id in edges_iter:
            if self.add_has_endpoint_edge(tool_id, endpoint_id):
                stats.edges_added += 1

        # Build domain nodes and IN_DOMAIN edges if requested
        if build_domains:
            domains_added, domains_skipped = self.build_domain_nodes(registry)
            stats.domains_added = domains_added
            stats.domains_skipped = domains_skipped

            stats.in_domain_edges_added = self.build_in_domain_edges(registry)

        stats.duration_seconds = time.time() - start_time

        logger.info(
            f"Build complete: {stats.tools_added} tools, "
            f"{stats.endpoints_added} endpoints, "
            f"{stats.edges_added} edges, "
            f"{stats.domains_added} domains, "
            f"{stats.in_domain_edges_added} IN_DOMAIN edges "
            f"in {stats.duration_seconds:.2f}s"
        )

        return stats

    def add_tool(self, tool: Tool, skip_if_exists: bool = False) -> bool:
        """Add a single tool node to the graph.

        Args:
            tool: The Tool to add
            skip_if_exists: If True, silently skip if tool already exists.
                           If False, raise DuplicateNodeError.

        Returns:
            True if the tool was added, False if it was skipped

        Raises:
            DuplicateNodeError: If tool exists and skip_if_exists is False
        """
        if self.client.node_exists(tool.id):
            if skip_if_exists:
                return False
            raise DuplicateNodeError(tool.id, "Tool")

        self.client.add_node(
            node_id=tool.id,
            node_type=TOOL,
            name=tool.name,
            category=tool.category,
            description=tool.description,
            completeness_score=tool.completeness_score,
            api_host=tool.api_host,
        )

        return True

    def add_endpoint(self, endpoint: Endpoint, skip_if_exists: bool = False) -> bool:
        """Add a single endpoint node to the graph.

        Args:
            endpoint: The Endpoint to add
            skip_if_exists: If True, silently skip if endpoint already exists.
                           If False, raise DuplicateNodeError.

        Returns:
            True if the endpoint was added, False if it was skipped

        Raises:
            DuplicateNodeError: If endpoint exists and skip_if_exists is False
        """
        if self.client.node_exists(endpoint.id):
            if skip_if_exists:
                return False
            raise DuplicateNodeError(endpoint.id, "Endpoint")

        # Serialize parameters to JSON
        parameters_json = json.dumps(
            [p.model_dump() for p in endpoint.parameters],
            cls=PydanticJSONEncoder
        )

        self.client.add_node(
            node_id=endpoint.id,
            node_type=ENDPOINT,
            tool_id=endpoint.tool_id,
            name=endpoint.name,
            method=endpoint.method,
            path=endpoint.path,
            description=endpoint.description,
            domain=endpoint.domain,
            completeness_score=endpoint.completeness_score,
            parameter_count=len(endpoint.parameters),
            parameters_json=parameters_json,
        )

        return True

    def add_has_endpoint_edge(self, tool_id: str, endpoint_id: str) -> bool:
        """Create a HAS_ENDPOINT edge from a tool to an endpoint.

        Args:
            tool_id: The source tool node ID
            endpoint_id: The target endpoint node ID

        Returns:
            True if the edge was created, False if it already existed
        """
        if self.client.edge_exists(tool_id, endpoint_id, HAS_ENDPOINT):
            return False

        self.client.add_edge(tool_id, endpoint_id, HAS_ENDPOINT)
        return True

    def add_domain(self, domain_name: str, skip_if_exists: bool = False) -> bool:
        """Add a single domain node to the graph.

        Args:
            domain_name: The domain name to add
            skip_if_exists: If True, silently skip if domain already exists.
                           If False, raise DuplicateNodeError.

        Returns:
            True if the domain was added, False if it was skipped

        Raises:
            DuplicateNodeError: If domain exists and skip_if_exists is False
        """
        domain_id = sanitize_domain_id(domain_name)

        if self.client.node_exists(domain_id):
            if skip_if_exists:
                return False
            raise DuplicateNodeError(domain_id, "Domain")

        self.client.add_node(
            node_id=domain_id,
            node_type=DOMAIN,
            name=domain_name,
        )

        return True

    def add_in_domain_edge(self, endpoint_id: str, domain_name: str) -> bool:
        """Create an IN_DOMAIN edge from an endpoint to a domain.

        Args:
            endpoint_id: The source endpoint node ID
            domain_name: The target domain name (will be converted to domain node ID)

        Returns:
            True if the edge was created, False if it already existed
        """
        domain_id = sanitize_domain_id(domain_name)

        if self.client.edge_exists(endpoint_id, domain_id, IN_DOMAIN):
            return False

        self.client.add_edge(endpoint_id, domain_id, IN_DOMAIN)
        return True

    def build_domain_nodes(
        self, registry: ToolRegistry, skip_if_exists: bool = True
    ) -> Tuple[int, int]:
        """Build domain nodes from all endpoints in the registry.

        Extracts unique domain names from all endpoints and creates
        Domain nodes for each.

        Args:
            registry: The ToolRegistry containing tools with endpoints
            skip_if_exists: If True (default), skip domains that already exist.

        Returns:
            Tuple of (domains_added, domains_skipped)
        """
        # Extract unique non-empty domains from all endpoints
        unique_domains: set = set()
        for tool in registry.tools.values():
            for endpoint in tool.endpoints:
                if endpoint.domain:  # Skip empty domains
                    unique_domains.add(endpoint.domain)

        domains_added = 0
        domains_skipped = 0

        domains_iter = unique_domains
        if self.show_progress:
            domains_iter = tqdm(
                list(domains_iter), desc="Creating domain nodes...", unit="domain"
            )

        for domain_name in domains_iter:
            if self.add_domain(domain_name, skip_if_exists=skip_if_exists):
                domains_added += 1
            else:
                domains_skipped += 1

        return domains_added, domains_skipped

    def build_in_domain_edges(self, registry: ToolRegistry) -> int:
        """Build IN_DOMAIN edges for all endpoints with domains.

        Creates edges from each endpoint to its corresponding domain node.

        Args:
            registry: The ToolRegistry containing tools with endpoints

        Returns:
            Number of IN_DOMAIN edges created
        """
        # Collect all endpoints with non-empty domains
        endpoints_with_domains: List[Tuple[str, str]] = []
        for tool in registry.tools.values():
            for endpoint in tool.endpoints:
                if endpoint.domain:  # Skip empty domains
                    endpoints_with_domains.append((endpoint.id, endpoint.domain))

        edges_added = 0

        edges_iter = endpoints_with_domains
        if self.show_progress:
            edges_iter = tqdm(
                list(edges_iter), desc="Creating IN_DOMAIN edges...", unit="edge"
            )

        for endpoint_id, domain_name in edges_iter:
            if self.add_in_domain_edge(endpoint_id, domain_name):
                edges_added += 1

        return edges_added

    def get_domain_stats(self) -> Dict[str, int]:
        """Get statistics about domain distribution.

        Returns:
            Dictionary mapping domain name to endpoint count
        """
        domain_stats: Dict[str, int] = {}
        for domain in self.client.get_all_domains():
            endpoints = self.client.get_endpoints_by_domain(domain)
            domain_stats[domain] = len(endpoints)
        return domain_stats

    def create_same_domain_edges(self, weight: float = 1.0) -> int:
        """Create SAME_DOMAIN edges between endpoints sharing a domain.

        For each domain, creates edges between all pairs of endpoints
        in that domain. Uses lexicographic ordering (id1 < id2) to avoid
        duplicate edges and self-loops.

        Args:
            weight: Edge weight attribute (default 1.0)

        Returns:
            Total number of edges created

        Example:
            >>> builder.create_same_domain_edges(weight=1.0)
            10  # If weather(3), maps(2), finance(4): 3+1+6 = 10 edges
        """
        domains = self.client.get_all_domains()
        total_edges_created = 0

        domains_iter = domains
        if self.show_progress:
            domains_iter = tqdm(
                list(domains_iter), desc="Creating same-domain edges...", unit="domain"
            )

        for domain in domains_iter:
            endpoints = self.client.get_endpoints_by_domain(domain)

            # Skip domains with fewer than 2 endpoints (no pairs possible)
            if len(endpoints) < 2:
                continue

            # Sort endpoints to ensure consistent ordering
            sorted_endpoints = sorted(endpoints)

            # Create edges for all unique pairs (id1 < id2 by using combinations on sorted list)
            for ep1, ep2 in combinations(sorted_endpoints, 2):
                # Check if edge already exists
                if not self.client.edge_exists(ep1, ep2, SAME_DOMAIN):
                    self.client.add_edge(ep1, ep2, SAME_DOMAIN, weight=weight)
                    total_edges_created += 1

        logger.info(
            f"Created {total_edges_created} same-domain edges across {len(domains)} domains"
        )

        return total_edges_created

    def create_semantic_edges(
        self,
        cache: "EmbeddingCache",  # noqa: F821 - imported at runtime
        threshold: float = 0.7,
        skip_same_domain: bool = True,
        top_k: Optional[int] = None,
        chunk_size: int = 1000,
    ) -> SimilarityStats:
        """Create SEMANTICALLY_SIMILAR edges based on embedding cosine similarity.

        Computes pairwise cosine similarity between all endpoint embeddings
        and creates edges where similarity meets the threshold. Supports
        memory-efficient chunking for large graphs.

        Args:
            cache: EmbeddingCache containing endpoint embeddings
            threshold: Minimum similarity score to create edge (default 0.7)
            skip_same_domain: If True, skip pairs that already have SAME_DOMAIN edge
            top_k: If set, only keep top-k most similar endpoints per endpoint
            chunk_size: Process embeddings in chunks for memory efficiency

        Returns:
            SimilarityStats with edge counts and similarity statistics

        Example:
            >>> stats = builder.create_semantic_edges(cache, threshold=0.7, top_k=10)
            >>> print(f"Created {stats.edges_created} semantic edges")
        """
        stats = SimilarityStats()

        # Get all embeddings from cache
        all_embeddings = cache.get_all()
        if not all_embeddings:
            logger.info("No embeddings in cache, skipping semantic edge creation")
            return stats

        # Get endpoint IDs and embeddings as aligned lists
        endpoint_ids = list(all_embeddings.keys())
        embeddings_list = [all_embeddings[ep_id] for ep_id in endpoint_ids]

        n = len(endpoint_ids)
        if n < 2:
            logger.info("Fewer than 2 embeddings, skipping semantic edge creation")
            return stats

        # Stack embeddings into matrix
        embeddings_matrix = np.vstack(embeddings_list)

        # Compute pairwise similarities
        logger.info(f"Computing pairwise similarities for {n} endpoints...")
        sim_matrix = compute_pairwise_similarity(embeddings_matrix)

        # Collect candidate edges (above threshold)
        # Format: List of (score, ep1_idx, ep2_idx)
        candidates: List[Tuple[float, int, int]] = []

        # Process upper triangle only (i < j to avoid duplicates)
        pairs_iter = range(n)
        if self.show_progress:
            pairs_iter = tqdm(pairs_iter, desc="Computing semantic similarities...", unit="row")

        for i in pairs_iter:
            for j in range(i + 1, n):
                score = float(sim_matrix[i, j])
                if score >= threshold:
                    candidates.append((score, i, j))

        # Sort candidates by score descending for top-k filtering
        candidates.sort(reverse=True, key=lambda x: x[0])

        # Track how many neighbors each endpoint has (for top-k)
        neighbor_count: Dict[int, int] = {i: 0 for i in range(n)}

        # Track similarity scores for statistics
        created_scores: List[float] = []

        for score, i, j in candidates:
            ep1_id = endpoint_ids[i]
            ep2_id = endpoint_ids[j]

            # Ensure consistent ordering (lexicographic)
            if ep1_id > ep2_id:
                ep1_id, ep2_id = ep2_id, ep1_id
                i, j = j, i

            # Check if edge already exists
            if self.client.edge_exists(ep1_id, ep2_id, SEMANTICALLY_SIMILAR):
                continue

            # Check skip_same_domain
            if skip_same_domain:
                if (self.client.edge_exists(ep1_id, ep2_id, SAME_DOMAIN) or
                    self.client.edge_exists(ep2_id, ep1_id, SAME_DOMAIN)):
                    stats.edges_skipped_same_domain += 1
                    continue

            # Check top-k limit
            if top_k is not None:
                if neighbor_count[i] >= top_k and neighbor_count[j] >= top_k:
                    # Both endpoints already have top_k neighbors
                    continue

            # Create edge
            self.client.add_edge(ep1_id, ep2_id, SEMANTICALLY_SIMILAR, score=score)
            stats.edges_created += 1
            created_scores.append(score)

            # Update neighbor counts
            neighbor_count[i] += 1
            neighbor_count[j] += 1

        # Count edges skipped due to threshold
        # Total possible pairs - created - same_domain_skipped - already_existed
        total_pairs = n * (n - 1) // 2
        stats.edges_skipped_threshold = total_pairs - len(candidates)

        # Compute statistics
        if created_scores:
            stats.mean_similarity = float(np.mean(created_scores))
            stats.max_similarity = float(np.max(created_scores))
            stats.min_similarity = float(np.min(created_scores))

        logger.info(
            f"Created {stats.edges_created} semantic edges "
            f"(skipped {stats.edges_skipped_threshold} below threshold, "
            f"{stats.edges_skipped_same_domain} same-domain)"
        )

        return stats