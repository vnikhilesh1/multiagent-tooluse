"""GraphClient - NetworkX-based graph client for tool relationship storage.

This module provides a wrapper around NetworkX MultiDiGraph for storing
and querying tool, endpoint, and domain relationships with support for
multiple persistence formats.
"""

import gzip
import json
import logging
import pickle
import random
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import networkx as nx
import numpy as np

from src.config import GraphConfig

from .constants import (
    DOMAIN,
    ENDPOINT,
    HAS_ENDPOINT,
    IN_DOMAIN,
    METADATA_ATTR,
    SAME_DOMAIN,
    SCHEMA_VERSION,
    SEMANTICALLY_SIMILAR,
    TOOL,
    TYPE_ATTR,
)

# Optional matplotlib import for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logger = logging.getLogger(__name__)


class GraphClient:
    """NetworkX-based graph client for tool relationship storage.

    Wraps a NetworkX MultiDiGraph to support multiple edge types between
    the same nodes. Provides persistence in multiple formats (pickle,
    graphml, json) and context manager support for atomic save operations.

    Attributes:
        config: GraphConfig instance with path and format settings
        graph: The underlying NetworkX MultiDiGraph

    Example:
        >>> from src.config import GraphConfig
        >>> config = GraphConfig()
        >>> client = GraphClient(config)
        >>> client.add_node("tool_1", "Tool", name="Weather API")
        >>> client.save_graph()

        # Context manager usage (auto-saves on exit)
        >>> with GraphClient(config) as client:
        ...     client.add_node("tool_2", "Tool", name="Maps API")
    """

    def __init__(self, config: GraphConfig) -> None:
        """Initialize GraphClient with configuration.

        Args:
            config: GraphConfig instance specifying path, format, and other settings
        """
        self.config = config
        self.graph: nx.MultiDiGraph = self._create_empty_graph()
        self._context_entered = False

        # Initialize in-memory indexes for O(1) lookups
        self._tool_index: Dict[str, Dict[str, Any]] = {}
        self._endpoint_index: Dict[str, Dict[str, Any]] = {}
        self._domain_index: Dict[str, List[str]] = {}
        self._category_index: Dict[str, List[str]] = {}

    def _create_empty_graph(self) -> nx.MultiDiGraph:
        """Create a new empty graph with metadata.

        Returns:
            Empty MultiDiGraph with metadata attributes
        """
        graph = nx.MultiDiGraph()
        graph.graph[METADATA_ATTR] = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        return graph

    def __enter__(self) -> "GraphClient":
        """Enter context manager.

        Returns:
            Self for use in with statement
        """
        self._context_entered = True
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager, auto-saving on successful exit.

        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
        """
        if exc_type is None:
            # No exception - save the graph
            self.save_graph()
        self._context_entered = False

    def add_node(self, node_id: str, node_type: str, **attributes: Any) -> None:
        """Add a node to the graph with the specified type.

        Also updates the relevant in-memory indexes for O(1) lookups.

        Args:
            node_id: Unique identifier for the node
            node_type: Type of the node (e.g., TOOL, ENDPOINT, DOMAIN)
            **attributes: Additional attributes to store on the node
        """
        node_attrs = {TYPE_ATTR: node_type, **attributes}
        self.graph.add_node(node_id, **node_attrs)

        # Update indexes based on node type
        if node_type == TOOL:
            self._tool_index[node_id] = node_attrs
        elif node_type == ENDPOINT:
            self._endpoint_index[node_id] = node_attrs

            # Update domain index
            domain = attributes.get("domain")
            if domain:
                if domain not in self._domain_index:
                    self._domain_index[domain] = []
                if node_id not in self._domain_index[domain]:
                    self._domain_index[domain].append(node_id)

            # Update category index based on tool_id
            tool_id = attributes.get("tool_id")
            if tool_id and tool_id in self._tool_index:
                category = self._tool_index[tool_id].get("category")
                if category:
                    if category not in self._category_index:
                        self._category_index[category] = []
                    if node_id not in self._category_index[category]:
                        self._category_index[category].append(node_id)

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        **attributes: Any
    ) -> None:
        """Add an edge between two nodes with the specified type.

        Args:
            source: Source node ID
            target: Target node ID
            edge_type: Type of the edge (e.g., HAS_ENDPOINT, SAME_DOMAIN)
            **attributes: Additional attributes to store on the edge
        """
        self.graph.add_edge(source, target, key=edge_type, **{TYPE_ATTR: edge_type, **attributes})

    def node_exists(self, node_id: str) -> bool:
        """Check if a node exists in the graph.

        Args:
            node_id: Node ID to check

        Returns:
            True if the node exists, False otherwise
        """
        return node_id in self.graph.nodes

    def edge_exists(self, source: str, target: str, edge_type: str) -> bool:
        """Check if an edge of the specified type exists between two nodes.

        Args:
            source: Source node ID
            target: Target node ID
            edge_type: Type of edge to check for

        Returns:
            True if the edge exists, False otherwise
        """
        if not self.graph.has_edge(source, target):
            return False
        # Check if an edge with this specific type exists
        return edge_type in self.graph[source][target]

    def get_edge_attributes(
        self, source: str, target: str, edge_type: str
    ) -> Optional[Dict[str, Any]]:
        """Get attributes of a specific edge.

        Args:
            source: Source node ID
            target: Target node ID
            edge_type: Type of edge to retrieve

        Returns:
            Dictionary of edge attributes, or None if edge doesn't exist
        """
        if not self.edge_exists(source, target, edge_type):
            return None
        return dict(self.graph[source][target][edge_type])

    def get_nodes_by_type(self, node_type: str) -> List[str]:
        """Get all node IDs of the specified type.

        Args:
            node_type: Type of nodes to retrieve

        Returns:
            List of node IDs matching the specified type
        """
        return [
            node_id
            for node_id, attrs in self.graph.nodes(data=True)
            if attrs.get(TYPE_ATTR) == node_type
        ]

    def get_edges_by_type(self, edge_type: str) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Get all edges of the specified type.

        Args:
            edge_type: Type of edges to retrieve

        Returns:
            List of tuples (source, target, attributes) for matching edges
        """
        result = []
        for source, target, key, attrs in self.graph.edges(keys=True, data=True):
            if key == edge_type or attrs.get(TYPE_ATTR) == edge_type:
                result.append((source, target, attrs))
        return result

    def get_node_attributes(self, node_id: str) -> Dict[str, Any]:
        """Get all attributes of a node.

        Args:
            node_id: Node ID to get attributes for

        Returns:
            Dictionary of node attributes

        Raises:
            KeyError: If node doesn't exist
        """
        return dict(self.graph.nodes[node_id])

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the graph.

        Returns:
            Dictionary with:
            - total_nodes: Total number of nodes
            - total_edges: Total number of edges
            - nodes_by_type: Counter of nodes by type
            - edges_by_type: Counter of edges by type
            - density: Graph density (edges / possible edges)
        """
        nodes_by_type: Counter = Counter()
        for _, attrs in self.graph.nodes(data=True):
            node_type = attrs.get(TYPE_ATTR, "Unknown")
            nodes_by_type[node_type] += 1

        edges_by_type: Counter = Counter()
        for _, _, key, attrs in self.graph.edges(keys=True, data=True):
            edge_type = attrs.get(TYPE_ATTR, key)
            edges_by_type[edge_type] += 1

        total_nodes = self.graph.number_of_nodes()
        total_edges = self.graph.number_of_edges()

        # Calculate density (for directed graph)
        if total_nodes > 1:
            possible_edges = total_nodes * (total_nodes - 1)
            density = total_edges / possible_edges
        else:
            density = 0.0

        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "nodes_by_type": dict(nodes_by_type),
            "edges_by_type": dict(edges_by_type),
            "density": density,
        }

    def clear_graph(self) -> None:
        """Clear all nodes and edges from the graph, keeping metadata.

        Also clears all in-memory indexes.
        """
        self.graph = self._create_empty_graph()
        self._tool_index.clear()
        self._endpoint_index.clear()
        self._domain_index.clear()
        self._category_index.clear()

    def save_graph(self, path: Optional[Path] = None) -> None:
        """Save the graph to disk in the configured format.

        Args:
            path: Optional path override. If not provided, uses config.path
        """
        save_path = path or self.config.path

        # Ensure parent directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Update metadata timestamp
        if METADATA_ATTR in self.graph.graph:
            self.graph.graph[METADATA_ATTR]["updated_at"] = datetime.utcnow().isoformat()

        format_type = self.config.format

        if format_type == "pickle":
            self._save_pickle(save_path)
        elif format_type == "json":
            self._save_json(save_path)
        elif format_type == "graphml":
            self._save_graphml(save_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

        logger.debug(f"Saved graph to {save_path} (format: {format_type})")

    def _save_pickle(self, path: Path) -> None:
        """Save graph in pickle format."""
        with open(path, "wb") as f:
            pickle.dump(self.graph, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _save_json(self, path: Path) -> None:
        """Save graph in JSON format using node-link data."""
        data = nx.node_link_data(self.graph, edges="links")
        # Add graph-level metadata
        data["metadata"] = self.graph.graph.get(METADATA_ATTR, {})
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _save_graphml(self, path: Path) -> None:
        """Save graph in GraphML format.

        Note: GraphML has limitations with complex attributes.
        Non-string attributes are converted to strings.
        """
        # Create a copy to avoid modifying the original graph
        graph_copy = self.graph.copy()

        # Convert graph-level metadata to string for GraphML compatibility
        if METADATA_ATTR in graph_copy.graph:
            graph_copy.graph[METADATA_ATTR] = json.dumps(graph_copy.graph[METADATA_ATTR], default=str)

        # Convert complex node attributes to strings for GraphML compatibility
        for node_id in graph_copy.nodes():
            for key, value in list(graph_copy.nodes[node_id].items()):
                if not isinstance(value, (str, int, float, bool)):
                    graph_copy.nodes[node_id][key] = json.dumps(value, default=str)

        # Convert complex edge attributes to strings for GraphML compatibility
        for u, v, k, attrs in list(graph_copy.edges(keys=True, data=True)):
            for key, value in list(attrs.items()):
                if not isinstance(value, (str, int, float, bool)):
                    graph_copy[u][v][k][key] = json.dumps(value, default=str)

        nx.write_graphml(graph_copy, path)

    def load_graph(self, path: Optional[Path] = None) -> None:
        """Load the graph from disk.

        If the file doesn't exist, creates an empty graph.
        After loading, rebuilds all in-memory indexes.

        Args:
            path: Optional path override. If not provided, uses config.path
        """
        load_path = path or self.config.path

        if not load_path.exists():
            logger.debug(f"Graph file not found at {load_path}, using empty graph")
            self.graph = self._create_empty_graph()
            self._clear_indexes()
            return

        format_type = self.config.format

        if format_type == "pickle":
            self._load_pickle(load_path)
        elif format_type == "json":
            self._load_json(load_path)
        elif format_type == "graphml":
            self._load_graphml(load_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

        # Rebuild indexes after loading
        self.rebuild_indexes()

        logger.debug(f"Loaded graph from {load_path} (format: {format_type})")

    def _load_pickle(self, path: Path) -> None:
        """Load graph from pickle format."""
        with open(path, "rb") as f:
            self.graph = pickle.load(f)

    def _load_json(self, path: Path) -> None:
        """Load graph from JSON format."""
        with open(path, "r") as f:
            data = json.load(f)
        # Restore metadata if present
        metadata = data.pop("metadata", None)
        self.graph = nx.node_link_graph(data, directed=True, multigraph=True, edges="links")
        if metadata:
            self.graph.graph[METADATA_ATTR] = metadata

    def _load_graphml(self, path: Path) -> None:
        """Load graph from GraphML format."""
        # Read as MultiDiGraph
        self.graph = nx.read_graphml(path)

        # Convert to MultiDiGraph if needed
        if not isinstance(self.graph, nx.MultiDiGraph):
            self.graph = nx.MultiDiGraph(self.graph)

        # Ensure metadata exists
        if METADATA_ATTR not in self.graph.graph:
            self.graph.graph[METADATA_ATTR] = {
                "schema_version": SCHEMA_VERSION,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            }

    # =========================================================================
    # Index Management Methods
    # =========================================================================

    def _clear_indexes(self) -> None:
        """Clear all in-memory indexes."""
        self._tool_index.clear()
        self._endpoint_index.clear()
        self._domain_index.clear()
        self._category_index.clear()

    def rebuild_indexes(self) -> None:
        """Rebuild all in-memory indexes from the graph.

        Scans all nodes and rebuilds tool, endpoint, domain, and category
        indexes. This method is idempotent and can be called multiple times.
        """
        # Clear existing indexes
        self._clear_indexes()

        # First pass: index all tools to get their categories
        for node_id, attrs in self.graph.nodes(data=True):
            node_type = attrs.get(TYPE_ATTR)
            if node_type == TOOL:
                self._tool_index[node_id] = dict(attrs)

        # Second pass: index all endpoints with domain and category
        for node_id, attrs in self.graph.nodes(data=True):
            node_type = attrs.get(TYPE_ATTR)
            if node_type == ENDPOINT:
                self._endpoint_index[node_id] = dict(attrs)

                # Update domain index
                domain = attrs.get("domain")
                if domain:
                    if domain not in self._domain_index:
                        self._domain_index[domain] = []
                    if node_id not in self._domain_index[domain]:
                        self._domain_index[domain].append(node_id)

                # Update category index based on tool_id
                tool_id = attrs.get("tool_id")
                if tool_id and tool_id in self._tool_index:
                    category = self._tool_index[tool_id].get("category")
                    if category:
                        if category not in self._category_index:
                            self._category_index[category] = []
                        if node_id not in self._category_index[category]:
                            self._category_index[category].append(node_id)

        logger.debug(
            f"Rebuilt indexes: {len(self._tool_index)} tools, "
            f"{len(self._endpoint_index)} endpoints, "
            f"{len(self._domain_index)} domains, "
            f"{len(self._category_index)} categories"
        )

    def get_tool_by_id(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """Get tool attributes by ID using O(1) index lookup.

        Args:
            tool_id: The tool ID to look up

        Returns:
            Dictionary of tool attributes, or None if not found
        """
        return self._tool_index.get(tool_id)

    def get_endpoint_by_id(self, endpoint_id: str) -> Optional[Dict[str, Any]]:
        """Get endpoint attributes by ID using O(1) index lookup.

        Args:
            endpoint_id: The endpoint ID to look up

        Returns:
            Dictionary of endpoint attributes, or None if not found
        """
        return self._endpoint_index.get(endpoint_id)

    def get_endpoints_by_domain(self, domain: str) -> List[str]:
        """Get all endpoint IDs in a domain using O(1) index lookup.

        Args:
            domain: The domain name to look up

        Returns:
            List of endpoint IDs in the domain, or empty list if domain not found
        """
        return self._domain_index.get(domain, []).copy()

    def get_endpoints_by_category(self, category: str) -> List[str]:
        """Get all endpoint IDs in a category using O(1) index lookup.

        The category is determined by the parent tool's category attribute.

        Args:
            category: The category name to look up

        Returns:
            List of endpoint IDs in the category, or empty list if category not found
        """
        return self._category_index.get(category, []).copy()

    def get_all_domains(self) -> List[str]:
        """Get all unique domain names in the graph.

        Returns:
            List of all domain names
        """
        return list(self._domain_index.keys())

    def get_all_categories(self) -> List[str]:
        """Get all unique category names in the graph.

        Returns:
            List of all category names
        """
        return list(self._category_index.keys())

    # =========================================================================
    # Graph Query Methods
    # =========================================================================

    def get_neighbors(self, node_id: str, edge_type: Optional[str] = None) -> List[str]:
        """Get adjacent nodes from a given node.

        Treats the graph as undirected for neighbor lookup, returning nodes
        connected via both incoming and outgoing edges.

        Args:
            node_id: The node ID to get neighbors for
            edge_type: Optional edge type filter. If provided, only return
                       neighbors connected by that edge type.

        Returns:
            List of neighbor node IDs. Empty list if node doesn't exist
            or has no neighbors.
        """
        if not self.node_exists(node_id):
            return []

        neighbors = set()

        # Get successors (outgoing edges)
        for successor in self.graph.successors(node_id):
            if edge_type is None:
                neighbors.add(successor)
            else:
                # Check if edge of specific type exists
                if edge_type in self.graph[node_id][successor]:
                    neighbors.add(successor)

        # Get predecessors (incoming edges)
        for predecessor in self.graph.predecessors(node_id):
            if edge_type is None:
                neighbors.add(predecessor)
            else:
                # Check if edge of specific type exists
                if edge_type in self.graph[predecessor][node_id]:
                    neighbors.add(predecessor)

        return list(neighbors)

    def get_connected_endpoints(self, start_id: str, max_depth: int = 2) -> List[str]:
        """Get all endpoints reachable from a starting endpoint via BFS.

        Uses breadth-first search to find connected endpoints within
        the specified depth limit.

        Args:
            start_id: The starting endpoint ID
            max_depth: Maximum search depth (1 = direct neighbors only)

        Returns:
            List of endpoint IDs reachable from start. Does not include
            the starting node. Empty list if start doesn't exist.
        """
        if not self.node_exists(start_id):
            return []

        visited = set([start_id])
        result = []
        queue = deque([(start_id, 0)])

        while queue:
            current_id, depth = queue.popleft()

            if depth >= max_depth:
                continue

            # Get all neighbors (undirected)
            neighbors = self.get_neighbors(current_id)

            for neighbor_id in neighbors:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)

                    # Only include ENDPOINT type nodes in result
                    node_attrs = self.graph.nodes.get(neighbor_id, {})
                    if node_attrs.get(TYPE_ATTR) == ENDPOINT:
                        result.append(neighbor_id)

                    queue.append((neighbor_id, depth + 1))

        return result

    def find_path(self, source: str, target: str) -> Optional[List[str]]:
        """Find shortest path between two nodes.

        Uses NetworkX's shortest path algorithm on an undirected view
        of the graph.

        Args:
            source: Source node ID
            target: Target node ID

        Returns:
            List of node IDs representing the path from source to target,
            including both endpoints. Returns None if no path exists.
            Returns [source] if source == target.
        """
        if source == target:
            return [source]

        if not self.node_exists(source) or not self.node_exists(target):
            return None

        try:
            # Use undirected view for path finding
            undirected = self.graph.to_undirected()
            path = nx.shortest_path(undirected, source, target)
            return list(path)
        except nx.NetworkXNoPath:
            return None
        except nx.NodeNotFound:
            return None

    def get_subgraph_by_domain(self, domains: List[str]) -> nx.MultiDiGraph:
        """Extract a subgraph containing only endpoints in the specified domains.

        Args:
            domains: List of domain names to include

        Returns:
            New MultiDiGraph containing only endpoints in the specified
            domains and all edges between them. Empty graph if no matches.
        """
        if not domains:
            return nx.MultiDiGraph()

        # Collect all endpoint IDs in the specified domains
        endpoint_ids = set()
        for domain in domains:
            domain_endpoints = self._domain_index.get(domain, [])
            endpoint_ids.update(domain_endpoints)

        if not endpoint_ids:
            return nx.MultiDiGraph()

        # Create subgraph with only these endpoints
        subgraph = self.graph.subgraph(endpoint_ids).copy()
        return subgraph

    def get_similar_endpoints(
        self,
        endpoint_id: str,
        min_score: float = 0.0,
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """Get endpoints connected via SEMANTICALLY_SIMILAR edges.

        Args:
            endpoint_id: The endpoint ID to find similar endpoints for
            min_score: Minimum similarity score threshold (default 0.0)
            top_k: Maximum number of results to return (default None = all)

        Returns:
            List of (endpoint_id, similarity_score) tuples, sorted by
            score descending. Empty list if endpoint doesn't exist or
            has no similar endpoints.
        """
        if not self.node_exists(endpoint_id):
            return []

        similar = []

        # Check outgoing SEMANTICALLY_SIMILAR edges
        for successor in self.graph.successors(endpoint_id):
            if SEMANTICALLY_SIMILAR in self.graph[endpoint_id][successor]:
                edge_data = self.graph[endpoint_id][successor][SEMANTICALLY_SIMILAR]
                score = edge_data.get("score", 0.0)
                if score >= min_score:
                    similar.append((successor, score))

        # Check incoming SEMANTICALLY_SIMILAR edges
        for predecessor in self.graph.predecessors(endpoint_id):
            if SEMANTICALLY_SIMILAR in self.graph[predecessor][endpoint_id]:
                edge_data = self.graph[predecessor][endpoint_id][SEMANTICALLY_SIMILAR]
                score = edge_data.get("score", 0.0)
                if score >= min_score:
                    # Avoid duplicates if edge exists in both directions
                    if predecessor not in [ep_id for ep_id, _ in similar]:
                        similar.append((predecessor, score))

        # Sort by score descending
        similar.sort(key=lambda x: x[1], reverse=True)

        # Apply top_k limit
        if top_k is not None:
            similar = similar[:top_k]

        return similar

    def random_walk(
        self,
        start_id: str,
        steps: int,
        edge_types: Optional[List[str]] = None
    ) -> List[str]:
        """Perform a random walk from the starting node.

        Args:
            start_id: Starting node ID
            steps: Number of steps to take
            edge_types: Optional list of edge types to follow.
                        If None, follow any edge type.

        Returns:
            List of node IDs visited during the walk, including the
            starting node. Stops early if a dead end is reached.
        """
        if not self.node_exists(start_id):
            return [start_id]

        walk = [start_id]
        current = start_id

        for _ in range(steps):
            # Get valid neighbors based on edge_types filter
            if edge_types is None:
                neighbors = self.get_neighbors(current)
            else:
                neighbors = []
                for edge_type in edge_types:
                    neighbors.extend(self.get_neighbors(current, edge_type=edge_type))
                # Remove duplicates while preserving order
                neighbors = list(dict.fromkeys(neighbors))

            if not neighbors:
                # Dead end - stop walk
                break

            # Randomly select next node
            current = random.choice(neighbors)
            walk.append(current)

        return walk

    def get_edge_weight(
        self,
        source: str,
        target: str,
        edge_type: str
    ) -> Optional[float]:
        """Get the weight/score of an edge.

        Args:
            source: Source node ID
            target: Target node ID
            edge_type: Type of edge to get weight for

        Returns:
            Edge weight (for SAME_DOMAIN) or score (for SEMANTICALLY_SIMILAR).
            Returns None if edge doesn't exist.
        """
        edge_attrs = self.get_edge_attributes(source, target, edge_type)
        if edge_attrs is None:
            return None

        # Try 'weight' first, then 'score'
        if "weight" in edge_attrs:
            return edge_attrs["weight"]
        if "score" in edge_attrs:
            return edge_attrs["score"]

        return None

    def filter_by_completeness(self, min_score: float = 0.5) -> List[str]:
        """Get endpoints that meet a minimum completeness threshold.

        Args:
            min_score: Minimum completeness score (default 0.5)

        Returns:
            List of endpoint IDs with completeness_score >= min_score.
            Empty list if no endpoints meet the threshold.
        """
        result = []

        for endpoint_id, attrs in self._endpoint_index.items():
            completeness = attrs.get("completeness_score", 0.0)
            if completeness is not None and completeness >= min_score:
                result.append(endpoint_id)

        return result

    # =========================================================================
    # Persistence & Serialization Methods
    # =========================================================================

    def _serialize_value(self, value: Any) -> Any:
        """Convert a value to a JSON-serializable type.

        Args:
            value: Any value to serialize

        Returns:
            JSON-serializable version of the value
        """
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        return value

    def _update_metadata_timestamp(self) -> None:
        """Update the updated_at timestamp in metadata."""
        if METADATA_ATTR in self.graph.graph:
            self.graph.graph[METADATA_ATTR]["updated_at"] = datetime.utcnow().isoformat()

    def save_to_pickle(self, path: Path, include_indexes: bool = False) -> None:
        """Save graph to pickle format (fast binary).

        Args:
            path: Path to save the pickle file
            include_indexes: If True, save indexes along with the graph for
                            faster loading (default False)
        """
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Update timestamp
        self._update_metadata_timestamp()

        if include_indexes:
            # Save graph with indexes
            data = {
                "graph": self.graph,
                "indexes": {
                    "tool_index": self._tool_index,
                    "endpoint_index": self._endpoint_index,
                    "domain_index": self._domain_index,
                    "category_index": self._category_index,
                }
            }
        else:
            # Save just the graph
            data = self.graph

        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.debug(f"Saved graph to pickle: {path} (include_indexes={include_indexes})")

    def load_from_pickle(self, path: Path) -> None:
        """Load graph from pickle format.

        Args:
            path: Path to the pickle file

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        if not path.exists():
            raise FileNotFoundError(f"Pickle file not found: {path}")

        with open(path, "rb") as f:
            data = pickle.load(f)

        # Detect format: dict with indexes or plain graph
        if isinstance(data, dict) and "graph" in data and "indexes" in data:
            # Load graph and indexes
            self.graph = data["graph"]
            indexes = data["indexes"]
            self._tool_index = indexes.get("tool_index", {})
            self._endpoint_index = indexes.get("endpoint_index", {})
            self._domain_index = indexes.get("domain_index", {})
            self._category_index = indexes.get("category_index", {})
            logger.debug(f"Loaded graph with indexes from pickle: {path}")
        else:
            # Plain graph - rebuild indexes
            self.graph = data
            self.rebuild_indexes()
            logger.debug(f"Loaded graph from pickle (rebuilding indexes): {path}")

    def save_to_graphml(self, path: Path, compress: bool = False) -> None:
        """Save graph to GraphML format (portable XML).

        Args:
            path: Path to save the GraphML file
            compress: If True, gzip compress the output (default False)
        """
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Update timestamp
        self._update_metadata_timestamp()

        # Create a copy to avoid modifying the original graph
        graph_copy = self.graph.copy()

        # Convert graph-level metadata to string for GraphML compatibility
        if METADATA_ATTR in graph_copy.graph:
            graph_copy.graph[METADATA_ATTR] = json.dumps(
                graph_copy.graph[METADATA_ATTR], default=str
            )

        # Convert complex node attributes to strings for GraphML compatibility
        for node_id in graph_copy.nodes():
            for key, value in list(graph_copy.nodes[node_id].items()):
                if not isinstance(value, (str, int, float, bool)):
                    serialized = self._serialize_value(value)
                    graph_copy.nodes[node_id][key] = json.dumps(serialized, default=str)

        # Convert complex edge attributes to strings for GraphML compatibility
        for u, v, k, attrs in list(graph_copy.edges(keys=True, data=True)):
            for key, value in list(attrs.items()):
                if not isinstance(value, (str, int, float, bool)):
                    serialized = self._serialize_value(value)
                    graph_copy[u][v][k][key] = json.dumps(serialized, default=str)

        if compress:
            # Write to gzip-compressed file
            with gzip.open(path, "wt", encoding="utf-8") as f:
                # Write GraphML to string first, then to gzip
                import io
                buffer = io.BytesIO()
                nx.write_graphml(graph_copy, buffer)
                f.write(buffer.getvalue().decode("utf-8"))
        else:
            nx.write_graphml(graph_copy, path)

        logger.debug(f"Saved graph to GraphML: {path} (compress={compress})")

    def load_from_graphml(self, path: Path, compressed: bool = False) -> None:
        """Load graph from GraphML format.

        Args:
            path: Path to the GraphML file
            compressed: If True, read from gzip-compressed file (default False)

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        if not path.exists():
            raise FileNotFoundError(f"GraphML file not found: {path}")

        if compressed:
            # Read from gzip-compressed file
            with gzip.open(path, "rt", encoding="utf-8") as f:
                import io
                content = f.read()
                buffer = io.BytesIO(content.encode("utf-8"))
                self.graph = nx.read_graphml(buffer)
        else:
            self.graph = nx.read_graphml(path)

        # Convert to MultiDiGraph if needed
        if not isinstance(self.graph, nx.MultiDiGraph):
            self.graph = nx.MultiDiGraph(self.graph)

        # Restore metadata from JSON string if present
        if METADATA_ATTR in self.graph.graph:
            try:
                if isinstance(self.graph.graph[METADATA_ATTR], str):
                    self.graph.graph[METADATA_ATTR] = json.loads(
                        self.graph.graph[METADATA_ATTR]
                    )
            except (json.JSONDecodeError, TypeError):
                pass

        # Ensure metadata exists
        if METADATA_ATTR not in self.graph.graph:
            self.graph.graph[METADATA_ATTR] = {
                "schema_version": SCHEMA_VERSION,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            }

        # Rebuild indexes
        self.rebuild_indexes()

        logger.debug(f"Loaded graph from GraphML: {path} (compressed={compressed})")

    def export_to_json(self, path: Path, compress: bool = False) -> None:
        """Export graph to human-readable JSON format.

        Creates a structured JSON with metadata, nodes, and edges lists.

        Args:
            path: Path to save the JSON file
            compress: If True, gzip compress the output (default False)
        """
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Update timestamp
        self._update_metadata_timestamp()

        # Build the JSON structure
        data = {
            "metadata": self.graph.graph.get(METADATA_ATTR, {
                "schema_version": SCHEMA_VERSION,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            }),
            "nodes": [],
            "edges": [],
        }

        # Export nodes
        for node_id, attrs in self.graph.nodes(data=True):
            node_data = {"id": node_id}
            for key, value in attrs.items():
                node_data[key] = self._serialize_value(value)
            data["nodes"].append(node_data)

        # Export edges
        for source, target, key, attrs in self.graph.edges(keys=True, data=True):
            edge_data = {
                "source": source,
                "target": target,
                "key": key,
            }
            for attr_key, value in attrs.items():
                edge_data[attr_key] = self._serialize_value(value)
            data["edges"].append(edge_data)

        # Write to file
        if compress:
            with gzip.open(path, "wt", encoding="utf-8") as f:
                json.dump(data, f, default=str)
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)

        logger.debug(f"Exported graph to JSON: {path} (compress={compress})")

    def import_from_json(self, path: Path, compressed: bool = False) -> None:
        """Import graph from JSON format.

        Args:
            path: Path to the JSON file
            compressed: If True, read from gzip-compressed file (default False)

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {path}")

        # Read from file
        if compressed:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

        # Create new graph
        self.graph = nx.MultiDiGraph()

        # Restore metadata
        if "metadata" in data:
            self.graph.graph[METADATA_ATTR] = data["metadata"]
        else:
            self.graph.graph[METADATA_ATTR] = {
                "schema_version": SCHEMA_VERSION,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            }

        # Import nodes
        for node_data in data.get("nodes", []):
            node_id = node_data.pop("id")
            self.graph.add_node(node_id, **node_data)

        # Import edges
        for edge_data in data.get("edges", []):
            source = edge_data.pop("source")
            target = edge_data.pop("target")
            key = edge_data.pop("key", None)
            if key:
                self.graph.add_edge(source, target, key=key, **edge_data)
            else:
                self.graph.add_edge(source, target, **edge_data)

        # Rebuild indexes
        self.rebuild_indexes()

        logger.debug(f"Imported graph from JSON: {path} (compressed={compressed})")

    # =========================================================================
    # Visualization Methods
    # =========================================================================

    # Node colors by type
    NODE_COLORS = {
        "Tool": "#4CAF50",      # Green
        "Endpoint": "#2196F3",  # Blue
        "Domain": "#FF9800",    # Orange
    }

    # Edge styles by type
    EDGE_STYLES = {
        "HAS_ENDPOINT": {"color": "#757575", "style": "solid", "width": 1.0},
        "IN_DOMAIN": {"color": "#9C27B0", "style": "dashed", "width": 1.0},
        "SAME_DOMAIN": {"color": "#4CAF50", "style": "solid", "width": 1.5},
        "SEMANTICALLY_SIMILAR": {"color": "#F44336", "style": "dotted", "width": 2.0},
    }

    def draw_subgraph(
        self,
        node_ids: List[str],
        figsize: Tuple[int, int] = (12, 8),
        show_labels: bool = True,
        save_path: Optional[Path] = None,
        label_high_degree: int = 5,
    ) -> Optional[Any]:
        """Visualize a subgraph using matplotlib.

        Args:
            node_ids: List of node IDs to include in the visualization
            figsize: Figure size as (width, height) tuple (default (12, 8))
            show_labels: Whether to show node labels (default True)
            save_path: If provided, save figure to this path instead of displaying
            label_high_degree: Only label nodes with degree >= this value
                              (default 5, use 0 to label all)

        Returns:
            The matplotlib figure object, or None if matplotlib is not available
            or node_ids is empty

        Raises:
            ImportError: If matplotlib is not installed
        """
        if not HAS_MATPLOTLIB:
            raise ImportError(
                "matplotlib is required for visualization. "
                "Install with: pip install matplotlib"
            )

        # Filter to only existing nodes
        valid_node_ids = [nid for nid in node_ids if self.node_exists(nid)]

        if not valid_node_ids:
            logger.warning("No valid nodes to visualize")
            # Create empty figure
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_title("Empty Subgraph")
            ax.axis("off")
            if save_path:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
            return fig

        # Extract subgraph
        subgraph = self.graph.subgraph(valid_node_ids).copy()

        # Create figure and axes
        fig, ax = plt.subplots(figsize=figsize)

        # Compute layout
        pos = nx.spring_layout(subgraph, k=2, iterations=50)

        # Assign node colors based on type
        node_colors = []
        for node_id in subgraph.nodes():
            node_type = subgraph.nodes[node_id].get(TYPE_ATTR, "Unknown")
            color = self.NODE_COLORS.get(node_type, "#CCCCCC")
            node_colors.append(color)

        # Draw nodes
        nx.draw_networkx_nodes(
            subgraph,
            pos,
            node_color=node_colors,
            node_size=500,
            ax=ax,
        )

        # Draw edges by type with different styles
        for edge_type, style in self.EDGE_STYLES.items():
            edges_of_type = [
                (u, v) for u, v, k in subgraph.edges(keys=True) if k == edge_type
            ]
            if edges_of_type:
                nx.draw_networkx_edges(
                    subgraph,
                    pos,
                    edgelist=edges_of_type,
                    edge_color=style["color"],
                    style=style["style"],
                    width=style["width"],
                    ax=ax,
                    arrows=True,
                    arrowsize=10,
                )

        # Draw labels
        if show_labels:
            if label_high_degree > 0:
                # Only label nodes with high degree
                labels = {
                    node: attrs.get("name", node)
                    for node, attrs in subgraph.nodes(data=True)
                    if subgraph.degree(node) >= label_high_degree
                }
            else:
                # Label all nodes
                labels = {
                    node: attrs.get("name", node)
                    for node, attrs in subgraph.nodes(data=True)
                }

            if labels:
                nx.draw_networkx_labels(
                    subgraph,
                    pos,
                    labels,
                    font_size=8,
                    ax=ax,
                )

        # Create legend for node types
        legend_patches = [
            mpatches.Patch(color=color, label=node_type)
            for node_type, color in self.NODE_COLORS.items()
        ]
        ax.legend(handles=legend_patches, loc="upper left")

        # Set title and turn off axis
        ax.set_title(f"Subgraph ({len(valid_node_ids)} nodes)")
        ax.axis("off")

        # Save or show
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.debug(f"Saved subgraph visualization to: {save_path}")
        else:
            plt.tight_layout()
            plt.show()

        return fig

    def export_to_gephi(self, path: Path) -> None:
        """Export graph to GEXF format for Gephi visualization.

        GEXF (Graph Exchange XML Format) is the native format for Gephi,
        a popular graph visualization tool.

        Args:
            path: Path to save the GEXF file
        """
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Create a copy to avoid modifying the original graph
        graph_copy = self.graph.copy()

        # Convert complex node attributes to strings for GEXF compatibility
        for node_id in graph_copy.nodes():
            for key, value in list(graph_copy.nodes[node_id].items()):
                if isinstance(value, np.ndarray):
                    graph_copy.nodes[node_id][key] = json.dumps(
                        value.tolist(), default=str
                    )
                elif not isinstance(value, (str, int, float, bool)):
                    graph_copy.nodes[node_id][key] = json.dumps(value, default=str)

        # Convert complex edge attributes to strings for GEXF compatibility
        for u, v, k, attrs in list(graph_copy.edges(keys=True, data=True)):
            for key, value in list(attrs.items()):
                if isinstance(value, np.ndarray):
                    graph_copy[u][v][k][key] = json.dumps(value.tolist(), default=str)
                elif not isinstance(value, (str, int, float, bool)):
                    graph_copy[u][v][k][key] = json.dumps(value, default=str)

        # Write GEXF file
        nx.write_gexf(graph_copy, path)

        logger.debug(f"Exported graph to GEXF: {path}")