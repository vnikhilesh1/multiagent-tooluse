"""Constants for graph node and edge types.

This module defines the type constants used throughout the graph system
to ensure consistency in node and edge type naming.
"""

# Node Types
TOOL = "Tool"
ENDPOINT = "Endpoint"
DOMAIN = "Domain"

# Edge Types
HAS_ENDPOINT = "HAS_ENDPOINT"
IN_DOMAIN = "IN_DOMAIN"
SAME_DOMAIN = "SAME_DOMAIN"
SEMANTICALLY_SIMILAR = "SEMANTICALLY_SIMILAR"

# Internal attribute names
TYPE_ATTR = "_type"
METADATA_ATTR = "_metadata"

# Schema version for compatibility checking
SCHEMA_VERSION = "1.0.0"