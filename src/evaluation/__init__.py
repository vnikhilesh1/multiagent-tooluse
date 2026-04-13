"""Evaluation module for conversation quality assessment.

This module provides validators, metrics, and scoring functions
for evaluating generated conversations.
"""

from src.evaluation.validator import (
    validate_structure,
    validate_conversation_dict,
    VALID_ROLES,
)
from src.evaluation.metrics import (
    compute_entropy,
    compute_tool_entropy,
    compute_domain_entropy,
    compute_max_entropy,
    entropy_ratio,
    # Pair ratio functions
    compute_pair_ratio,
    compute_tool_pair_coverage,
    count_possible_pairs,
)
from src.evaluation.aggregation import (
    AggregateStats,
    aggregate_scores,
    compute_mean_scores,
    compute_pass_rate,
    format_stats_report,
)
from src.evaluation.serialization import (
    serialize_message,
    serialize_scores,
    serialize_conversation,
    write_dataset,
    read_dataset,
    load_dataset,
)

__all__ = [
    # Validator
    "validate_structure",
    "validate_conversation_dict",
    "VALID_ROLES",
    # Entropy Metrics
    "compute_entropy",
    "compute_tool_entropy",
    "compute_domain_entropy",
    "compute_max_entropy",
    "entropy_ratio",
    # Pair Ratio Metrics
    "compute_pair_ratio",
    "compute_tool_pair_coverage",
    "count_possible_pairs",
    # Aggregation
    "AggregateStats",
    "aggregate_scores",
    "compute_mean_scores",
    "compute_pass_rate",
    "format_stats_report",
    # Serialization
    "serialize_message",
    "serialize_scores",
    "serialize_conversation",
    "write_dataset",
    "read_dataset",
    "load_dataset",
]
