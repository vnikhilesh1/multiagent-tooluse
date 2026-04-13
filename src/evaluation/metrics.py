"""Diversity and quality metrics for generated conversations.

This module provides functions to compute various metrics for
evaluating the diversity and quality of generated datasets.
"""

import math
from collections import Counter
from typing import Dict, List, Optional, Union


def compute_entropy(
    counts: Union[Counter, Dict[str, int], List[int]],
    normalize: bool = False,
) -> float:
    """Compute Shannon entropy of a distribution.

    Shannon entropy measures the uncertainty/diversity of a distribution.
    Higher entropy means more even distribution (better diversity).
    Lower entropy means some items dominate (poor diversity).

    Formula: H = -Σ(p_i * log(p_i)) where p_i = count_i / total

    Args:
        counts: Item counts as Counter, dict, or list of counts
        normalize: If True, normalize to [0, 1] range (divide by max entropy)

    Returns:
        Entropy value (float). Returns 0.0 for empty or single-item distributions.
        If normalize=True, returns value in [0, 1] where 1 = perfectly uniform.

    Examples:
        >>> compute_entropy(Counter({"a": 5, "b": 5}))  # Uniform
        0.6931471805599453
        >>> compute_entropy(Counter({"a": 10, "b": 0}))  # Skewed
        0.0
        >>> compute_entropy([1, 1, 1, 1])  # Uniform list
        1.3862943611198906
        >>> compute_entropy([1, 1, 1, 1], normalize=True)
        1.0
    """
    # Convert to list of counts
    if isinstance(counts, Counter):
        count_values = list(counts.values())
    elif isinstance(counts, dict):
        count_values = list(counts.values())
    elif isinstance(counts, list):
        count_values = counts
    else:
        raise TypeError(f"counts must be Counter, dict, or list, got {type(counts).__name__}")

    # Filter out zero and negative counts
    count_values = [c for c in count_values if c > 0]

    # Handle edge cases
    if len(count_values) == 0:
        return 0.0

    if len(count_values) == 1:
        return 0.0

    # Calculate total
    total = sum(count_values)
    if total == 0:
        return 0.0

    # Calculate entropy: H = -Σ(p_i * log(p_i))
    entropy = 0.0
    for count in count_values:
        if count > 0:
            p = count / total
            entropy -= p * math.log(p)

    # Optionally normalize to [0, 1]
    if normalize:
        # Maximum entropy is log(n) for n items (uniform distribution)
        max_entropy = math.log(len(count_values))
        if max_entropy > 0:
            entropy = entropy / max_entropy
        else:
            entropy = 0.0

    return entropy


def compute_tool_entropy(
    tool_counts: Union[Counter, Dict[str, int]],
    normalize: bool = False,
) -> float:
    """Compute entropy specifically for tool usage distribution.

    Wrapper around compute_entropy with tool-specific semantics.

    Args:
        tool_counts: Mapping of tool_id to usage count
        normalize: If True, normalize to [0, 1] range

    Returns:
        Tool usage entropy. Higher = more diverse tool usage.

    Example:
        >>> tool_counts = Counter({"weather_api": 50, "maps_api": 30, "calendar_api": 20})
        >>> compute_tool_entropy(tool_counts)
        1.0296530140645737
    """
    return compute_entropy(tool_counts, normalize=normalize)


def compute_domain_entropy(
    domain_counts: Union[Counter, Dict[str, int]],
    normalize: bool = False,
) -> float:
    """Compute entropy specifically for domain usage distribution.

    Wrapper around compute_entropy with domain-specific semantics.

    Args:
        domain_counts: Mapping of domain to usage count
        normalize: If True, normalize to [0, 1] range

    Returns:
        Domain usage entropy. Higher = more diverse domain coverage.

    Example:
        >>> domain_counts = Counter({"weather": 40, "productivity": 35, "social": 25})
        >>> compute_domain_entropy(domain_counts)
        1.0784128205616382
    """
    return compute_entropy(domain_counts, normalize=normalize)


def compute_max_entropy(n: int) -> float:
    """Compute maximum possible entropy for n items.

    Maximum entropy occurs with uniform distribution: H_max = log(n)

    Args:
        n: Number of unique items

    Returns:
        Maximum entropy value

    Example:
        >>> compute_max_entropy(10)
        2.302585092994046
    """
    if n <= 1:
        return 0.0
    return math.log(n)


def entropy_ratio(
    counts: Union[Counter, Dict[str, int], List[int]],
) -> float:
    """Compute entropy as ratio of maximum possible entropy.

    Equivalent to compute_entropy(counts, normalize=True).
    Returns value in [0, 1] where 1 = perfectly uniform distribution.

    Args:
        counts: Item counts

    Returns:
        Entropy ratio in [0, 1]

    Example:
        >>> entropy_ratio(Counter({"a": 5, "b": 5}))
        1.0
        >>> entropy_ratio(Counter({"a": 9, "b": 1}))
        0.4689955935892812
    """
    return compute_entropy(counts, normalize=True)


def compute_pair_ratio(
    pair_counts: Union[Counter, Dict[tuple, int]],
    total_tools: Optional[int] = None,
) -> float:
    """Compute ratio of unique tool pairs to possible pairs.

    Measures combinatorial diversity - how many tool combinations
    have been explored out of all possible combinations.

    Formula: ratio = unique_pairs / (n * (n-1) / 2)

    Args:
        pair_counts: Counter or dict mapping (tool1, tool2) tuples to counts
        total_tools: Optional total number of unique tools. If not provided,
                     inferred from the pairs in pair_counts.

    Returns:
        Ratio in [0, 1]. Returns 0.0 if fewer than 2 tools.
        1.0 means all possible pairs have been used at least once.

    Examples:
        >>> pairs = Counter({("a", "b"): 5, ("a", "c"): 3})
        >>> compute_pair_ratio(pairs, total_tools=3)  # 2 of 3 possible pairs
        0.6666666666666666
        >>> compute_pair_ratio(pairs)  # Infers 3 tools from pairs
        0.6666666666666666
    """
    # Count unique pairs (pairs with count > 0)
    if isinstance(pair_counts, Counter):
        unique_pairs = sum(1 for count in pair_counts.values() if count > 0)
    elif isinstance(pair_counts, dict):
        unique_pairs = sum(1 for count in pair_counts.values() if count > 0)
    else:
        raise TypeError(f"pair_counts must be Counter or dict, got {type(pair_counts).__name__}")

    # Determine number of unique tools
    if total_tools is not None:
        n = total_tools
    else:
        # Infer from pairs - collect all unique tool IDs
        tools = set()
        for pair in pair_counts.keys():
            if isinstance(pair, tuple) and len(pair) == 2:
                tools.add(pair[0])
                tools.add(pair[1])
        n = len(tools)

    # Handle edge cases
    if n < 2:
        return 0.0

    # Calculate possible pairs: n choose 2 = n * (n-1) / 2
    possible_pairs = n * (n - 1) // 2

    if possible_pairs == 0:
        return 0.0

    # Return ratio
    return unique_pairs / possible_pairs


def compute_tool_pair_coverage(
    pair_counts: Union[Counter, Dict[tuple, int]],
    tool_counts: Union[Counter, Dict[str, int]],
) -> float:
    """Compute pair coverage using tool_counts for total tools.

    Convenience wrapper that uses tool_counts to determine the
    total number of unique tools.

    Args:
        pair_counts: Counter mapping (tool1, tool2) tuples to counts
        tool_counts: Counter mapping tool_id to usage counts

    Returns:
        Ratio of unique pairs to possible pairs.

    Example:
        >>> tool_counts = Counter({"a": 10, "b": 5, "c": 3})
        >>> pair_counts = Counter({("a", "b"): 2, ("b", "c"): 1})
        >>> compute_tool_pair_coverage(pair_counts, tool_counts)
        0.6666666666666666
    """
    total_tools = len(tool_counts)
    return compute_pair_ratio(pair_counts, total_tools=total_tools)


def count_possible_pairs(n: int) -> int:
    """Calculate the number of possible pairs for n items.

    Formula: n choose 2 = n * (n-1) / 2

    Args:
        n: Number of unique items

    Returns:
        Number of possible unique pairs

    Examples:
        >>> count_possible_pairs(2)
        1
        >>> count_possible_pairs(5)
        10
        >>> count_possible_pairs(10)
        45
    """
    if n < 2:
        return 0
    return n * (n - 1) // 2
