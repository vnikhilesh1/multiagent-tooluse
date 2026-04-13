"""Aggregation functions for dataset quality statistics.

This module provides functions to compute aggregate metrics
across a dataset of generated conversations.
"""

from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.judge_scores import JudgeScores
    from src.orchestrator import GenerationResult


@dataclass
class AggregateStats:
    """Aggregate statistics for a dataset of conversations.

    Attributes:
        total_conversations: Total number of conversations
        successful_conversations: Number of successful generations
        mean_tool_correctness: Average tool correctness score
        mean_argument_grounding: Average argument grounding score
        mean_task_completion: Average task completion score
        mean_naturalness: Average naturalness score
        mean_overall: Average of all dimension averages
        pass_rate: Percentage of conversations above threshold
        repair_rate: Percentage of conversations that were repaired
        multi_step_rate: Percentage with 3+ tool calls
        multi_tool_rate: Percentage with 2+ distinct tools
        total_attempts: Total generation attempts
        mean_attempts: Average attempts per conversation
    """
    total_conversations: int = 0
    successful_conversations: int = 0
    mean_tool_correctness: float = 0.0
    mean_argument_grounding: float = 0.0
    mean_task_completion: float = 0.0
    mean_naturalness: float = 0.0
    mean_overall: float = 0.0
    pass_rate: float = 0.0
    repair_rate: float = 0.0
    multi_step_rate: float = 0.0
    multi_tool_rate: float = 0.0
    total_attempts: int = 0
    mean_attempts: float = 0.0


def aggregate_scores(
    results: List["GenerationResult"],
    pass_threshold: float = 6.0,
) -> AggregateStats:
    """Compute aggregate statistics for a dataset.

    Args:
        results: List of GenerationResult from orchestrator
        pass_threshold: Minimum average score to count as "passing"

    Returns:
        AggregateStats with all computed metrics

    Example:
        >>> stats = aggregate_scores(results, pass_threshold=6.0)
        >>> print(f"Pass rate: {stats.pass_rate:.1%}")
        >>> print(f"Mean overall: {stats.mean_overall:.2f}")
    """
    stats = AggregateStats()

    if not results:
        return stats

    # Filter to successful results with scores
    successful = [r for r in results if r.success and r.scores is not None]

    stats.total_conversations = len(results)
    stats.successful_conversations = len(successful)
    stats.total_attempts = sum(r.attempts for r in results)

    if not successful:
        return stats

    # Calculate mean attempts
    stats.mean_attempts = stats.total_attempts / stats.total_conversations

    # Collect scores
    tool_correctness_scores = []
    argument_grounding_scores = []
    task_completion_scores = []
    naturalness_scores = []
    overall_scores = []

    passing_count = 0
    repaired_count = 0
    multi_step_count = 0
    multi_tool_count = 0

    for result in successful:
        scores = result.scores

        # Collect individual dimension scores
        tool_correctness_scores.append(scores.tool_correctness)
        argument_grounding_scores.append(scores.argument_grounding)
        task_completion_scores.append(scores.task_completion)
        naturalness_scores.append(scores.naturalness)
        overall_scores.append(scores.average)

        # Count passing
        if scores.average >= pass_threshold:
            passing_count += 1

        # Count repaired
        if result.repaired:
            repaired_count += 1

        # Count multi-step (3+ tool calls)
        if result.conversation:
            tool_count = len(result.conversation.tool_outputs)
            if tool_count >= 3:
                multi_step_count += 1

            # Count multi-tool (2+ distinct tools)
            unique_tools = set()
            for output in result.conversation.tool_outputs:
                # Extract tool_id from endpoint_id (format: tool_id.endpoint_name or just endpoint_id)
                endpoint_id = output.endpoint_id
                # Try to get tool_id - might be stored or inferred from endpoint
                tool_id = getattr(output, 'tool_id', None) or (endpoint_id.split('.')[0] if '.' in endpoint_id else endpoint_id)
                unique_tools.add(tool_id)

            if len(unique_tools) >= 2:
                multi_tool_count += 1

    # Calculate means
    n = len(successful)
    stats.mean_tool_correctness = sum(tool_correctness_scores) / n
    stats.mean_argument_grounding = sum(argument_grounding_scores) / n
    stats.mean_task_completion = sum(task_completion_scores) / n
    stats.mean_naturalness = sum(naturalness_scores) / n
    stats.mean_overall = sum(overall_scores) / n

    # Calculate rates (as percentages 0-1)
    stats.pass_rate = passing_count / n
    stats.repair_rate = repaired_count / n
    stats.multi_step_rate = multi_step_count / n
    stats.multi_tool_rate = multi_tool_count / n

    return stats


def compute_mean_scores(
    scores_list: List["JudgeScores"],
) -> dict:
    """Compute mean scores from a list of JudgeScores.

    Simpler function that just computes means without full aggregation.

    Args:
        scores_list: List of JudgeScores objects

    Returns:
        Dict with mean values for each dimension

    Example:
        >>> means = compute_mean_scores(scores_list)
        >>> print(means["tool_correctness"])
    """
    if not scores_list:
        return {
            "tool_correctness": 0.0,
            "argument_grounding": 0.0,
            "task_completion": 0.0,
            "naturalness": 0.0,
            "overall": 0.0,
        }

    n = len(scores_list)
    return {
        "tool_correctness": sum(s.tool_correctness for s in scores_list) / n,
        "argument_grounding": sum(s.argument_grounding for s in scores_list) / n,
        "task_completion": sum(s.task_completion for s in scores_list) / n,
        "naturalness": sum(s.naturalness for s in scores_list) / n,
        "overall": sum(s.average for s in scores_list) / n,
    }


def compute_pass_rate(
    scores_list: List["JudgeScores"],
    threshold: float = 6.0,
) -> float:
    """Compute the pass rate for a list of scores.

    Args:
        scores_list: List of JudgeScores objects
        threshold: Minimum average score to pass

    Returns:
        Pass rate as float in [0, 1]
    """
    if not scores_list:
        return 0.0

    passing = sum(1 for s in scores_list if s.average >= threshold)
    return passing / len(scores_list)


def format_stats_report(stats: AggregateStats) -> str:
    """Format aggregate stats as a readable report.

    Args:
        stats: AggregateStats to format

    Returns:
        Formatted string report
    """
    lines = [
        "=" * 50,
        "Dataset Quality Report",
        "=" * 50,
        "",
        f"Total Conversations: {stats.total_conversations}",
        f"Successful: {stats.successful_conversations}",
        f"Total Attempts: {stats.total_attempts}",
        f"Mean Attempts: {stats.mean_attempts:.2f}",
        "",
        "--- Quality Scores (1-10) ---",
        f"Tool Correctness:    {stats.mean_tool_correctness:.2f}",
        f"Argument Grounding:  {stats.mean_argument_grounding:.2f}",
        f"Task Completion:     {stats.mean_task_completion:.2f}",
        f"Naturalness:         {stats.mean_naturalness:.2f}",
        f"Overall Average:     {stats.mean_overall:.2f}",
        "",
        "--- Rates ---",
        f"Pass Rate:           {stats.pass_rate:.1%}",
        f"Repair Rate:         {stats.repair_rate:.1%}",
        f"Multi-Step Rate:     {stats.multi_step_rate:.1%}",
        f"Multi-Tool Rate:     {stats.multi_tool_rate:.1%}",
        "=" * 50,
    ]
    return "\n".join(lines)
