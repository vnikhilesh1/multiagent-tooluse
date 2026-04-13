"""JSONL serialization for conversation datasets.

This module provides functions to serialize conversations to JSONL format
and read them back for analysis or training.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.context import ConversationContext, Message
    from src.models.judge_scores import JudgeScores
    from src.orchestrator import GenerationResult


def serialize_message(message: "Message") -> Dict[str, Any]:
    """Serialize a Message to a dictionary.

    Args:
        message: Message object to serialize

    Returns:
        Dict with role, content, and optional tool_calls/tool_call_id
    """
    result: Dict[str, Any] = {
        "role": message.role,
        "content": message.content,
    }

    # Include tool_calls for assistant messages
    if message.tool_calls:
        result["tool_calls"] = message.tool_calls

    # Include tool_call_id for tool messages
    if message.tool_call_id:
        result["tool_call_id"] = message.tool_call_id

    return result


def serialize_scores(scores: Optional["JudgeScores"]) -> Optional[Dict[str, Any]]:
    """Serialize JudgeScores to a dictionary.

    Args:
        scores: JudgeScores object or None

    Returns:
        Dict with all score dimensions or None
    """
    if scores is None:
        return None

    return {
        "tool_correctness": scores.tool_correctness,
        "argument_grounding": scores.argument_grounding,
        "task_completion": scores.task_completion,
        "naturalness": scores.naturalness,
        "reasoning": scores.reasoning,
        "average": scores.average,
    }


def serialize_conversation(
    result: "GenerationResult",
    include_metadata: bool = True,
) -> Dict[str, Any]:
    """Serialize a GenerationResult to a JSON-serializable dictionary.

    Args:
        result: GenerationResult from orchestrator
        include_metadata: Whether to include metadata fields

    Returns:
        Dict ready for JSON serialization

    Example:
        >>> data = serialize_conversation(result)
        >>> json_line = json.dumps(data)
    """
    if not result.conversation:
        return {
            "conversation_id": None,
            "messages": [],
            "judge_scores": None,
            "metadata": {
                "error": result.error,
                "success": False,
            } if include_metadata else None,
        }

    conv = result.conversation

    # Serialize messages
    messages = [serialize_message(msg) for msg in conv.messages]

    # Build the output dict
    output: Dict[str, Any] = {
        "conversation_id": conv.conversation_id,
        "messages": messages,
        "judge_scores": serialize_scores(result.scores),
    }

    # Add metadata if requested
    if include_metadata:
        # Extract tools used from tool_outputs
        tools_used = [tool_output.endpoint_id for tool_output in conv.tool_outputs]

        output["metadata"] = {
            "tools_used": tools_used,
            "num_turns": len(conv.messages),
            "num_tool_calls": len(conv.tool_outputs),
            "pattern_type": getattr(conv, "pattern_type", None),
            "scenario_description": conv.scenario_description,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "attempts": result.attempts,
            "repaired": result.repaired,
            "success": result.success,
        }

    return output


def write_dataset(
    results: List["GenerationResult"],
    output_path: Union[str, Path],
    include_failed: bool = False,
    include_metadata: bool = True,
) -> int:
    """Write a list of results to a JSONL file.

    Args:
        results: List of GenerationResult objects
        output_path: Path to output JSONL file
        include_failed: Whether to include failed generations
        include_metadata: Whether to include metadata in each record

    Returns:
        Number of conversations written

    Example:
        >>> count = write_dataset(results, "output/dataset.jsonl")
        >>> print(f"Wrote {count} conversations")
    """
    output_path = Path(output_path)

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            # Skip failed results unless requested
            if not include_failed and not result.success:
                continue

            data = serialize_conversation(result, include_metadata=include_metadata)
            json_line = json.dumps(data, ensure_ascii=False)
            f.write(json_line + "\n")
            count += 1

    return count


def read_dataset(
    input_path: Union[str, Path],
) -> Iterator[Dict[str, Any]]:
    """Read conversations from a JSONL file.

    Args:
        input_path: Path to JSONL file

    Yields:
        Dict for each conversation record

    Example:
        >>> for conv in read_dataset("output/dataset.jsonl"):
        ...     print(conv["conversation_id"])
    """
    input_path = Path(input_path)

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                yield json.loads(line)


def load_dataset(
    input_path: Union[str, Path],
) -> List[Dict[str, Any]]:
    """Load all conversations from a JSONL file into memory.

    Args:
        input_path: Path to JSONL file

    Returns:
        List of conversation dicts

    Example:
        >>> conversations = load_dataset("output/dataset.jsonl")
        >>> print(f"Loaded {len(conversations)} conversations")
    """
    return list(read_dataset(input_path))
