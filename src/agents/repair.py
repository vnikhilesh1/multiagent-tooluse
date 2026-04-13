"""Repair Agent for fixing low-quality conversations.

This agent attempts to repair conversations that received low scores
from the Judge Agent by asking the LLM to fix specific issues.
"""

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.agents.base import BaseAgent
from src.models.context import Message

if TYPE_CHECKING:
    from src.llm import LLMClient
    from src.models.context import ConversationContext
    from src.models.judge_scores import JudgeScores


class RepairAgent(BaseAgent):
    """Agent that repairs low-quality conversations.

    Analyzes judge scores to identify problem areas, then asks the LLM
    to fix those specific issues while preserving good aspects.

    Attributes:
        llm: LLMClient for repair generation
        name: Agent identifier
        repair_threshold: Score below which repair is attempted (default 6)

    Example:
        >>> agent = RepairAgent(llm=client, name="repair")
        >>> # Context has low judge_scores
        >>> repaired_context = agent.generate(context)
    """

    def __init__(
        self,
        llm: "LLMClient",
        name: str = "repair",
        repair_threshold: int = 6,
    ) -> None:
        """Initialize the repair agent.

        Args:
            llm: LLMClient instance for repair
            name: Agent identifier
            repair_threshold: Score threshold below which to repair
        """
        super().__init__(llm=llm, name=name)
        self.repair_threshold = repair_threshold

    def generate(self, context: "ConversationContext") -> "ConversationContext":
        """Attempt to repair a low-quality conversation.

        Checks if scores warrant repair, builds a repair prompt,
        and parses the repaired conversation. Falls back to original
        if repair fails.

        Args:
            context: Conversation context with judge_scores

        Returns:
            Repaired context, or original if repair fails
        """
        # Get judge scores
        scores = context.grounding_values.get("judge_scores")

        if not scores:
            # No scores, return original
            return context

        # Check if repair is needed
        if not self._needs_repair(scores):
            return context

        try:
            # Build repair prompt
            prompt = self._build_repair_prompt(context)

            # Get repaired conversation from LLM
            repair_data = self.llm.complete_json(
                prompt=prompt,
                temperature=0.3,
                max_tokens=2048,
            )

            # Parse the repaired conversation
            repaired_context = self._parse_repaired_conversation(repair_data, context)

            # Validate the repair
            if self._validate_repair(repaired_context):
                return repaired_context
            else:
                # Validation failed, return original
                return context

        except Exception:
            # Parse or LLM error, return original
            return context

    def _needs_repair(self, scores: "JudgeScores") -> bool:
        """Check if the conversation needs repair based on scores.

        Args:
            scores: JudgeScores from the judge agent

        Returns:
            True if any score is below threshold
        """
        return (
            scores.tool_correctness < self.repair_threshold or
            scores.argument_grounding < self.repair_threshold or
            scores.task_completion < self.repair_threshold or
            scores.naturalness < self.repair_threshold
        )

    def _serialize_conversation(self, context: "ConversationContext") -> str:
        """Serialize the conversation context to JSON.

        Args:
            context: Conversation context to serialize

        Returns:
            JSON string representation
        """
        messages_data = []
        for msg in context.messages:
            msg_dict = {
                "role": msg.role,
                "content": msg.content,
            }
            if msg.tool_calls:
                msg_dict["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                msg_dict["tool_call_id"] = msg.tool_call_id
            messages_data.append(msg_dict)

        # Include tool outputs
        tool_outputs_data = []
        for output in context.tool_outputs:
            tool_outputs_data.append({
                "endpoint_id": output.endpoint_id,
                "arguments": output.arguments,
                "result": output.result,
                "call_id": output.call_id,
            })

        data = {
            "messages": messages_data,
            "tool_outputs": tool_outputs_data,
            "scenario": context.grounding_values.get("scenario", {}),
        }

        return json.dumps(data, indent=2)

    def _get_low_score_feedback(self, scores: "JudgeScores") -> str:
        """Generate feedback focusing on low-scoring dimensions.

        Args:
            scores: JudgeScores with evaluation results

        Returns:
            Feedback string highlighting issues to fix
        """
        feedback_lines = []

        if scores.tool_correctness < self.repair_threshold:
            feedback_lines.append(
                f"- **Tool Correctness** ({scores.tool_correctness}/10): "
                "The tools selected may not be appropriate for the task. "
                "Ensure the right tools are used in the right order."
            )

        if scores.argument_grounding < self.repair_threshold:
            feedback_lines.append(
                f"- **Argument Grounding** ({scores.argument_grounding}/10): "
                "Tool arguments contain hallucinated or made-up values. "
                "Use ONLY values that come from prior tool outputs or user input."
            )

        if scores.task_completion < self.repair_threshold:
            feedback_lines.append(
                f"- **Task Completion** ({scores.task_completion}/10): "
                "The user's goal was not achieved. "
                "Ensure the conversation reaches a satisfying conclusion."
            )

        if scores.naturalness < self.repair_threshold:
            feedback_lines.append(
                f"- **Naturalness** ({scores.naturalness}/10): "
                "The dialogue feels artificial or unrealistic. "
                "Make messages sound more like real human conversation."
            )

        if scores.reasoning:
            feedback_lines.append(f"\nJudge's reasoning: {scores.reasoning}")

        return "\n".join(feedback_lines)

    def _build_repair_prompt(self, context: "ConversationContext") -> str:
        """Build the repair prompt for the LLM.

        Args:
            context: Conversation context to repair

        Returns:
            Repair prompt string
        """
        scores = context.grounding_values.get("judge_scores")
        serialized = self._serialize_conversation(context)

        # Get user goal
        scenario = context.grounding_values.get("scenario", {})
        user_goal = scenario.get("user_goal", context.scenario_description or "Unknown goal")

        # Get focused feedback
        feedback = self._get_low_score_feedback(scores) if scores else "General quality issues"

        prompt = f"""Repair this AI assistant conversation to fix the identified issues.

## User's Goal
{user_goal}

## Current Conversation (with issues)
{serialized}

## Issues to Fix
{feedback}

## Repair Instructions

1. Fix the specific issues identified above
2. Preserve the overall structure and flow
3. Keep tool calls that are correct
4. Ensure tool arguments use REAL values from:
   - Prior tool outputs (use actual IDs, names, values returned)
   - User input (use what the user actually said)
   - DO NOT make up or hallucinate any IDs, names, or values
5. Make sure the conversation achieves the user's goal
6. Keep dialogue natural and realistic

## Output Format

Return a JSON object with this structure:
{{
  "messages": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "...", "tool_calls": [...]}},
    {{"role": "tool", "content": "...", "tool_call_id": "..."}},
    ...
  ]
}}

Return ONLY the JSON object with the repaired conversation:"""

        return prompt

    def _parse_repaired_conversation(
        self,
        repair_data: Dict[str, Any],
        original_context: "ConversationContext",
    ) -> "ConversationContext":
        """Parse the repaired conversation from LLM response.

        Args:
            repair_data: Parsed JSON from LLM
            original_context: Original context for reference

        Returns:
            New ConversationContext with repaired messages
        """
        from src.models.context import ConversationContext

        # Create new context preserving metadata
        new_context = ConversationContext(
            tool_chain=original_context.tool_chain,
            target_steps=original_context.target_steps,
            scenario_description=original_context.scenario_description,
            seed=original_context.seed,
        )

        # Copy grounding values (except messages will be new)
        new_context.grounding_values = dict(original_context.grounding_values)

        # Parse messages
        messages = repair_data.get("messages", [])
        for msg_data in messages:
            role = msg_data.get("role", "user")
            content = msg_data.get("content", "")
            tool_calls = msg_data.get("tool_calls")
            tool_call_id = msg_data.get("tool_call_id")

            message = Message(
                role=role,
                content=content,
                tool_calls=tool_calls,
                tool_call_id=tool_call_id,
            )
            new_context.add_message(message)

        # Preserve tool outputs from original
        new_context.tool_outputs = list(original_context.tool_outputs)

        return new_context

    def _validate_repair(self, context: "ConversationContext") -> bool:
        """Validate that the repaired conversation has correct structure.

        Args:
            context: Repaired conversation context

        Returns:
            True if valid, False otherwise
        """
        # Must have at least one message
        if not context.messages:
            return False

        # Check message structure
        for msg in context.messages:
            if not hasattr(msg, 'role') or not hasattr(msg, 'content'):
                return False
            if msg.role not in ("user", "assistant", "tool", "system"):
                return False

        # Should start with user message (usually)
        if context.messages and context.messages[0].role not in ("user", "system"):
            # Allow but don't require - some repairs might be valid
            pass

        return True

    def __repr__(self) -> str:
        """Return string representation."""
        return f"RepairAgent(name={self.name!r}, threshold={self.repair_threshold})"
