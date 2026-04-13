"""Judge Agent for evaluating generated conversations.

This agent scores conversations on multiple quality dimensions
to ensure training data quality.
"""

from typing import TYPE_CHECKING, Any, Dict, List

from src.agents.base import BaseAgent
from src.models.judge_scores import JudgeScores

if TYPE_CHECKING:
    from src.llm import LLMClient
    from src.models.context import ConversationContext


class JudgeAgent(BaseAgent):
    """Agent that evaluates generated conversations.

    Scores conversations on:
    - Tool correctness: Were appropriate tools selected?
    - Argument grounding: Were values from prior outputs used?
    - Task completion: Was the user's goal achieved?
    - Naturalness: Is the dialogue realistic?

    Attributes:
        llm: LLMClient for evaluation
        name: Agent identifier

    Example:
        >>> agent = JudgeAgent(llm=client, name="judge")
        >>> context = agent.generate(context)  # Evaluate conversation
        >>> scores = context.grounding_values["judge_scores"]
        >>> print(scores.average)
        8.5
    """

    def __init__(
        self,
        llm: "LLMClient",
        name: str = "judge",
    ) -> None:
        """Initialize the judge agent.

        Args:
            llm: LLMClient instance for evaluation
            name: Agent identifier
        """
        super().__init__(llm=llm, name=name)

    def generate(self, context: "ConversationContext") -> "ConversationContext":
        """Evaluate the conversation and add scores to context.

        Builds an evaluation prompt from the conversation, uses the LLM
        to score it, and stores the results in context.grounding_values.

        Args:
            context: Conversation context to evaluate

        Returns:
            Context with judge_scores added to grounding_values
        """
        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(context)

        # Get structured scores from LLM
        scores = self.llm.complete_structured(
            prompt=prompt,
            response_model=JudgeScores,
            temperature=0.0,  # Deterministic evaluation
            max_tokens=1024,
        )

        # Store scores in context
        context.grounding_values["judge_scores"] = scores

        return context

    def _build_evaluation_prompt(self, context: "ConversationContext") -> str:
        """Build the evaluation prompt for the judge.

        Includes the full conversation, user goal, and scoring criteria.

        Args:
            context: Conversation context to evaluate

        Returns:
            Evaluation prompt string
        """
        # Get user goal
        scenario = context.grounding_values.get("scenario", {})
        user_goal = scenario.get("user_goal", context.scenario_description or "Unknown goal")

        # Format conversation
        conversation_lines = []
        for msg in context.messages:
            if msg.role == "user":
                conversation_lines.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                if msg.content:
                    conversation_lines.append(f"Assistant: {msg.content}")
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        func = tc.get("function", {})
                        conversation_lines.append(
                            f"Assistant [Tool Call]: {func.get('name', 'unknown')}({func.get('arguments', '{}')})"
                        )
            elif msg.role == "tool":
                conversation_lines.append(f"Tool Response: {msg.content[:200]}...")

        conversation_text = "\n".join(conversation_lines) if conversation_lines else "No messages"

        # Format tool outputs for grounding reference
        tool_outputs_text = ""
        if context.tool_outputs:
            output_lines = []
            for output in context.tool_outputs:
                output_lines.append(f"- {output.endpoint_id}: {output.result}")
            tool_outputs_text = f"""

Tool Outputs (for grounding reference):
{chr(10).join(output_lines)}"""

        prompt = f"""Evaluate this AI assistant conversation for training data quality.

## User's Goal
{user_goal}

## Conversation
{conversation_text}{tool_outputs_text}

## Scoring Criteria

Score each dimension from 1-10:

### 1. Tool Correctness (tool_correctness)
- Were the appropriate tools selected for the task?
- Were tools used in a logical order?
- Were unnecessary tools avoided?
- 10 = Perfect tool selection, 1 = Completely wrong tools

### 2. Argument Grounding (argument_grounding)
- Were tool arguments grounded in real values from prior outputs?
- Did the assistant use actual IDs, names, and values from tool responses?
- Were hallucinated or made-up values avoided?
- 10 = All values properly grounded, 1 = All values hallucinated

### 3. Task Completion (task_completion)
- Was the user's goal successfully achieved?
- Did the conversation reach a satisfying conclusion?
- Were all necessary steps completed?
- 10 = Goal fully achieved, 1 = Goal not addressed

### 4. Naturalness (naturalness)
- How realistic is the dialogue flow?
- Do the user messages sound like real user requests?
- Are the assistant responses appropriate and helpful?
- Is the conversation coherent and well-structured?
- 10 = Perfectly natural, 1 = Completely artificial

## Instructions
Provide scores for all four dimensions and explain your reasoning.
Be critical but fair - training data quality matters.

Return your evaluation as a JudgeScores object."""

        return prompt

    def __repr__(self) -> str:
        """Return string representation."""
        return f"JudgeAgent(name={self.name!r})"
