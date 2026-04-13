"""Judge scores model for conversation evaluation.

Provides a structured format for scoring generated conversations
across multiple quality dimensions.
"""

from pydantic import BaseModel, Field, computed_field


class JudgeScores(BaseModel):
    """Scores for evaluating a generated conversation.

    Each score is on a 1-10 scale where:
    - 1-3: Poor quality, significant issues
    - 4-6: Acceptable but has problems
    - 7-8: Good quality, minor issues
    - 9-10: Excellent quality

    Attributes:
        tool_correctness: How appropriate were the tool selections?
        argument_grounding: Were arguments grounded in prior values?
        task_completion: Was the user's goal achieved?
        naturalness: How realistic is the dialogue flow?
        reasoning: Explanation justifying the scores

    Example:
        >>> scores = JudgeScores(
        ...     tool_correctness=8,
        ...     argument_grounding=7,
        ...     task_completion=9,
        ...     naturalness=8,
        ...     reasoning="Good tool usage, natural flow, goal achieved."
        ... )
        >>> scores.average
        8.0
    """

    tool_correctness: int = Field(
        ...,
        ge=1,
        le=10,
        description="Score 1-10: Were the right tools selected for the task?"
    )
    argument_grounding: int = Field(
        ...,
        ge=1,
        le=10,
        description="Score 1-10: Were tool arguments grounded in real values from prior outputs?"
    )
    task_completion: int = Field(
        ...,
        ge=1,
        le=10,
        description="Score 1-10: Was the user's goal successfully achieved?"
    )
    naturalness: int = Field(
        ...,
        ge=1,
        le=10,
        description="Score 1-10: How realistic and natural is the conversation flow?"
    )
    reasoning: str = Field(
        ...,
        description="Explanation and justification for the scores given"
    )

    @computed_field
    @property
    def average(self) -> float:
        """Calculate the mean of the four numeric scores.

        Returns:
            Average score as a float
        """
        return (
            self.tool_correctness +
            self.argument_grounding +
            self.task_completion +
            self.naturalness
        ) / 4
