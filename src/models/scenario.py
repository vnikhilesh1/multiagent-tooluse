"""Scenario model for conversation generation.

A scenario describes a realistic user goal and the expected flow
of tool calls needed to accomplish it.
"""

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class Scenario(BaseModel):
    """Represents a planned conversation scenario.

    Attributes:
        description: Human-readable description of the scenario
        user_goal: What the user is trying to accomplish
        expected_flow: List of endpoint IDs in expected execution order
        disambiguation_points: Turn indices where clarification may be needed
        available_tools: Tools in Anthropic tool_use format

    Example:
        >>> scenario = Scenario(
        ...     description="User checks weather before booking a flight",
        ...     user_goal="Plan a trip to Paris with good weather",
        ...     expected_flow=["weather_api_get", "flight_api_search", "flight_api_book"],
        ...     disambiguation_points=[1],  # Clarify dates at turn 1
        ...     available_tools=[{"name": "weather_api_get", "description": "..."}]
        ... )
    """

    description: str
    user_goal: str
    expected_flow: List[str] = Field(default_factory=list)
    disambiguation_points: List[int] = Field(default_factory=list)
    available_tools: List[Dict[str, Any]] = Field(default_factory=list)
