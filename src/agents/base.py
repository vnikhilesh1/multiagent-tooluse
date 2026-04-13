"""Base agent class for multi-agent conversation generation.

This module provides the abstract base class that all agents must inherit from.
Each agent type (UserAgent, AssistantAgent, ToolAgent, etc.) will implement
the generate() method to handle their specific role in conversation generation.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.llm import LLMClient
    from src.models.context import ConversationContext


class BaseAgent(ABC):
    """Abstract base class for all conversation agents.

    All agents in the multi-agent system must inherit from this class and
    implement the generate() method. The base class provides common
    infrastructure for LLM access and agent identification.

    Attributes:
        llm: LLMClient instance for making LLM API calls
        name: String identifier for this agent

    Example:
        >>> class MyAgent(BaseAgent):
        ...     def generate(self, context):
        ...         # Add a message to the context
        ...         context.add_message(Message(role="user", content="Hello"))
        ...         return context
        ...
        >>> agent = MyAgent(llm=client, name="my_agent")
        >>> context = agent.generate(context)
    """

    def __init__(self, llm: "LLMClient", name: str) -> None:
        """Initialize the base agent.

        Args:
            llm: LLMClient instance for LLM API calls
            name: String identifier for this agent
        """
        self.llm = llm
        self.name = name

    @abstractmethod
    def generate(self, context: "ConversationContext") -> "ConversationContext":
        """Generate the next part of the conversation.

        This method must be implemented by all subclasses. It takes the current
        conversation context, performs the agent's role (generating user input,
        assistant response, tool calls, etc.), and returns the updated context.

        Args:
            context: The current conversation context

        Returns:
            The updated conversation context after this agent's contribution

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass

    def __repr__(self) -> str:
        """Return string representation of the agent."""
        return f"{self.__class__.__name__}(name={self.name!r})"
