"""Conversation context models for multi-agent tool-use generation.

This module provides dataclasses for tracking conversation state during
tool chain execution, including messages, tool outputs, and context values.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class Message:
    """Represents a single message in a conversation.

    Attributes:
        role: The message role ("user", "assistant", "system", "tool")
        content: The message content text
        tool_calls: Optional list of tool calls (for assistant messages)
        tool_call_id: Optional tool call ID (for tool response messages)
        timestamp: When the message was created (auto-generated)

    Example:
        >>> msg = Message(role="user", content="What's the weather?")
        >>> msg.timestamp  # Auto-generated
        datetime.datetime(2024, 1, 15, 10, 30, 45, 123456)

        >>> # Assistant message with tool calls
        >>> msg = Message(
        ...     role="assistant",
        ...     content="",
        ...     tool_calls=[{"id": "call_123", "function": {"name": "get_weather"}}]
        ... )

        >>> # Tool response message
        >>> msg = Message(
        ...     role="tool",
        ...     content='{"temperature": 72}',
        ...     tool_call_id="call_123"
        ... )
    """

    role: str
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ToolOutput:
    """Represents the output from a tool/endpoint call.

    Attributes:
        endpoint_id: The endpoint that was called
        arguments: Dict of arguments passed to the tool
        result: Dict of returned data from the tool
        call_id: Unique identifier linking to the tool call in messages
        timestamp: When the tool was called (auto-generated)
        success: Whether the call succeeded (default True)

    Example:
        >>> output = ToolOutput(
        ...     endpoint_id="weather_api_get_current",
        ...     arguments={"location": "Seattle", "units": "fahrenheit"},
        ...     result={"temperature": 65, "condition": "cloudy"},
        ...     call_id="call_abc123"
        ... )
        >>> output.success
        True
        >>> output.timestamp  # Auto-generated
        datetime.datetime(2024, 1, 15, 10, 30, 45)

        >>> # Error case
        >>> error_output = ToolOutput(
        ...     endpoint_id="database_query",
        ...     arguments={"query": "SELECT ..."},
        ...     result={"error": "Connection failed"},
        ...     call_id="call_db001",
        ...     success=False
        ... )
    """

    endpoint_id: str
    arguments: Dict[str, Any]
    result: Dict[str, Any]
    call_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True


def _generate_conversation_id() -> str:
    """Generate a unique conversation ID."""
    return str(uuid.uuid4())


@dataclass
class ConversationContext:
    """Tracks all state during conversation generation.

    This dataclass maintains the complete context needed for multi-agent
    tool-use conversation generation, including the tool chain to execute,
    generated values, and conversation history.

    Attributes:
        conversation_id: Unique identifier for this conversation (auto-generated)
        messages: List of Message objects in conversation order
        tool_outputs: List of ToolOutput objects from executed tools
        generated_ids: Dict mapping entity type to generated ID (e.g., "user_id" → "U123")
        grounding_values: Dict of extracted referenceable values from tool outputs
        tool_chain: List of endpoint IDs to execute in order
        current_step: Current position in tool_chain (0-indexed)
        target_steps: Total number of steps to execute
        scenario_description: Human-readable description of the scenario
        seed: Optional random seed for reproducibility
        start_time: Timestamp when generation started (auto-generated)

    Example:
        >>> ctx = ConversationContext(
        ...     tool_chain=["weather_get", "format_response"],
        ...     target_steps=2,
        ...     scenario_description="User checks weather in Seattle"
        ... )
        >>> ctx.current_step
        0
        >>> ctx.conversation_id  # Auto-generated UUID
        'a1b2c3d4-...'
    """

    # Auto-generated fields
    conversation_id: str = field(default_factory=_generate_conversation_id)
    start_time: datetime = field(default_factory=datetime.now)

    # Conversation state
    messages: List[Message] = field(default_factory=list)
    tool_outputs: List[ToolOutput] = field(default_factory=list)

    # ID and value tracking
    generated_ids: Dict[str, str] = field(default_factory=dict)
    grounding_values: Dict[str, Any] = field(default_factory=dict)

    # Tool chain execution
    tool_chain: List[str] = field(default_factory=list)
    current_step: int = 0
    target_steps: int = 0

    # Scenario metadata
    scenario_description: str = ""
    seed: Optional[int] = None

    def add_message(self, message: Message) -> None:
        """Append a message to the conversation.

        Args:
            message: Message object to add
        """
        self.messages.append(message)

    def add_tool_output(self, output: ToolOutput) -> None:
        """Append a tool output and increment the step counter.

        Args:
            output: ToolOutput object to add
        """
        self.tool_outputs.append(output)
        self.current_step += 1

    def get_history_for_prompt(self) -> str:
        """Format conversation history for prompt injection.

        Returns:
            Formatted string with "User: ...\nAssistant: ..." format
        """
        lines = []
        for msg in self.messages:
            if msg.role == "user":
                lines.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                if msg.content:  # Skip empty assistant messages (tool calls only)
                    lines.append(f"Assistant: {msg.content}")
            elif msg.role == "system":
                lines.append(f"System: {msg.content}")
            # Skip tool messages for cleaner history
        return "\n".join(lines)

    def get_available_values(self) -> str:
        """Format grounding values and generated IDs for prompt injection.

        Returns:
            Formatted string listing available values
        """
        lines = []

        if self.grounding_values:
            lines.append("Available values:")
            for key, value in self.grounding_values.items():
                lines.append(f"  - {key}: {value}")

        if self.generated_ids:
            lines.append("Generated IDs:")
            for key, value in self.generated_ids.items():
                lines.append(f"  - {key}: {value}")

        return "\n".join(lines)

    def generate_id(self, entity_type: str) -> str:
        """Generate or retrieve a unique ID for an entity type.

        This method implements lazy ID generation - it only generates a new ID
        if one doesn't already exist for the given entity type. This ensures
        consistency within a conversation (the same entity type always gets
        the same ID).

        Args:
            entity_type: The type of entity (e.g., "user", "order", "product")

        Returns:
            A unique ID in the format "{entity_type}_{8_hex_chars}"

        Example:
            >>> ctx = ConversationContext()
            >>> user_id = ctx.generate_id("user")
            >>> user_id
            'user_a1b2c3d4'
            >>> ctx.generate_id("user")  # Returns same ID
            'user_a1b2c3d4'
            >>> ctx.generate_id("order")  # Different entity type
            'order_e5f6a7b8'
        """
        # Check if ID already exists for this entity type
        if entity_type in self.generated_ids:
            return self.generated_ids[entity_type]

        # Generate new ID: entity_type + underscore + 8 hex chars from UUID
        new_id = f"{entity_type}_{uuid.uuid4().hex[:8]}"

        # Store for future reference
        self.generated_ids[entity_type] = new_id

        return new_id

    @property
    def is_complete(self) -> bool:
        """Check if the conversation has reached its target steps.

        Returns:
            True if current_step >= target_steps
        """
        return self.current_step >= self.target_steps

    def to_conversation(self) -> Dict[str, Any]:
        """Convert context to final output dictionary.

        Returns:
            Dict containing conversation_id, messages, tool_outputs, and metadata
        """
        return {
            "conversation_id": self.conversation_id,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "tool_calls": msg.tool_calls,
                    "tool_call_id": msg.tool_call_id,
                    "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                }
                for msg in self.messages
            ],
            "tool_outputs": [
                {
                    "endpoint_id": out.endpoint_id,
                    "arguments": out.arguments,
                    "result": out.result,
                    "call_id": out.call_id,
                    "timestamp": out.timestamp.isoformat() if out.timestamp else None,
                    "success": out.success,
                }
                for out in self.tool_outputs
            ],
            "metadata": {
                "scenario_description": self.scenario_description,
                "tool_chain": self.tool_chain,
                "target_steps": self.target_steps,
                "seed": self.seed,
                "start_time": self.start_time.isoformat() if self.start_time else None,
            },
        }
