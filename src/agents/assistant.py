"""Assistant Agent for generating responses with function calling.

This agent generates assistant responses that may include tool calls
to help accomplish the user's goal.
"""

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.agents.base import BaseAgent
from src.models.context import Message

if TYPE_CHECKING:
    from src.llm import LLMClient, LLMResponse
    from src.models.context import ConversationContext


class AssistantAgent(BaseAgent):
    """Agent that generates assistant responses with tool calling.

    Generates responses that:
    - Respond helpfully to user messages
    - Use available tools when appropriate
    - Make progress toward the user's goal
    - Ask for clarification when needed

    Attributes:
        llm: LLMClient for generating responses
        name: Agent identifier

    Example:
        >>> agent = AssistantAgent(llm=client, name="assistant")
        >>> context.add_message(Message(role="user", content="What's the weather?"))
        >>> context = agent.generate(context)
        >>> print(context.messages[-1].tool_calls)
        [{"id": "call_123", "function": {"name": "weather_get", ...}}]
    """

    def __init__(
        self,
        llm: "LLMClient",
        name: str = "assistant",
    ) -> None:
        """Initialize the assistant agent.

        Args:
            llm: LLMClient instance for LLM calls
            name: Agent identifier
        """
        super().__init__(llm=llm, name=name)

    def generate(self, context: "ConversationContext") -> "ConversationContext":
        """Generate an assistant response based on conversation context.

        Builds the conversation history, gets available tools, and calls
        the LLM to generate a response that may include tool calls.

        Args:
            context: Current conversation context

        Returns:
            Updated context with new assistant message added
        """
        # Build system prompt with user goal and grounding values
        system_prompt = self._build_system_prompt(context)

        # Build conversation messages for API
        messages = self._build_messages(context)

        # Get available tools
        tools = self._get_available_tools(context)

        # Call LLM with tools
        if tools:
            response = self.llm.chat(
                messages=messages,
                tools=tools,
                system=system_prompt,
                temperature=0.3,
                max_tokens=1024,
            )
        else:
            response = self.llm.chat(
                messages=messages,
                system=system_prompt,
                temperature=0.3,
                max_tokens=1024,
            )

        # Extract text content and tool calls from response
        text_content = self._extract_text_content(response.raw_response)
        tool_calls = self._parse_tool_calls(response.raw_response)

        # Create and add assistant message
        message = Message(
            role="assistant",
            content=text_content,
            tool_calls=tool_calls if tool_calls else None,
        )
        context.add_message(message)

        return context

    def _build_system_prompt(self, context: "ConversationContext") -> str:
        """Build the system prompt for the assistant.

        Includes the user's goal and any available grounding values
        to help the assistant provide accurate responses.

        Args:
            context: Current conversation context

        Returns:
            System prompt string
        """
        scenario = context.grounding_values.get("scenario", {})
        user_goal = scenario.get("user_goal", context.scenario_description or "Help the user")

        # Build grounding values section
        grounding_section = ""
        grounding_values = {k: v for k, v in context.grounding_values.items() if k != "scenario"}
        if grounding_values:
            grounding_lines = []
            for key, value in grounding_values.items():
                if isinstance(value, dict):
                    grounding_lines.append(f"- {key}: {json.dumps(value)}")
                else:
                    grounding_lines.append(f"- {key}: {value}")
            grounding_section = f"""

Available information:
{chr(10).join(grounding_lines)}"""

        prompt = f"""You are a helpful AI assistant helping a user accomplish their goal.

User's goal: {user_goal}

Instructions:
- Be helpful and concise
- Use the available tools when they can help accomplish the user's goal
- If you need more information, ask the user
- When you have the information needed, provide a clear response
- Use grounding values when available to provide accurate information{grounding_section}"""

        return prompt

    def _build_messages(self, context: "ConversationContext") -> List[Dict[str, Any]]:
        """Build the message list for the API call.

        Converts Message objects to the format expected by the Anthropic API,
        handling tool calls and tool responses appropriately.

        Args:
            context: Current conversation context

        Returns:
            List of message dicts for the API
        """
        messages = []

        for msg in context.messages:
            if msg.role == "user":
                messages.append({
                    "role": "user",
                    "content": msg.content,
                })
            elif msg.role == "assistant":
                # Handle assistant messages with potential tool calls
                content = []

                if msg.content:
                    content.append({"type": "text", "text": msg.content})

                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        # Convert from our format to Anthropic format
                        tool_use = {
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": tc.get("function", {}).get("name", ""),
                            "input": json.loads(tc.get("function", {}).get("arguments", "{}")),
                        }
                        content.append(tool_use)

                if not content:
                    content = [{"type": "text", "text": ""}]

                messages.append({
                    "role": "assistant",
                    "content": content,
                })
            elif msg.role == "tool":
                # Tool response message
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id or "",
                            "content": msg.content,
                        }
                    ],
                })
            elif msg.role == "system":
                # System messages are handled separately
                pass

        return messages

    def _get_available_tools(self, context: "ConversationContext") -> List[Dict[str, Any]]:
        """Get the list of available tools from context.

        Retrieves tools from the scenario's available_tools list.
        Optionally filters out tools that have already been used.

        Args:
            context: Current conversation context

        Returns:
            List of tool definitions in Anthropic format
        """
        scenario = context.grounding_values.get("scenario", {})
        available_tools = scenario.get("available_tools", [])

        # Return all available tools (filtering is optional)
        return available_tools

    def _parse_tool_calls(
        self, raw_response: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Parse tool calls from the raw API response.

        Extracts tool_use blocks from the response and converts them
        to our internal format compatible with Message.tool_calls.

        Args:
            raw_response: Raw response dict from LLM

        Returns:
            List of tool call dicts in format:
            [{"id": "...", "function": {"name": "...", "arguments": "..."}}]
        """
        if not raw_response:
            return []

        tool_calls = []
        content_blocks = raw_response.get("content", [])

        for block in content_blocks:
            if block.get("type") == "tool_use":
                tool_call = {
                    "id": block.get("id", ""),
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                }
                tool_calls.append(tool_call)

        return tool_calls

    def _extract_text_content(
        self, raw_response: Optional[Dict[str, Any]]
    ) -> str:
        """Extract text content from the raw API response.

        Concatenates all text blocks from the response.

        Args:
            raw_response: Raw response dict from LLM

        Returns:
            Combined text content string
        """
        if not raw_response:
            return ""

        text_parts = []
        content_blocks = raw_response.get("content", [])

        for block in content_blocks:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))

        return "".join(text_parts)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"AssistantAgent(name={self.name!r})"
