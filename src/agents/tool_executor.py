"""Tool Executor Agent for generating mock tool responses.

This agent executes tool calls by generating realistic mock responses
using the LLM, with fallback to basic mocks if needed.
"""

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.agents.base import BaseAgent
from src.models.context import Message, ToolOutput

if TYPE_CHECKING:
    from src.llm import LLMClient
    from src.models.context import ConversationContext


class ToolExecutorAgent(BaseAgent):
    """Agent that executes tool calls with LLM-generated mock responses.

    Finds pending tool calls in the conversation, generates realistic
    mock responses using the LLM, and adds the results to the context.

    Attributes:
        llm: LLMClient for generating mock responses
        name: Agent identifier

    Example:
        >>> agent = ToolExecutorAgent(llm=client, name="executor")
        >>> # Context has assistant message with tool_calls
        >>> context = agent.generate(context)
        >>> print(context.tool_outputs[-1].result)
        {"temperature": 72, "condition": "sunny"}
    """

    def __init__(
        self,
        llm: "LLMClient",
        name: str = "tool_executor",
    ) -> None:
        """Initialize the tool executor agent.

        Args:
            llm: LLMClient instance for generating mocks
            name: Agent identifier
        """
        super().__init__(llm=llm, name=name)

    def generate(self, context: "ConversationContext") -> "ConversationContext":
        """Execute a pending tool call and add the result to context.

        Finds the most recent unprocessed tool call, generates a mock
        response, and adds both a ToolOutput and tool Message to context.

        Args:
            context: Current conversation context

        Returns:
            Updated context with tool output and message added
        """
        # Get the pending tool call
        tool_call = self._get_pending_tool_call(context)

        if not tool_call:
            # No pending tool call, return context unchanged
            return context

        # Extract tool call details
        call_id = tool_call.get("id", "")
        function_info = tool_call.get("function", {})
        endpoint_id = function_info.get("name", "")
        arguments_str = function_info.get("arguments", "{}")

        # Parse arguments
        try:
            arguments = json.loads(arguments_str)
        except json.JSONDecodeError:
            arguments = {}

        # Generate mock response
        try:
            prompt = self._build_execution_prompt(tool_call, context)
            result = self.llm.complete_json(
                prompt=prompt,
                temperature=0.3,
                max_tokens=512,
            )
        except Exception:
            # Use fallback mock if LLM fails
            result = self._create_fallback_mock(tool_call)

        # Create ToolOutput
        tool_output = ToolOutput(
            endpoint_id=endpoint_id,
            arguments=arguments,
            result=result,
            call_id=call_id,
            success=True,
        )

        # Add ToolOutput to context (this increments current_step)
        context.add_tool_output(tool_output)

        # Extract and store grounding values from result
        self._extract_grounding_values(result, context)

        # Create tool message with JSON result
        tool_message = Message(
            role="tool",
            content=json.dumps(result),
            tool_call_id=call_id,
        )
        context.add_message(tool_message)

        return context

    def _get_pending_tool_call(
        self, context: "ConversationContext"
    ) -> Optional[Dict[str, Any]]:
        """Find the most recent unprocessed tool call.

        Looks at the last assistant message for tool_calls that haven't
        been responded to yet.

        Args:
            context: Current conversation context

        Returns:
            Tool call dict if found, None otherwise
        """
        # Find the last assistant message with tool_calls
        for msg in reversed(context.messages):
            if msg.role == "assistant" and msg.tool_calls:
                # Check which tool calls have been processed
                processed_ids = {
                    m.tool_call_id
                    for m in context.messages
                    if m.role == "tool" and m.tool_call_id
                }

                # Find first unprocessed tool call
                for tc in msg.tool_calls:
                    if tc.get("id") not in processed_ids:
                        return tc

                # All tool calls processed
                return None

        return None

    def _build_execution_prompt(
        self, tool_call: Dict[str, Any], context: "ConversationContext"
    ) -> str:
        """Build the prompt for generating a mock tool response.

        Includes endpoint details, arguments, and grounding values to
        ensure the mock response is contextually appropriate.

        Args:
            tool_call: The tool call to execute
            context: Current conversation context

        Returns:
            Prompt string for LLM
        """
        function_info = tool_call.get("function", {})
        endpoint_id = function_info.get("name", "unknown")
        arguments_str = function_info.get("arguments", "{}")

        try:
            arguments = json.loads(arguments_str)
        except json.JSONDecodeError:
            arguments = {}

        # Format arguments
        args_formatted = "\n".join(
            f"  - {k}: {v}" for k, v in arguments.items()
        ) if arguments else "  (no arguments)"

        # Format grounding values (exclude scenario metadata)
        grounding_formatted = ""
        grounding_values = {
            k: v for k, v in context.grounding_values.items()
            if k != "scenario"
        }
        if grounding_values:
            grounding_lines = []
            for k, v in grounding_values.items():
                if isinstance(v, dict):
                    grounding_lines.append(f"  - {k}: {json.dumps(v)}")
                else:
                    grounding_lines.append(f"  - {k}: {v}")
            grounding_formatted = f"""

Previously established values (use these for consistency):
{chr(10).join(grounding_lines)}"""

        # Get scenario context
        scenario = context.grounding_values.get("scenario", {})
        user_goal = scenario.get("user_goal", "")
        goal_section = f"\nUser's goal: {user_goal}" if user_goal else ""

        prompt = f"""Generate a realistic mock API response for this tool call.

Tool/Endpoint: {endpoint_id}
Arguments:
{args_formatted}{goal_section}{grounding_formatted}

Requirements:
1. Return valid JSON that a real API would return
2. Include realistic data that matches the arguments
3. Use proper IDs (e.g., "order_12345", "user_abc123")
4. Reference previously established values when relevant
5. Include appropriate fields for this type of API (status, data, metadata, etc.)

Generate ONLY the JSON response object, no explanation:"""

        return prompt

    def _create_fallback_mock(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Create a basic fallback mock when LLM fails.

        Generates a simple but valid response structure.

        Args:
            tool_call: The tool call that needs a response

        Returns:
            Basic mock response dict
        """
        function_info = tool_call.get("function", {})
        endpoint_id = function_info.get("name", "unknown")
        call_id = tool_call.get("id", "call_unknown")

        # Generate a simple ID based on endpoint
        mock_id = f"{endpoint_id}_{call_id[-8:]}" if len(call_id) > 8 else f"{endpoint_id}_001"

        return {
            "status": "success",
            "id": mock_id,
            "data": {
                "message": f"Mock response for {endpoint_id}",
                "timestamp": "2024-01-15T10:30:00Z",
            },
        }

    def _extract_grounding_values(
        self, result: Dict[str, Any], context: "ConversationContext"
    ) -> None:
        """Extract useful values from the result for grounding.

        Stores important values like IDs, names, and key data points
        in context.grounding_values for use in later turns.

        Args:
            result: The tool execution result
            context: Current conversation context
        """
        # Extract common grounding value patterns
        grounding_keys = [
            "id", "user_id", "order_id", "booking_id", "transaction_id",
            "name", "email", "phone",
            "total", "amount", "price", "balance",
            "status", "confirmation_number",
            "temperature", "weather", "condition",
            "flight_number", "departure", "arrival",
        ]

        def extract_from_dict(d: Dict[str, Any], prefix: str = "") -> None:
            for key, value in d.items():
                full_key = f"{prefix}{key}" if prefix else key

                # Store if it's a grounding key
                if key.lower() in grounding_keys or any(gk in key.lower() for gk in grounding_keys):
                    if value is not None and not isinstance(value, (dict, list)):
                        context.grounding_values[full_key] = value

                # Recurse into nested dicts
                if isinstance(value, dict):
                    extract_from_dict(value, f"{full_key}_")

        extract_from_dict(result)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"ToolExecutorAgent(name={self.name!r})"
