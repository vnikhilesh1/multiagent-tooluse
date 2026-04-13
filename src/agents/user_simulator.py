"""User Simulator Agent for generating realistic user messages.

This agent simulates a user in conversation, generating natural messages
that progress toward the user's goal while responding appropriately to
the assistant.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.agents.base import BaseAgent
from src.models.context import Message

if TYPE_CHECKING:
    from src.llm import LLMClient
    from src.models.context import ConversationContext


class UserSimulatorAgent(BaseAgent):
    """Agent that simulates user behavior in conversations.

    Generates natural user messages that:
    - Start with an initial request based on the user's goal
    - Respond to assistant questions and prompts
    - Provide clarification at disambiguation points
    - Progress the conversation toward completing the goal

    Attributes:
        llm: LLMClient for generating messages
        name: Agent identifier

    Example:
        >>> agent = UserSimulatorAgent(llm=client, name="user_sim")
        >>> context = ConversationContext(scenario_description="Weather check")
        >>> context.grounding_values["scenario"] = {"user_goal": "Check weather in Seattle"}
        >>> context = agent.generate(context)
        >>> print(context.messages[-1].content)
        "What's the weather like in Seattle today?"
    """

    def __init__(
        self,
        llm: "LLMClient",
        name: str = "user_simulator",
    ) -> None:
        """Initialize the user simulator agent.

        Args:
            llm: LLMClient instance for LLM calls
            name: Agent identifier
        """
        super().__init__(llm=llm, name=name)

    def generate(self, context: "ConversationContext") -> "ConversationContext":
        """Generate the next user message based on conversation context.

        Determines whether to generate an initial message, follow-up,
        or clarification based on the current conversation state.

        Args:
            context: Current conversation context

        Returns:
            Updated context with new user message added
        """
        # Determine message type and build appropriate prompt
        if self._is_initial_message(context):
            prompt = self._build_initial_prompt(context)
        elif self._is_at_disambiguation_point(context):
            prompt = self._build_clarification_prompt(context)
        else:
            prompt = self._build_followup_prompt(context)

        # Generate message using LLM
        response = self.llm.complete(
            prompt=prompt,
            temperature=0.8,  # Higher temperature for natural variation
            max_tokens=256,
        )

        # Clean up response (remove quotes, extra whitespace)
        content = self._clean_response(response)

        # Add user message to context
        message = Message(role="user", content=content)
        context.add_message(message)

        return context

    def _is_initial_message(self, context: "ConversationContext") -> bool:
        """Check if this is the first user message in the conversation.

        Args:
            context: Current conversation context

        Returns:
            True if no user messages exist yet
        """
        user_messages = [m for m in context.messages if m.role == "user"]
        return len(user_messages) == 0

    def _is_at_disambiguation_point(self, context: "ConversationContext") -> bool:
        """Check if the conversation is at a disambiguation point.

        Disambiguation points are turns where the user should provide
        clarification or additional details.

        Args:
            context: Current conversation context

        Returns:
            True if current turn is a disambiguation point
        """
        scenario = context.grounding_values.get("scenario", {})
        disambiguation_points = scenario.get("disambiguation_points", [])

        if not disambiguation_points:
            return False

        # Count user messages to determine current turn
        user_messages = [m for m in context.messages if m.role == "user"]
        current_turn = len(user_messages)

        return current_turn in disambiguation_points

    def _get_user_goal(self, context: "ConversationContext") -> str:
        """Extract the user goal from context.

        Args:
            context: Current conversation context

        Returns:
            User goal string, or scenario description as fallback
        """
        scenario = context.grounding_values.get("scenario", {})
        return scenario.get("user_goal", context.scenario_description or "Complete the task")

    def _build_initial_prompt(self, context: "ConversationContext") -> str:
        """Build prompt for generating the initial user message.

        The initial message should be natural and may be somewhat vague,
        like a real user starting a conversation.

        Args:
            context: Current conversation context

        Returns:
            Prompt string for LLM
        """
        user_goal = self._get_user_goal(context)

        prompt = f"""You are simulating a user who wants to accomplish this goal:
{user_goal}

Generate the FIRST message this user would send to an AI assistant.

Guidelines:
- Be natural and conversational, like a real person typing
- You can be slightly vague or incomplete (real users often are)
- Don't include all details upfront - let the conversation develop
- Use casual language, not overly formal
- Keep it brief - 1-2 sentences typically
- Don't use quotation marks around your response

Example styles:
- "Hey, I need help booking a flight to Paris"
- "What's the weather gonna be like this weekend?"
- "Can you help me find a good restaurant nearby?"

Generate the user's opening message:"""

        return prompt

    def _build_followup_prompt(self, context: "ConversationContext") -> str:
        """Build prompt for generating a follow-up user message.

        Follow-up messages respond to the assistant and continue
        progressing toward the goal.

        Args:
            context: Current conversation context

        Returns:
            Prompt string for LLM
        """
        user_goal = self._get_user_goal(context)
        history = context.get_history_for_prompt()

        # Get the last assistant message
        last_assistant_msg = ""
        for msg in reversed(context.messages):
            if msg.role == "assistant" and msg.content:
                last_assistant_msg = msg.content
                break

        prompt = f"""You are simulating a user in a conversation with an AI assistant.

User's goal: {user_goal}

Conversation so far:
{history}

The assistant just said: "{last_assistant_msg}"

Generate the user's next response.

Guidelines:
- Respond naturally to what the assistant said
- Continue working toward your goal
- Provide information the assistant needs
- Stay in character as a regular user
- Keep it conversational and brief
- Don't use quotation marks around your response

Generate the user's response:"""

        return prompt

    def _build_clarification_prompt(self, context: "ConversationContext") -> str:
        """Build prompt for generating a clarification message.

        Clarification messages provide specific details that help
        disambiguate the user's request.

        Args:
            context: Current conversation context

        Returns:
            Prompt string for LLM
        """
        user_goal = self._get_user_goal(context)
        history = context.get_history_for_prompt()

        # Get the last assistant message
        last_assistant_msg = ""
        for msg in reversed(context.messages):
            if msg.role == "assistant" and msg.content:
                last_assistant_msg = msg.content
                break

        prompt = f"""You are simulating a user in a conversation with an AI assistant.

User's goal: {user_goal}

Conversation so far:
{history}

The assistant just asked: "{last_assistant_msg}"

This is a CLARIFICATION point - the user needs to provide specific details.

Generate a response that:
- Directly answers the assistant's question
- Provides specific, concrete information
- Helps move the conversation forward
- Stays natural and conversational
- Don't use quotation marks around your response

Generate the user's clarifying response:"""

        return prompt

    def _clean_response(self, response: str) -> str:
        """Clean up the LLM response for use as a user message.

        Removes quotes, extra whitespace, and other artifacts.

        Args:
            response: Raw LLM response

        Returns:
            Cleaned message content
        """
        # Strip whitespace
        content = response.strip()

        # Remove surrounding quotes if present
        if content.startswith('"') and content.endswith('"'):
            content = content[1:-1]
        if content.startswith("'") and content.endswith("'"):
            content = content[1:-1]

        # Remove common prefixes the LLM might add
        prefixes_to_remove = [
            "User: ",
            "User message: ",
            "Response: ",
            "Message: ",
        ]
        for prefix in prefixes_to_remove:
            if content.startswith(prefix):
                content = content[len(prefix):]

        return content.strip()

    def __repr__(self) -> str:
        """Return string representation."""
        return f"UserSimulatorAgent(name={self.name!r})"
