"""Structural validator for generated conversations.

This module provides validation functions to ensure conversations
have the correct structure before being used for training.
"""

import json
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.context import ConversationContext, Message, ToolOutput


# Valid message roles
VALID_ROLES = {"user", "assistant", "tool", "system"}


def validate_structure(context: "ConversationContext") -> Tuple[bool, List[str]]:
    """Validate the structural integrity of a conversation.

    Checks that all required fields exist and have valid formats.
    Returns a tuple of (is_valid, error_messages).

    Args:
        context: ConversationContext to validate

    Returns:
        Tuple of (bool, List[str]):
            - bool: True if valid, False otherwise
            - List[str]: List of error messages (empty if valid)

    Example:
        >>> is_valid, errors = validate_structure(context)
        >>> if not is_valid:
        ...     for error in errors:
        ...         print(f"Error: {error}")
    """
    errors: List[str] = []

    # Check conversation_id
    errors.extend(_validate_conversation_id(context))

    # Check messages
    errors.extend(_validate_messages(context))

    # Check tool outputs
    errors.extend(_validate_tool_outputs(context))

    return (len(errors) == 0, errors)


def _validate_conversation_id(context: "ConversationContext") -> List[str]:
    """Validate the conversation_id field.

    Args:
        context: ConversationContext to validate

    Returns:
        List of error messages
    """
    errors: List[str] = []

    if not hasattr(context, 'conversation_id'):
        errors.append("Missing required field: conversation_id")
    elif context.conversation_id is None:
        errors.append("conversation_id is None")
    elif not isinstance(context.conversation_id, str):
        errors.append(f"conversation_id must be string, got {type(context.conversation_id).__name__}")
    elif len(context.conversation_id.strip()) == 0:
        errors.append("conversation_id is empty")

    return errors


def _validate_messages(context: "ConversationContext") -> List[str]:
    """Validate the messages list.

    Args:
        context: ConversationContext to validate

    Returns:
        List of error messages
    """
    errors: List[str] = []

    # Check messages list exists
    if not hasattr(context, 'messages'):
        errors.append("Missing required field: messages")
        return errors

    if context.messages is None:
        errors.append("messages is None")
        return errors

    if not isinstance(context.messages, list):
        errors.append(f"messages must be list, got {type(context.messages).__name__}")
        return errors

    if len(context.messages) == 0:
        errors.append("messages list is empty")
        return errors

    # Validate each message
    for i, message in enumerate(context.messages):
        errors.extend(_validate_message(message, i))

    return errors


def _validate_message(message: "Message", index: int) -> List[str]:
    """Validate a single message.

    Args:
        message: Message to validate
        index: Message index for error reporting

    Returns:
        List of error messages
    """
    errors: List[str] = []
    prefix = f"messages[{index}]"

    # Check role
    if not hasattr(message, 'role'):
        errors.append(f"{prefix}: Missing required field 'role'")
    elif message.role is None:
        errors.append(f"{prefix}: role is None")
    elif message.role not in VALID_ROLES:
        errors.append(f"{prefix}: Invalid role '{message.role}', must be one of {VALID_ROLES}")

    # Check content/tool_calls
    has_content = hasattr(message, 'content') and message.content is not None and len(str(message.content).strip()) > 0
    has_tool_calls = hasattr(message, 'tool_calls') and message.tool_calls is not None and len(message.tool_calls) > 0

    # User messages must have content
    if hasattr(message, 'role') and message.role == "user":
        if not has_content:
            errors.append(f"{prefix}: User message must have content")

    # Assistant messages must have content or tool_calls
    if hasattr(message, 'role') and message.role == "assistant":
        if not has_content and not has_tool_calls:
            errors.append(f"{prefix}: Assistant message must have content or tool_calls")

    # Tool messages must have content and tool_call_id
    if hasattr(message, 'role') and message.role == "tool":
        if not has_content:
            errors.append(f"{prefix}: Tool message must have content")
        if not hasattr(message, 'tool_call_id') or message.tool_call_id is None:
            errors.append(f"{prefix}: Tool message must have tool_call_id")

    # Validate tool_calls format if present
    if has_tool_calls:
        errors.extend(_validate_tool_calls(message.tool_calls, prefix))

    return errors


def _validate_tool_calls(tool_calls: List[Dict[str, Any]], prefix: str) -> List[str]:
    """Validate tool_calls format.

    Args:
        tool_calls: List of tool call dicts
        prefix: Prefix for error messages

    Returns:
        List of error messages
    """
    errors: List[str] = []

    if not isinstance(tool_calls, list):
        errors.append(f"{prefix}.tool_calls: Must be a list")
        return errors

    for j, tool_call in enumerate(tool_calls):
        tc_prefix = f"{prefix}.tool_calls[{j}]"

        if not isinstance(tool_call, dict):
            errors.append(f"{tc_prefix}: Must be a dict")
            continue

        # Check for function field (Anthropic format)
        if "function" in tool_call:
            func = tool_call["function"]
            if not isinstance(func, dict):
                errors.append(f"{tc_prefix}.function: Must be a dict")
            else:
                if "name" not in func:
                    errors.append(f"{tc_prefix}.function: Missing 'name' field")
                elif not isinstance(func["name"], str) or len(func["name"].strip()) == 0:
                    errors.append(f"{tc_prefix}.function.name: Must be non-empty string")

                if "arguments" not in func:
                    errors.append(f"{tc_prefix}.function: Missing 'arguments' field")
                else:
                    # Arguments can be string (JSON) or dict
                    args = func["arguments"]
                    if isinstance(args, str):
                        try:
                            json.loads(args)
                        except json.JSONDecodeError:
                            errors.append(f"{tc_prefix}.function.arguments: Invalid JSON string")
                    elif not isinstance(args, dict):
                        errors.append(f"{tc_prefix}.function.arguments: Must be dict or JSON string")

        # Alternative format: direct name/arguments
        elif "name" in tool_call:
            if not isinstance(tool_call["name"], str) or len(tool_call["name"].strip()) == 0:
                errors.append(f"{tc_prefix}.name: Must be non-empty string")

            if "arguments" not in tool_call:
                errors.append(f"{tc_prefix}: Missing 'arguments' field")
            else:
                args = tool_call["arguments"]
                if isinstance(args, str):
                    try:
                        json.loads(args)
                    except json.JSONDecodeError:
                        errors.append(f"{tc_prefix}.arguments: Invalid JSON string")
                elif not isinstance(args, dict):
                    errors.append(f"{tc_prefix}.arguments: Must be dict or JSON string")

        else:
            errors.append(f"{tc_prefix}: Must have 'function' or 'name' field")

    return errors


def _validate_tool_outputs(context: "ConversationContext") -> List[str]:
    """Validate tool outputs.

    Args:
        context: ConversationContext to validate

    Returns:
        List of error messages
    """
    errors: List[str] = []

    # tool_outputs is optional but if present must be valid
    if not hasattr(context, 'tool_outputs'):
        return errors

    if context.tool_outputs is None:
        return errors

    if not isinstance(context.tool_outputs, list):
        errors.append(f"tool_outputs must be list, got {type(context.tool_outputs).__name__}")
        return errors

    for i, output in enumerate(context.tool_outputs):
        errors.extend(_validate_tool_output(output, i))

    return errors


def _validate_tool_output(output: "ToolOutput", index: int) -> List[str]:
    """Validate a single tool output.

    Args:
        output: ToolOutput to validate
        index: Output index for error reporting

    Returns:
        List of error messages
    """
    errors: List[str] = []
    prefix = f"tool_outputs[{index}]"

    # Check endpoint_id
    if not hasattr(output, 'endpoint_id'):
        errors.append(f"{prefix}: Missing required field 'endpoint_id'")
    elif output.endpoint_id is None or not isinstance(output.endpoint_id, str):
        errors.append(f"{prefix}.endpoint_id: Must be non-empty string")
    elif len(output.endpoint_id.strip()) == 0:
        errors.append(f"{prefix}.endpoint_id: Must be non-empty string")

    # Check arguments
    if not hasattr(output, 'arguments'):
        errors.append(f"{prefix}: Missing required field 'arguments'")
    elif output.arguments is not None and not isinstance(output.arguments, dict):
        errors.append(f"{prefix}.arguments: Must be dict")

    # Check result
    if not hasattr(output, 'result'):
        errors.append(f"{prefix}: Missing required field 'result'")
    elif output.result is not None and not isinstance(output.result, dict):
        errors.append(f"{prefix}.result: Must be dict")

    # Check call_id
    if not hasattr(output, 'call_id'):
        errors.append(f"{prefix}: Missing required field 'call_id'")
    elif output.call_id is None or not isinstance(output.call_id, str):
        errors.append(f"{prefix}.call_id: Must be non-empty string")
    elif len(output.call_id.strip()) == 0:
        errors.append(f"{prefix}.call_id: Must be non-empty string")

    return errors


def validate_conversation_dict(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a conversation in dictionary format.

    Useful for validating serialized conversations (e.g., from JSONL).

    Args:
        data: Dictionary representation of conversation

    Returns:
        Tuple of (bool, List[str]):
            - bool: True if valid, False otherwise
            - List[str]: List of error messages

    Example:
        >>> data = {"conversation_id": "123", "messages": [...]}
        >>> is_valid, errors = validate_conversation_dict(data)
    """
    errors: List[str] = []

    # Check conversation_id
    if "conversation_id" not in data:
        errors.append("Missing required field: conversation_id")
    elif not isinstance(data["conversation_id"], str) or len(data["conversation_id"].strip()) == 0:
        errors.append("conversation_id must be non-empty string")

    # Check messages
    if "messages" not in data:
        errors.append("Missing required field: messages")
    elif not isinstance(data["messages"], list):
        errors.append("messages must be list")
    elif len(data["messages"]) == 0:
        errors.append("messages list is empty")
    else:
        for i, msg in enumerate(data["messages"]):
            if not isinstance(msg, dict):
                errors.append(f"messages[{i}]: Must be dict")
                continue

            # Check role
            if "role" not in msg:
                errors.append(f"messages[{i}]: Missing 'role' field")
            elif msg["role"] not in VALID_ROLES:
                errors.append(f"messages[{i}]: Invalid role '{msg['role']}'")

            # Check content/tool_calls based on role
            has_content = "content" in msg and msg["content"] is not None and len(str(msg["content"]).strip()) > 0
            has_tool_calls = "tool_calls" in msg and msg["tool_calls"] is not None and len(msg["tool_calls"]) > 0

            role = msg.get("role")
            if role == "user" and not has_content:
                errors.append(f"messages[{i}]: User message must have content")
            if role == "assistant" and not has_content and not has_tool_calls:
                errors.append(f"messages[{i}]: Assistant message must have content or tool_calls")
            if role == "tool":
                if not has_content:
                    errors.append(f"messages[{i}]: Tool message must have content")
                if "tool_call_id" not in msg or msg["tool_call_id"] is None:
                    errors.append(f"messages[{i}]: Tool message must have tool_call_id")

            # Validate tool_calls if present
            if has_tool_calls:
                for j, tc in enumerate(msg["tool_calls"]):
                    tc_prefix = f"messages[{i}].tool_calls[{j}]"
                    if not isinstance(tc, dict):
                        errors.append(f"{tc_prefix}: Must be dict")
                    elif "function" not in tc and "name" not in tc:
                        errors.append(f"{tc_prefix}: Must have 'function' or 'name' field")

    return (len(errors) == 0, errors)
