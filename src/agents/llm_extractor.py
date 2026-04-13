"""LLM-based value extractor for grounding conversation data.

This module provides extraction of referenceable values (IDs, names, status)
from tool responses to ensure consistency across conversation turns.
"""

import json
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from src.llm import LLMClient
    from src.models.context import ConversationContext


class LLMExtractor:
    """Extracts referenceable values from tool responses using LLM.

    Uses the LLM to intelligently identify important values that should
    be tracked for consistency in later conversation turns. Falls back
    to regex-based extraction for *_id fields if the LLM fails.

    Attributes:
        llm: LLMClient for intelligent extraction

    Example:
        >>> extractor = LLMExtractor(llm=client)
        >>> data = {"user": {"id": "U123", "name": "John"}, "status": "active"}
        >>> values = extractor.extract(data)
        >>> print(values)
        {"user_id": "U123", "user_name": "John", "status": "active"}
    """

    def __init__(self, llm: "LLMClient") -> None:
        """Initialize the extractor.

        Args:
            llm: LLMClient instance for LLM-based extraction
        """
        self.llm = llm

    def extract(
        self,
        data: Dict[str, Any],
        context: Optional["ConversationContext"] = None,
    ) -> Dict[str, Any]:
        """Extract referenceable values from data.

        Uses the LLM to identify important values like IDs, names,
        and status fields. Falls back to regex extraction if LLM fails.

        Args:
            data: Tool response data to extract from
            context: Optional context to update with extracted values

        Returns:
            Dict of extracted key-value pairs
        """
        if not data:
            return {}

        try:
            # Build extraction prompt
            prompt = self._build_extraction_prompt(data)

            # Get LLM to extract values
            extracted = self.llm.complete_json(
                prompt=prompt,
                temperature=0.0,
                max_tokens=512,
            )

            if not isinstance(extracted, dict):
                extracted = {}

        except Exception:
            # Fall back to regex-based extraction
            extracted = self._fallback_extraction(data)

        # Update context if provided
        if context is not None:
            self.update_context(context, extracted)

        return extracted

    def _build_extraction_prompt(self, data: Dict[str, Any]) -> str:
        """Build the prompt for LLM-based extraction.

        Asks the LLM to identify important referenceable values
        from the data structure.

        Args:
            data: Data to extract from

        Returns:
            Prompt string for LLM
        """
        data_str = json.dumps(data, indent=2)

        prompt = f"""Extract important referenceable values from this API response data.

Data:
{data_str}

Identify and extract:
1. **IDs** - Any field ending in "_id" or named "id" (e.g., user_id, order_id, booking_id)
2. **Names** - Person names, product names, location names
3. **Status values** - Any status, state, or condition fields
4. **Key amounts** - Totals, prices, balances, quantities
5. **Contact info** - Email addresses, phone numbers
6. **Reference numbers** - Confirmation numbers, tracking numbers, reference codes

Return a flat JSON object with descriptive keys and their values.
Use snake_case for keys (e.g., "user_id", "order_total", "customer_name").

For nested data, flatten the keys (e.g., "order_id" not "order.id").

Return ONLY the JSON object with extracted values, no explanation:"""

        return prompt

    def _fallback_extraction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback extraction using pattern matching.

        Extracts values using regex patterns when LLM is unavailable.
        Focuses on *_id fields and common important patterns.

        Args:
            data: Data to extract from

        Returns:
            Dict of extracted key-value pairs
        """
        extracted = {}

        def extract_recursive(obj: Any, prefix: str = "") -> None:
            """Recursively extract values from nested structures."""
            if isinstance(obj, dict):
                for key, value in obj.items():
                    full_key = f"{prefix}_{key}" if prefix else key

                    # Extract *_id fields
                    if key.lower().endswith("_id") or key.lower() == "id":
                        if value is not None and not isinstance(value, (dict, list)):
                            # Use descriptive key
                            if key.lower() == "id" and prefix:
                                extracted[f"{prefix}_id"] = value
                            else:
                                extracted[key] = value

                    # Extract status fields
                    elif "status" in key.lower() or "state" in key.lower():
                        if value is not None and not isinstance(value, (dict, list)):
                            extracted[key] = value

                    # Extract name fields
                    elif key.lower() in ("name", "full_name", "first_name", "last_name"):
                        if value is not None and not isinstance(value, (dict, list)):
                            result_key = f"{prefix}_{key}" if prefix else key
                            extracted[result_key] = value

                    # Extract amount/total fields
                    elif key.lower() in ("total", "amount", "price", "balance", "quantity"):
                        if value is not None and isinstance(value, (int, float)):
                            result_key = f"{prefix}_{key}" if prefix else key
                            extracted[result_key] = value

                    # Extract confirmation/reference numbers
                    elif "confirmation" in key.lower() or "reference" in key.lower():
                        if value is not None and not isinstance(value, (dict, list)):
                            extracted[key] = value

                    # Extract email
                    elif key.lower() == "email":
                        if value is not None and isinstance(value, str):
                            extracted[key] = value

                    # Recurse into nested dicts
                    if isinstance(value, dict):
                        extract_recursive(value, key)

                    # Handle lists (extract from first item if dict)
                    elif isinstance(value, list) and value and isinstance(value[0], dict):
                        extract_recursive(value[0], key)

            elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
                extract_recursive(obj[0], prefix)

        extract_recursive(data)
        return extracted

    def update_context(
        self,
        context: "ConversationContext",
        values: Dict[str, Any],
    ) -> None:
        """Update context's grounding_values with extracted values.

        Args:
            context: ConversationContext to update
            values: Extracted values to add
        """
        for key, value in values.items():
            context.grounding_values[key] = value

    def __repr__(self) -> str:
        """Return string representation."""
        return "LLMExtractor()"
