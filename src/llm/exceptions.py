"""Custom exceptions for LLM client.

This module defines the exception hierarchy for LLM-related errors,
providing specific exception types for different failure modes.
"""

from typing import Optional


class LLMError(Exception):
    """Base exception for LLM client errors.

    All LLM-related exceptions inherit from this class, allowing
    callers to catch all LLM errors with a single except clause.
    """
    pass


class LLMRateLimitError(LLMError):
    """Raised when rate limited by the API.

    Attributes:
        retry_after: Optional number of seconds to wait before retrying.
    """

    def __init__(self, message: str, retry_after: Optional[float] = None):
        """Initialize rate limit error.

        Args:
            message: Error message describing the rate limit.
            retry_after: Optional seconds to wait before retrying.
        """
        super().__init__(message)
        self.retry_after = retry_after


class LLMAPIError(LLMError):
    """Raised when API returns an error.

    Attributes:
        status_code: HTTP status code from the API response.
    """

    def __init__(self, message: str, status_code: Optional[int] = None):
        """Initialize API error.

        Args:
            message: Error message from the API.
            status_code: HTTP status code of the error response.
        """
        super().__init__(message)
        self.status_code = status_code


class LLMValidationError(LLMError):
    """Raised when response validation fails.

    This includes JSON parsing errors, Pydantic validation failures,
    and other response format issues.
    """
    pass


class LLMConnectionError(LLMError):
    """Raised when connection to the API fails.

    This includes network errors, timeouts, and DNS resolution failures.
    """
    pass
