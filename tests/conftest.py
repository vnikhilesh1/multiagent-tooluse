"""Pytest configuration and fixtures for toolbench-conversation-generator tests."""

import pytest


@pytest.fixture
def sample_config():
    """Provide a sample configuration for tests."""
    return {
        "models": {
            "primary": "claude-sonnet-4-20250514",
            "embedding": "text-embedding-3-small",
        },
        "sampling": {
            "min_steps": 2,
            "max_steps": 5,
        },
        "quality": {
            "min_score": 3.5,
            "max_retries": 3,
        },
    }
