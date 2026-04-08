"""Logging configuration for the CLI.

Provides configurable logging with:
- Console output with rich formatting
- Optional file logging
- Configurable log levels
- Colored output for different log levels
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Try to use rich for pretty console output
try:
    from rich.console import Console
    from rich.logging import RichHandler

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Application logger name
APP_LOGGER_NAME = "toolgen"


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    verbose: bool = False,
    quiet: bool = False,
) -> logging.Logger:
    """Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to log file.
        verbose: If True, set level to DEBUG.
        quiet: If True, set level to WARNING.

    Returns:
        Configured root logger for the application.
    """
    # Determine effective level
    if verbose:
        effective_level = logging.DEBUG
    elif quiet:
        effective_level = logging.WARNING
    else:
        effective_level = getattr(logging, level.upper(), logging.INFO)

    # Get root logger for our app
    logger = logging.getLogger(APP_LOGGER_NAME)
    logger.setLevel(effective_level)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    if RICH_AVAILABLE and not quiet:
        console_handler = RichHandler(
            console=Console(stderr=True),
            show_time=verbose,
            show_path=verbose,
            rich_tracebacks=True,
        )
    else:
        console_handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            "%(levelname)s - %(name)s - %(message)s"
            if not verbose
            else "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
        console_handler.setFormatter(formatter)

    console_handler.setLevel(effective_level)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Logger instance.
    """
    return logging.getLogger(f"{APP_LOGGER_NAME}.{name}")
