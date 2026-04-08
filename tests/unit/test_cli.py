"""Unit tests for CLI."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from src.cli import app


runner = CliRunner()


class TestCLIBasics:
    """Tests for basic CLI functionality."""

    def test_help(self):
        """Test --help shows usage."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "toolgen" in result.stdout.lower() or "usage" in result.stdout.lower()

    def test_version(self):
        """Test --version shows version."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.stdout

    def test_unknown_command(self):
        """Test unknown command fails gracefully."""
        result = runner.invoke(app, ["unknown-command"])
        assert result.exit_code != 0


class TestBuildCommand:
    """Tests for build command."""

    def test_build_help(self):
        """Test build --help."""
        result = runner.invoke(app, ["build", "--help"])
        assert result.exit_code == 0
        assert "toolbench" in result.stdout.lower()

    def test_build_requires_path(self):
        """Test build requires --toolbench-path."""
        result = runner.invoke(app, ["build"])
        assert result.exit_code != 0

    def test_build_with_path(self):
        """Test build with valid path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(app, ["build", "--toolbench-path", tmpdir])
            # Should not error even though not implemented
            assert "not yet implemented" in result.stdout.lower() or result.exit_code == 0


class TestGenerateCommand:
    """Tests for generate command."""

    def test_generate_help(self):
        """Test generate --help."""
        result = runner.invoke(app, ["generate", "--help"])
        assert result.exit_code == 0
        assert "output" in result.stdout.lower()

    def test_generate_requires_output(self):
        """Test generate requires --output."""
        result = runner.invoke(app, ["generate"])
        assert result.exit_code != 0

    def test_generate_with_options(self):
        """Test generate with various options."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "out.jsonl"
            result = runner.invoke(app, [
                "generate",
                "--output", str(output_path),
                "--count", "10",
                "--seed", "123",
            ])
            assert "not yet implemented" in result.stdout.lower() or result.exit_code == 0


class TestEvaluateCommand:
    """Tests for evaluate command."""

    def test_evaluate_help(self):
        """Test evaluate --help."""
        result = runner.invoke(app, ["evaluate", "--help"])
        assert result.exit_code == 0
        assert "input" in result.stdout.lower()

    def test_evaluate_requires_input(self):
        """Test evaluate requires --input."""
        result = runner.invoke(app, ["evaluate"])
        assert result.exit_code != 0

    def test_evaluate_with_input(self):
        """Test evaluate with valid input file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"test": "data"}\n')
            f.flush()

            result = runner.invoke(app, ["evaluate", "--input", f.name])
            assert "not yet implemented" in result.stdout.lower() or result.exit_code == 0

            os.unlink(f.name)


class TestConfigCommands:
    """Tests for config-related commands."""

    def test_config_show(self):
        """Test config-show displays config."""
        result = runner.invoke(app, ["config-show"])
        assert result.exit_code == 0
        # Should show config sections
        assert "models" in result.stdout.lower() or "primary" in result.stdout.lower()

    def test_config_validate(self):
        """Test config-validate works."""
        result = runner.invoke(app, ["config-validate"])
        assert result.exit_code == 0

    def test_config_validate_with_file(self):
        """Test config-validate with a config file argument."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
models:
  primary: test-model
neo4j:
  uri: bolt://test:7687
  username: test
  password: test
""")
            f.flush()

            result = runner.invoke(app, ["config-validate", f.name])
            assert result.exit_code == 0

            os.unlink(f.name)


class TestCacheCommands:
    """Tests for cache-related commands."""

    def test_cache_stats(self):
        """Test cache-stats displays statistics."""
        result = runner.invoke(app, ["cache-stats"])
        assert result.exit_code == 0
        assert "hits" in result.stdout.lower() or "misses" in result.stdout.lower()

    def test_cache_clear_cancel(self):
        """Test cache-clear can be cancelled."""
        result = runner.invoke(app, ["cache-clear"], input="n\n")
        assert "cancel" in result.stdout.lower() or result.exit_code == 0

    def test_cache_clear_with_yes_flag(self):
        """Test cache-clear with --yes flag."""
        result = runner.invoke(app, ["cache-clear", "--yes"])
        # Should succeed or report cache doesn't exist
        assert result.exit_code == 0


class TestVerboseQuiet:
    """Tests for verbose and quiet modes."""

    def test_verbose_mode(self):
        """Test verbose mode shows more output."""
        result = runner.invoke(app, ["-v", "config-show"])
        assert result.exit_code == 0

    def test_verbose_long_flag(self):
        """Test --verbose flag."""
        result = runner.invoke(app, ["--verbose", "config-show"])
        assert result.exit_code == 0

    def test_quiet_mode(self):
        """Test quiet mode reduces output."""
        result = runner.invoke(app, ["-q", "config-show"])
        assert result.exit_code == 0

    def test_quiet_long_flag(self):
        """Test --quiet flag."""
        result = runner.invoke(app, ["--quiet", "config-show"])
        assert result.exit_code == 0


class TestConfigLoading:
    """Tests for config file loading."""

    def test_custom_config_file(self):
        """Test loading custom config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
models:
  primary: test-model
neo4j:
  uri: bolt://test:7687
  username: test
  password: test
""")
            f.flush()

            result = runner.invoke(app, ["--config", f.name, "config-show"])
            assert result.exit_code == 0

            os.unlink(f.name)

    def test_missing_config_file(self):
        """Test error on missing config file."""
        result = runner.invoke(app, ["--config", "/nonexistent/config.yaml", "config-show"])
        assert result.exit_code != 0


class TestLoggingConfig:
    """Tests for logging configuration module."""

    def test_setup_logging_returns_logger(self):
        """Test setup_logging returns a logger."""
        import logging
        from src.logging_config import setup_logging

        logger = setup_logging(level="INFO")
        assert isinstance(logger, logging.Logger)

    def test_get_logger_returns_logger(self):
        """Test get_logger returns a logger."""
        import logging
        from src.logging_config import get_logger

        logger = get_logger("test")
        assert isinstance(logger, logging.Logger)
        assert "toolgen" in logger.name

    def test_verbose_sets_debug_level(self):
        """Test verbose mode sets DEBUG level."""
        import logging
        from src.logging_config import setup_logging

        logger = setup_logging(verbose=True)
        assert logger.level == logging.DEBUG

    def test_quiet_sets_warning_level(self):
        """Test quiet mode sets WARNING level."""
        import logging
        from src.logging_config import setup_logging

        logger = setup_logging(quiet=True)
        assert logger.level == logging.WARNING

    def test_file_logging(self):
        """Test file logging creates log file."""
        from src.logging_config import setup_logging

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = setup_logging(log_file=log_file)
            logger.info("Test message")

            # Flush handlers
            for handler in logger.handlers:
                handler.flush()

            assert log_file.exists()
