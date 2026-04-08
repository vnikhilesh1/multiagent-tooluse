"""CLI for toolbench-conversation-generator.

This module provides the command-line interface using Typer with:
- Configuration file support
- Verbose/quiet modes
- Structured logging
- Rich console output
"""

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from src.config import Config, apply_cli_overrides, get_default_config, load_config, validate_config
from src.logging_config import get_logger, setup_logging

# Console for rich output
console = Console()
err_console = Console(stderr=True)

# Logger for this module
logger = get_logger(__name__)


# Global state
class AppState:
    """Application state shared across commands."""

    config: Optional[Config] = None
    verbose: bool = False
    quiet: bool = False


state = AppState()


def version_callback(value: bool):
    """Print version and exit."""
    if value:
        console.print("toolgen version 0.1.0")
        raise typer.Exit()


def config_callback(
    ctx: typer.Context,
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML configuration file",
        exists=True,
        readable=True,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose (debug) output",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress non-essential output",
    ),
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-V",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
):
    """Initialize application state before running commands."""
    # Set up logging first
    log_level = "DEBUG" if verbose else ("WARNING" if quiet else "INFO")
    setup_logging(level=log_level, verbose=verbose, quiet=quiet)

    state.verbose = verbose
    state.quiet = quiet

    # Load configuration
    if config_path is not None:
        try:
            state.config = load_config(config_path)
            logger.debug(f"Loaded config from {config_path}")
        except FileNotFoundError:
            err_console.print(f"[red]Error:[/red] Config file not found: {config_path}")
            raise typer.Exit(1)
        except ValueError as e:
            err_console.print(f"[red]Error:[/red] Configuration error: {e}")
            raise typer.Exit(1)
    elif Path("config.yaml").exists():
        try:
            state.config = load_config(Path("config.yaml"))
            logger.debug("Loaded config from config.yaml")
        except (FileNotFoundError, ValueError):
            state.config = get_default_config()
            logger.debug("Using default config")
    else:
        state.config = get_default_config()
        logger.debug("Using default config")


app = typer.Typer(
    name="toolgen",
    help="Multi-agent system for generating synthetic tool-use conversation datasets",
    add_completion=True,
    callback=config_callback,
    rich_markup_mode="rich",
)


@app.command()
def build(
    toolbench_path: Path = typer.Option(
        ...,
        "--toolbench-path",
        "-t",
        help="Path to ToolBench data directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    use_llm_inference: bool = typer.Option(
        False,
        "--use-llm-inference",
        help="Enable LLM-based schema inference for missing specs",
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        "-l",
        help="Limit number of tools to process (for testing)",
        min=1,
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force rebuild even if graph exists",
    ),
):
    """Build tool graph from ToolBench data.

    Ingests ToolBench API specifications, creates tool/endpoint nodes,
    and builds the tool graph in Neo4j with semantic similarity edges.

    Example:
        toolgen build --toolbench-path ./data/toolbench
    """
    logger.info(f"Building graph from: {toolbench_path}")

    if state.verbose:
        console.print(f"[dim]Config: Neo4j at {state.config.neo4j.uri}[/dim]")
        console.print(f"[dim]LLM inference: {'enabled' if use_llm_inference else 'disabled'}[/dim]")

    # TODO: Implement actual build logic
    console.print("[yellow]Build command not yet implemented[/yellow]")

    if limit:
        console.print(f"[dim]Would process up to {limit} tools[/dim]")


@app.command()
def generate(
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output path for JSONL file",
    ),
    count: Optional[int] = typer.Option(
        None,
        "--count",
        "-n",
        help="Number of conversations to generate",
        min=1,
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        "-s",
        help="Random seed for reproducibility",
    ),
    no_cross_conversation_steering: bool = typer.Option(
        False,
        "--no-cross-conversation-steering",
        help="Disable cross-conversation diversity steering",
    ),
    # Sampling overrides
    min_steps: Optional[int] = typer.Option(
        None,
        "--min-steps",
        help="Minimum tool chain length (overrides config)",
        min=1,
    ),
    max_steps: Optional[int] = typer.Option(
        None,
        "--max-steps",
        help="Maximum tool chain length (overrides config)",
        min=1,
    ),
    # Quality overrides
    min_score: Optional[float] = typer.Option(
        None,
        "--min-score",
        help="Minimum judge score to accept (1-5, overrides config)",
        min=1.0,
        max=5.0,
    ),
    max_retries: Optional[int] = typer.Option(
        None,
        "--max-retries",
        help="Maximum retry attempts per conversation (overrides config)",
        min=1,
    ),
    # Performance
    parallel_workers: Optional[int] = typer.Option(
        None,
        "--parallel-workers",
        "-j",
        help="Number of parallel workers (overrides config)",
        min=1,
    ),
    # Cache control
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Disable LLM response caching",
    ),
):
    """Generate synthetic tool-use conversations.

    Samples tool chains from the graph and generates multi-turn
    conversations with tool calls using the multi-agent system.

    Example:
        toolgen generate --output conversations.jsonl --count 100
    """
    config = state.config

    # Apply CLI overrides
    overrides = {}
    if min_steps is not None:
        overrides["sampling.min_steps"] = min_steps
    if max_steps is not None:
        overrides["sampling.max_steps"] = max_steps
    if min_score is not None:
        overrides["quality.min_score"] = min_score
    if max_retries is not None:
        overrides["quality.max_retries"] = max_retries
    if parallel_workers is not None:
        overrides["generation.parallel_workers"] = parallel_workers

    if overrides:
        config = apply_cli_overrides(config, **overrides)

    # Use config default if count not specified
    actual_count = count if count is not None else config.generation.default_count

    logger.info(f"Generating {actual_count} conversations to: {output}")

    if state.verbose:
        console.print(f"[dim]Sampling: min_steps={config.sampling.min_steps}, max_steps={config.sampling.max_steps}[/dim]")
        console.print(f"[dim]Quality: min_score={config.quality.min_score}, max_retries={config.quality.max_retries}[/dim]")
        console.print(f"[dim]Parallel workers: {config.generation.parallel_workers}[/dim]")
        console.print(f"[dim]Cache: {'disabled' if no_cache else 'enabled'}[/dim]")
        console.print(f"[dim]Seed: {seed}[/dim]")

    # TODO: Implement actual generation logic
    console.print("[yellow]Generate command not yet implemented[/yellow]")


@app.command()
def evaluate(
    input_path: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Input JSONL file to evaluate",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    output_report: Optional[Path] = typer.Option(
        None,
        "--output-report",
        "-o",
        help="Output path for evaluation report (JSON)",
    ),
    metrics: Optional[str] = typer.Option(
        None,
        "--metrics",
        "-m",
        help="Comma-separated list of metrics to compute",
    ),
):
    """Evaluate generated conversations and compute metrics.

    Analyzes conversation quality using multiple dimensions:
    tool correctness, argument grounding, task completion, naturalness.

    Example:
        toolgen evaluate --input conversations.jsonl --output-report report.json
    """
    logger.info(f"Evaluating: {input_path}")

    if state.verbose:
        console.print(f"[dim]Quality threshold: {state.config.quality.min_score}[/dim]")
        if metrics:
            console.print(f"[dim]Metrics: {metrics}[/dim]")

    # TODO: Implement actual evaluation logic
    console.print("[yellow]Evaluate command not yet implemented[/yellow]")


@app.command("config-show")
def config_show():
    """Display current configuration.

    Shows all configuration values from the loaded config file
    or defaults if no config file is specified.
    """
    config = state.config

    table = Table(title="Current Configuration", show_header=True)
    table.add_column("Section", style="cyan")
    table.add_column("Setting", style="green")
    table.add_column("Value", style="yellow")

    # Models
    table.add_row("models", "primary", config.models.primary)
    table.add_row("models", "embedding", config.models.embedding)
    table.add_row("models", "fallback", config.models.fallback)

    # Neo4j
    table.add_row("neo4j", "uri", config.neo4j.uri)
    table.add_row("neo4j", "username", config.neo4j.username)
    table.add_row("neo4j", "password", "***" if config.neo4j.password else "(not set)")

    # Sampling
    table.add_row("sampling", "min_steps", str(config.sampling.min_steps))
    table.add_row("sampling", "max_steps", str(config.sampling.max_steps))
    table.add_row("sampling", "semantic_threshold", str(config.sampling.semantic_threshold))
    table.add_row("sampling", "max_start_candidates", str(config.sampling.max_start_candidates))
    table.add_row("sampling", "max_neighbors", str(config.sampling.max_neighbors))

    # Quality
    table.add_row("quality", "min_score", str(config.quality.min_score))
    table.add_row("quality", "max_retries", str(config.quality.max_retries))
    table.add_row("quality", "require_multi_tool", str(config.quality.require_multi_tool))

    # Generation
    table.add_row("generation", "default_count", str(config.generation.default_count))
    table.add_row("generation", "parallel_workers", str(config.generation.parallel_workers))

    # Cache
    table.add_row("cache", "enabled", str(config.cache.enabled))
    table.add_row("cache", "directory", str(config.cache.directory))

    console.print(table)


@app.command("config-validate")
def config_validate(
    config_path: Optional[Path] = typer.Argument(
        None,
        help="Config file to validate (uses current if not specified)",
    ),
):
    """Validate a configuration file.

    Checks the configuration for errors and potential issues,
    reporting any warnings or problems found.
    """
    if config_path:
        try:
            config = load_config(config_path)
        except Exception as e:
            err_console.print(f"[red]Error loading config:[/red] {e}")
            raise typer.Exit(1)
    else:
        config = state.config

    warnings = validate_config(config)

    if warnings:
        console.print("[yellow]Configuration warnings:[/yellow]")
        for warning in warnings:
            console.print(f"  [yellow]\u2022[/yellow] {warning}")
    else:
        console.print("[green]Configuration is valid with no warnings[/green]")


@app.command("cache-clear")
def cache_clear(
    confirm: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt",
    ),
):
    """Clear the LLM response cache.

    Removes all cached LLM responses from both memory and disk.
    """
    from src.llm import create_cache_from_config

    cache_dir = state.config.cache.directory / "llm_responses"

    if not cache_dir.exists():
        console.print("[dim]Cache directory does not exist[/dim]")
        return

    if not confirm:
        confirm = typer.confirm(f"Clear cache at {cache_dir}?")

    if confirm:
        cache = create_cache_from_config(state.config)
        cache.clear()
        console.print("[green]Cache cleared[/green]")
    else:
        console.print("[dim]Cancelled[/dim]")


@app.command("cache-stats")
def cache_stats():
    """Show LLM cache statistics.

    Displays hit/miss counts and cache sizes.
    """
    from src.llm import create_cache_from_config

    cache = create_cache_from_config(state.config)
    stats = cache.get_stats()

    table = Table(title="Cache Statistics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="yellow", justify="right")

    table.add_row("Memory Hits", str(stats.memory_hits))
    table.add_row("Disk Hits", str(stats.disk_hits))
    table.add_row("Misses", str(stats.misses))
    table.add_row("Memory Size", str(stats.memory_size))
    table.add_row("Disk Entries", str(stats.disk_entries))

    total_hits = stats.memory_hits + stats.disk_hits
    total_requests = total_hits + stats.misses
    hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0

    table.add_row("", "")
    table.add_row("Total Requests", str(total_requests))
    table.add_row("Hit Rate", f"{hit_rate:.1f}%")

    console.print(table)


if __name__ == "__main__":
    app()
