"""CLI entry point for the toolgen command."""

import click
import random
import signal
import sys
from collections import Counter
from pathlib import Path
from typing import Optional


@click.group()
@click.version_option(version="0.1.0", prog_name="toolgen")
def cli():
    """Multi-agent tool-use conversation generator.

    Use 'toolgen build' to process ToolBench data and build the tool registry.
    Use 'toolgen generate' to generate conversations.
    """
    pass


@cli.command()
@click.option(
    "--toolbench-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Path to ToolBench data directory containing tool JSON files.",
)
@click.option(
    "--use-llm-inference",
    is_flag=True,
    default=False,
    help="Enable LLM-based parameter schema inference.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Limit number of tools to process (for testing).",
)
@click.option(
    "--output-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to save the graph file (default: data/graph.pkl).",
)
@click.option(
    "--similarity-threshold",
    type=float,
    default=0.7,
    help="Threshold for semantic similarity edges (0.0-1.0).",
)
@click.option(
    "--skip-embeddings",
    is_flag=True,
    default=False,
    help="Skip embedding generation and semantic similarity edges.",
)
def build(
    toolbench_path: Path,
    use_llm_inference: bool,
    limit: Optional[int],
    output_path: Optional[Path],
    similarity_threshold: float,
    skip_embeddings: bool,
):
    """Build the tool registry from ToolBench data.

    This command:
    1. Loads tools from ToolBench JSON files
    2. Optionally runs LLM inference on parameter schemas
    3. Registers all tools in the registry
    4. Builds NetworkX graph with tool/endpoint/domain nodes
    5. Creates edges (HAS_ENDPOINT, IN_DOMAIN, SAME_DOMAIN)
    6. Generates embeddings for semantic similarity (optional)

    Example:
        toolgen build --toolbench-path ./data/toolbench --limit 10
    """
    from src.cli.commands.build import run_build

    try:
        run_build(
            toolbench_path=toolbench_path,
            use_llm_inference=use_llm_inference,
            limit=limit,
            output_path=output_path,
            similarity_threshold=similarity_threshold,
            skip_embeddings=skip_embeddings,
        )
    except Exception as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output path for JSONL file.",
)
@click.option(
    "--count", "-n",
    type=int,
    default=10,
    help="Number of conversations to generate.",
)
@click.option(
    "--seed", "-s",
    type=int,
    default=42,
    help="Random seed for reproducibility.",
)
@click.option(
    "--no-cross-conversation-steering",
    is_flag=True,
    default=False,
    help="Disable cross-conversation diversity steering.",
)
@click.option(
    "--min-steps",
    type=int,
    default=None,
    help="Minimum tool chain length (overrides config).",
)
@click.option(
    "--max-steps",
    type=int,
    default=None,
    help="Maximum tool chain length (overrides config).",
)
@click.option(
    "--min-score",
    type=float,
    default=None,
    help="Minimum judge score to accept (1-5, overrides config).",
)
@click.option(
    "--max-retries",
    type=int,
    default=None,
    help="Maximum retry attempts per conversation (overrides config).",
)
@click.option(
    "--parallel-workers", "-j",
    type=int,
    default=None,
    help="Number of parallel workers (overrides config).",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable LLM response caching.",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Enable verbose output.",
)
@click.option(
    "--quiet", "-q",
    is_flag=True,
    default=False,
    help="Suppress non-essential output.",
)
def generate(
    output: Path,
    count: int,
    seed: int,
    no_cross_conversation_steering: bool,
    min_steps: Optional[int],
    max_steps: Optional[int],
    min_score: Optional[float],
    max_retries: Optional[int],
    parallel_workers: Optional[int],
    no_cache: bool,
    verbose: bool,
    quiet: bool,
):
    """Generate synthetic tool-use conversations.

    Samples tool chains from the graph and generates multi-turn
    conversations with tool calls using the multi-agent system.

    Example:
        toolgen generate --output conversations.jsonl --count 100 --seed 42
    """
    from src.cli.commands.generate import run_generate

    try:
        run_generate(
            output=output,
            count=count,
            seed=seed,
            no_cross_conversation_steering=no_cross_conversation_steering,
            min_steps=min_steps,
            max_steps=max_steps,
            min_score=min_score,
            max_retries=max_retries,
            parallel_workers=parallel_workers,
            no_cache=no_cache,
            verbose=verbose,
            quiet=quiet,
        )
    except Exception as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()