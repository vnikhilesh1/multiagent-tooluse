"""Generate command implementation.

This module contains the main logic for the 'toolgen generate' command,
which generates synthetic tool-use conversations using the multi-agent system.
"""

import random
import signal
from collections import Counter
from pathlib import Path
from typing import Optional

import click


def run_generate(
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
    """Run the generate command.

    Args:
        output: Path for JSONL output file
        count: Number of conversations to generate
        seed: Random seed for reproducibility
        no_cross_conversation_steering: Disable diversity steering
        min_steps: Minimum tool chain length
        max_steps: Maximum tool chain length
        min_score: Minimum judge score threshold
        max_retries: Maximum retry attempts
        parallel_workers: Number of parallel workers
        no_cache: Disable LLM caching
        verbose: Enable verbose output
        quiet: Suppress non-essential output
    """
    from src.config import get_default_config, apply_cli_overrides, load_config
    from src.evaluation.serialization import write_dataset
    from src.graph.client import GraphClient
    from src.llm import create_client_from_config, create_cache_from_config
    from src.orchestrator import ConversationOrchestrator
    from src.sampling.constraints import SamplingConstraints
    from src.sampling.dfs_sampler import DFSSampler

    # Load config
    config_path = Path("config.yaml")
    if config_path.exists():
        config = load_config(config_path)
    else:
        config = get_default_config()

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

    if verbose:
        click.echo(f"Sampling: min_steps={config.sampling.min_steps}, max_steps={config.sampling.max_steps}")
        click.echo(f"Quality: min_score={config.quality.min_score}, max_retries={config.quality.max_retries}")
        click.echo(f"Cache: {'disabled' if no_cache else 'enabled'}")
        click.echo(f"Seed: {seed}")
        click.echo(f"Cross-conversation steering: {'disabled' if no_cross_conversation_steering else 'enabled'}")

    # Set random seed
    random.seed(seed)

    # Track results for graceful interrupt handling
    results = []
    interrupted = False

    def signal_handler(sig, frame):
        nonlocal interrupted
        interrupted = True
        if not quiet:
            click.echo("\nInterrupted! Saving progress...", err=True)

    # Register signal handler
    original_handler = signal.signal(signal.SIGINT, signal_handler)

    try:
        # Step 1: Load graph
        if not quiet:
            click.echo("Loading tool graph...")
            # Check graph file size for user information
            graph_path = config.graph.path
            if graph_path.exists():
                size_mb = graph_path.stat().st_size / (1024 * 1024)
                if size_mb > 100:
                    click.echo(f"  Graph file is {size_mb:.0f} MB - this may take a moment...")

        graph_client = GraphClient(config.graph)
        graph_client.load_graph()

        stats = graph_client.get_stats()
        if stats["total_nodes"] == 0:
            raise click.ClickException("Graph is empty. Run 'toolgen build' first to populate the graph.")

        if not quiet:
            endpoint_count = stats['nodes_by_type'].get('Endpoint', 0)
            tool_count = stats['nodes_by_type'].get('Tool', 0)
            click.echo(f"Loaded graph: {endpoint_count} endpoints, {tool_count} tools")

        # Step 2: Initialize LLM client
        if not quiet:
            click.echo("Initializing LLM client...")

        llm_client = create_client_from_config(config)

        # Set up cache if enabled
        if not no_cache:
            cache = create_cache_from_config(config)
            llm_client.cache = cache

        # Step 3: Initialize sampler
        constraints = SamplingConstraints(
            min_steps=config.sampling.min_steps,
            max_steps=config.sampling.max_steps,
            min_completeness=0.5,
        )

        sampler = DFSSampler(
            client=graph_client,
            constraints=constraints,
            max_start_candidates=config.sampling.max_start_candidates,
            max_neighbors=config.sampling.max_neighbors,
        )

        # Step 4: Initialize orchestrator
        if not quiet:
            click.echo("Initializing orchestrator with agents...")

        orchestrator = ConversationOrchestrator(
            llm=llm_client,
            graph=graph_client,
            sampler=sampler,
            quality_threshold=config.quality.min_score,
            max_retries=config.quality.max_retries,
            use_steering=not no_cross_conversation_steering,
        )

        # Register endpoints for diversity tracking
        endpoints = graph_client._endpoint_index
        orchestrator.register_endpoints(endpoints)

        # Step 5: Generate conversations
        if not quiet:
            click.echo(f"Generating {count} conversations...")

        with click.progressbar(
            range(count),
            label="Generating",
            show_pos=True,
            item_show_func=lambda x: f"Conv {x+1}" if x is not None else "",
        ) as bar:
            for i in bar:
                if interrupted:
                    break

                result = orchestrator.generate_single(
                    seed=seed + i if seed is not None else None,
                )
                results.append(result)

        # Step 6: Write output
        if not quiet:
            click.echo(f"Writing output to {output}...")

        output.parent.mkdir(parents=True, exist_ok=True)
        count_written = write_dataset(
            results,
            output,
            include_failed=False,
            include_metadata=True,
        )

        # Step 7: Print summary
        _print_summary(results, count_written, output, verbose, quiet)

    finally:
        signal.signal(signal.SIGINT, original_handler)


def _print_summary(results, count_written: int, output: Path, verbose: bool, quiet: bool):
    """Print generation summary statistics."""
    if quiet:
        click.echo(f"Generated {count_written} conversations to {output}")
        return

    total = len(results)
    successful = sum(1 for r in results if r.success)
    failed = total - successful
    repaired = sum(1 for r in results if r.repaired)

    # Compute statistics
    turn_counts = []
    tool_counts = []
    tool_usage = Counter()
    domain_usage = Counter()

    for r in results:
        if r.success and r.conversation:
            turn_counts.append(len(r.conversation.messages))
            tool_outputs = r.conversation.tool_outputs
            tool_counts.append(len(tool_outputs))

            for tool_output in tool_outputs:
                tool_id = getattr(tool_output, 'tool_id', None) or tool_output.endpoint_id.split('/')[0]
                tool_usage[tool_id] += 1
                domain = getattr(tool_output, 'domain', None)
                if domain:
                    domain_usage[domain] += 1

    avg_turns = sum(turn_counts) / len(turn_counts) if turn_counts else 0
    avg_tools = sum(tool_counts) / len(tool_counts) if tool_counts else 0

    scores = [r.scores.average for r in results if r.success and r.scores]
    avg_score = sum(scores) / len(scores) if scores else 0
    min_score_val = min(scores) if scores else 0
    max_score_val = max(scores) if scores else 0

    # Print summary
    click.echo("\n" + "=" * 50)
    click.echo("GENERATION SUMMARY")
    click.echo("=" * 50)
    click.echo(f"Total Requested:           {total}")
    click.echo(f"Successful:                {successful}")
    click.echo(f"Failed:                    {failed}")
    click.echo(f"Repaired:                  {repaired}")
    click.echo(f"Written to File:           {count_written}")
    click.echo("")
    click.echo(f"Avg Turns/Conversation:    {avg_turns:.1f}")
    click.echo(f"Avg Tool Calls/Conv:       {avg_tools:.1f}")
    click.echo("")
    click.echo(f"Avg Judge Score:           {avg_score:.2f}")
    click.echo(f"Min Judge Score:           {min_score_val:.2f}")
    click.echo(f"Max Judge Score:           {max_score_val:.2f}")
    click.echo("")
    click.echo(f"Unique Tools Used:         {len(tool_usage)}")
    click.echo(f"Unique Domains:            {len(domain_usage)}")
    click.echo("")
    click.echo(f"Output File:               {output}")
    click.echo("=" * 50)

    if verbose and tool_usage:
        click.echo("\nTop 5 Tools Used:")
        for tool_id, cnt in tool_usage.most_common(5):
            click.echo(f"  {tool_id}: {cnt}")
