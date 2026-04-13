"""Implementation of the build command."""

import click
from pathlib import Path
from typing import Optional

from src.loaders import ToolBenchLoader
from src.registry import ToolRegistry


def run_build(
    toolbench_path: Path,
    use_llm_inference: bool,
    limit: Optional[int],
    output_path: Optional[Path],
    similarity_threshold: float,
    skip_embeddings: bool,
) -> None:
    """Execute the build command.

    Args:
        toolbench_path: Path to ToolBench data directory
        use_llm_inference: Whether to run LLM schema inference
        limit: Optional limit on number of tools to process
        output_path: Path to save graph file
        similarity_threshold: Threshold for similarity edges
        skip_embeddings: Whether to skip embedding generation
    """
    click.echo("=" * 50)
    click.echo("Tool Registry Build")
    click.echo("=" * 50)
    click.echo()

    # Statistics tracking
    stats = {
        "tools_loaded": 0,
        "tools_processed": 0,
        "endpoints_total": 0,
        "domains": set(),
        "graph_nodes": 0,
        "graph_edges": 0,
    }

    # Step 1: Load ToolBench data
    click.echo(f"[1/6] Loading ToolBench data from {toolbench_path}...")
    loader = ToolBenchLoader()
    registry = loader.load_directory(toolbench_path, limit=limit)
    tools = list(registry.tools.values())
    stats["tools_loaded"] = len(tools)

    click.secho(f"  ✓ Loaded {len(tools)} tools", fg="green")

    # Step 2: Optional LLM inference
    if use_llm_inference:
        click.echo("\n[2/6] Running LLM schema inference...")
        try:
            from src.services.llm_inference import LLMInferenceService

            inference_service = LLMInferenceService()
            processed_tools = []

            with click.progressbar(tools, label="  Inferring schemas") as bar:
                for tool in bar:
                    processed = inference_service.infer_tool_schemas(tool)
                    processed_tools.append(processed)

            tools = processed_tools
            click.secho("  ✓ Schema inference complete", fg="green")
        except ImportError as e:
            click.secho(f"  ⚠ LLM inference unavailable: {e}", fg="yellow")
    else:
        click.echo("\n[2/6] Skipping LLM schema inference (not enabled)")

    # Step 3: Summarize tool registry
    click.echo("\n[3/6] Analyzing tool registry...")

    for tool in tools:
        stats["endpoints_total"] += len(tool.endpoints)
        if tool.category:
            stats["domains"].add(tool.category)

    stats["tools_processed"] = len(tools)
    click.secho(f"  ✓ Registry has {stats['tools_processed']} tools with {stats['endpoints_total']} endpoints", fg="green")

    # Step 4-6: Build graph
    _build_graph(
        registry=registry,
        output_path=output_path,
        similarity_threshold=similarity_threshold,
        skip_embeddings=skip_embeddings,
        stats=stats,
    )

    # Print final statistics
    _print_statistics(stats)


def _build_graph(
    registry: ToolRegistry,
    output_path: Optional[Path],
    similarity_threshold: float,
    skip_embeddings: bool,
    stats: dict,
) -> None:
    """Build the NetworkX graph.

    Args:
        registry: Populated tool registry
        output_path: Path to save graph file
        similarity_threshold: Threshold for similarity edges
        skip_embeddings: Whether to skip embedding generation
        stats: Statistics dictionary to update
    """
    try:
        from src.graph import GraphClient, ToolGraphBuilder
        from src.config import GraphConfig
    except ImportError as e:
        click.secho(f"\n⚠ Graph module unavailable: {e}", fg="yellow")
        return

    # Step 4: Set up graph
    click.echo("\n[4/6] Setting up NetworkX graph...")
    try:
        config = GraphConfig()
        if output_path:
            config.path = output_path

        client = GraphClient(config)
        builder = ToolGraphBuilder(client, show_progress=True)
        click.secho("  ✓ Graph client initialized", fg="green")
    except Exception as e:
        click.secho(f"  ✗ Failed to initialize graph: {e}", fg="red")
        return

    # Step 5: Build nodes and basic edges
    click.echo("\n[5/6] Building graph from registry...")
    try:
        build_stats = builder.build_from_registry(registry, incremental=False, build_domains=True)

        stats["graph_nodes"] = build_stats.tools_added + build_stats.endpoints_added + build_stats.domains_added
        stats["graph_edges"] = build_stats.edges_added + build_stats.in_domain_edges_added

        click.secho(f"  ✓ Created {build_stats.tools_added} tool nodes", fg="green")
        click.secho(f"  ✓ Created {build_stats.endpoints_added} endpoint nodes", fg="green")
        click.secho(f"  ✓ Created {build_stats.domains_added} domain nodes", fg="green")
        click.secho(f"  ✓ Created {build_stats.edges_added} HAS_ENDPOINT edges", fg="green")
        click.secho(f"  ✓ Created {build_stats.in_domain_edges_added} IN_DOMAIN edges", fg="green")
    except Exception as e:
        click.secho(f"  ✗ Failed to build graph: {e}", fg="red")
        return

    # Create SAME_DOMAIN edges
    click.echo("\n  Creating SAME_DOMAIN edges...")
    try:
        same_domain_count = builder.create_same_domain_edges()
        stats["graph_edges"] += same_domain_count
        click.secho(f"  ✓ Created {same_domain_count} SAME_DOMAIN edges", fg="green")
    except Exception as e:
        click.secho(f"  ⚠ SAME_DOMAIN edges skipped: {e}", fg="yellow")

    # Step 6: Embeddings and semantic similarity
    if skip_embeddings:
        click.echo("\n[6/6] Skipping embeddings (--skip-embeddings)")
    else:
        click.echo(f"\n[6/6] Creating semantic similarity edges (threshold={similarity_threshold})...")
        try:
            from src.graph import EmbeddingGenerator, EmbeddingCache

            # Generate embeddings
            click.echo("  Generating embeddings...")
            generator = EmbeddingGenerator()
            cache = EmbeddingCache()

            endpoints = list(registry.endpoints.values())
            with click.progressbar(endpoints, label="  Embedding endpoints") as bar:
                for endpoint in bar:
                    embedding = generator.embed_endpoint(endpoint)
                    cache.set(endpoint.id, embedding)

            click.secho(f"  ✓ Generated {len(cache)} embeddings", fg="green")

            # Create semantic edges
            sim_stats = builder.create_semantic_edges(
                cache=cache,
                threshold=similarity_threshold,
                skip_same_domain=True,
            )
            stats["graph_edges"] += sim_stats.edges_created
            click.secho(f"  ✓ Created {sim_stats.edges_created} SEMANTICALLY_SIMILAR edges", fg="green")

            if sim_stats.edges_created > 0:
                click.echo(f"    Mean similarity: {sim_stats.mean_similarity:.3f}")
        except ImportError as e:
            click.secho(f"  ⚠ Embeddings unavailable: {e}", fg="yellow")
        except Exception as e:
            click.secho(f"  ⚠ Semantic edges skipped: {e}", fg="yellow")

    # Save graph
    click.echo("\n  Saving graph...")
    try:
        client.save_graph()
        click.secho(f"  ✓ Graph saved to {config.path}", fg="green")
    except Exception as e:
        click.secho(f"  ⚠ Failed to save graph: {e}", fg="yellow")


def _print_statistics(stats: dict) -> None:
    """Print final build statistics.

    Args:
        stats: Statistics dictionary
    """
    click.echo()
    click.echo("=" * 50)
    click.echo("Build Statistics")
    click.echo("=" * 50)
    click.echo(f"  Tools loaded:      {stats['tools_loaded']}")
    click.echo(f"  Tools processed:   {stats['tools_processed']}")
    click.echo(f"  Total endpoints:   {stats['endpoints_total']}")
    click.echo(f"  Unique domains:    {len(stats['domains'])}")

    if stats["graph_nodes"] > 0:
        click.echo(f"  Graph nodes:       {stats['graph_nodes']}")
        click.echo(f"  Graph edges:       {stats['graph_edges']}")

    click.echo("=" * 50)
    click.secho("✓ Build complete!", fg="green", bold=True)
