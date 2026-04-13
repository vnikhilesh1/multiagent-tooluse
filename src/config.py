"""Configuration module for toolbench-conversation-generator.

This module provides typed configuration models using Pydantic, with support for:
- YAML configuration files
- Environment variable expansion (${VAR:-default} syntax)
- CLI argument overrides
- Validation with sensible defaults
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator


class ModelsConfig(BaseModel):
    """Configuration for LLM models."""

    primary: str = Field(
        default="claude-sonnet-4-20250514",
        description="Primary LLM for generation"
    )
    embedding: str = Field(
        default="text-embedding-3-small",
        description="Embedding model for semantic similarity"
    )
    fallback: str = Field(
        default="gpt-4o-mini",
        description="Fallback model when primary fails"
    )


class GraphConfig(BaseModel):
    """Configuration for NetworkX graph storage."""

    path: Path = Field(
        default=Path(".cache/graph.pkl"),
        description="Path to graph file"
    )
    format: Literal["pickle", "graphml", "json"] = Field(
        default="pickle",
        description="Graph serialization format"
    )
    embeddings_path: Path = Field(
        default=Path(".cache/embeddings.npz"),
        description="Path to embeddings cache"
    )
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Threshold for semantic similarity edges"
    )

    @field_validator('path', 'embeddings_path', mode='before')
    @classmethod
    def convert_to_path(cls, v: Any) -> Path:
        """Convert string to Path."""
        if isinstance(v, str):
            return Path(v)
        return v


class SamplingConfig(BaseModel):
    """Configuration for tool chain sampling."""

    min_steps: int = Field(
        default=2,
        ge=1,
        description="Minimum tool chain length"
    )
    max_steps: int = Field(
        default=5,
        description="Maximum tool chain length"
    )
    semantic_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Threshold for semantic similarity edges"
    )
    max_start_candidates: int = Field(
        default=10,
        ge=1,
        description="Maximum starting points for DFS"
    )
    max_neighbors: int = Field(
        default=10,
        ge=1,
        description="Maximum neighbors to consider in DFS"
    )

    @model_validator(mode='after')
    def validate_step_range(self) -> 'SamplingConfig':
        """Ensure max_steps >= min_steps."""
        if self.max_steps < self.min_steps:
            raise ValueError(
                f"max_steps ({self.max_steps}) must be >= min_steps ({self.min_steps})"
            )
        return self


class QualityConfig(BaseModel):
    """Configuration for quality thresholds."""

    min_score: float = Field(
        default=3.5,
        ge=1.0,
        le=5.0,
        description="Minimum judge score to accept (1-5 scale)"
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        description="Maximum retry attempts per conversation"
    )
    require_multi_tool: bool = Field(
        default=True,
        description="Require >= 2 distinct tools in conversation"
    )


class GenerationConfig(BaseModel):
    """Configuration for conversation generation."""

    default_count: int = Field(
        default=100,
        ge=1,
        description="Default number of conversations to generate"
    )
    parallel_workers: int = Field(
        default=4,
        ge=1,
        description="Number of parallel workers"
    )


class CacheConfig(BaseModel):
    """Configuration for caching."""

    enabled: bool = Field(
        default=True,
        description="Enable/disable caching"
    )
    directory: Path = Field(
        default=Path(".cache"),
        description="Cache directory path"
    )

    @field_validator('directory', mode='before')
    @classmethod
    def convert_to_path(cls, v: Any) -> Path:
        """Convert string to Path."""
        if isinstance(v, str):
            return Path(v)
        return v


class Config(BaseModel):
    """Main configuration combining all config sections."""

    models: ModelsConfig = Field(default_factory=ModelsConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)

    def copy_with_overrides(self, **overrides: Any) -> 'Config':
        """Create a copy of this config with specified overrides.

        Args:
            **overrides: Key-value pairs using dot notation (e.g., sampling.min_steps=3)

        Returns:
            New Config instance with overrides applied
        """
        return apply_cli_overrides(self, **overrides)


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand environment variables in a value.

    Supports patterns:
    - ${VAR_NAME} - Required variable, raises error if not set
    - ${VAR_NAME:-default} - Optional variable with default value

    Args:
        value: Value to expand (string, dict, list, or other)

    Returns:
        Value with environment variables expanded

    Raises:
        ValueError: If required environment variable is not set
    """
    if isinstance(value, str):
        # Pattern: ${VAR_NAME:-default} or ${VAR_NAME}
        pattern = r'\$\{([^}:]+)(?::-([^}]*))?\}'

        def replace_var(match: re.Match) -> str:
            var_name = match.group(1)
            default_value = match.group(2)

            env_value = os.environ.get(var_name)

            if env_value is not None:
                return env_value
            elif default_value is not None:
                return default_value
            else:
                raise ValueError(
                    f"Required environment variable '{var_name}' is not set. "
                    f"Set it or provide a default: ${{{var_name}:-default_value}}"
                )

        return re.sub(pattern, replace_var, value)

    elif isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}

    elif isinstance(value, list):
        return [_expand_env_vars(item) for item in value]

    else:
        return value


def load_config(
    path: Path = Path("config.yaml"),
    env_file: Optional[Path] = None
) -> Config:
    """Load configuration from YAML file with environment variable expansion.

    Supports patterns:
    - ${VAR_NAME} - Required variable, raises error if not set
    - ${VAR_NAME:-default} - Optional variable with default value

    Args:
        path: Path to YAML config file
        env_file: Optional path to .env file to load

    Returns:
        Config instance with all settings

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If required environment variable is not set
        yaml.YAMLError: If YAML is invalid
    """
    # Load .env file if specified or if .env exists
    if env_file is not None:
        load_dotenv(env_file)
    elif Path(".env").exists():
        load_dotenv(Path(".env"))

    # Check if config file exists
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    # Load YAML
    with open(path, 'r') as f:
        raw_config = yaml.safe_load(f)

    if raw_config is None:
        raw_config = {}

    # Expand environment variables
    expanded_config = _expand_env_vars(raw_config)

    # Handle boolean string conversion for YAML loaded values
    expanded_config = _convert_boolean_strings(expanded_config)

    # Create and return Config instance
    return Config(**expanded_config)


def _convert_boolean_strings(value: Any) -> Any:
    """Convert string boolean values to actual booleans.

    Handles cases where env vars return 'true'/'false' strings.
    """
    if isinstance(value, str):
        if value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False
        return value
    elif isinstance(value, dict):
        return {k: _convert_boolean_strings(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_convert_boolean_strings(item) for item in value]
    return value


def apply_cli_overrides(config: Config, **overrides: Any) -> Config:
    """Apply CLI argument overrides to configuration.

    Supports dot notation for nested values:
    - sampling.min_steps=3
    - quality.min_score=4.0
    - graph.path=.cache/mygraph.pkl

    Args:
        config: Base configuration
        **overrides: Key-value pairs to override (use dot notation keys)

    Returns:
        New Config instance with overrides applied

    Example:
        >>> config = Config()
        >>> new_config = apply_cli_overrides(config, **{"sampling.min_steps": 3})
        >>> new_config.sampling.min_steps
        3
    """
    if not overrides:
        return config

    # Convert config to dict
    config_dict = config.model_dump()

    # Apply overrides
    for key, value in overrides.items():
        if value is None:
            continue

        parts = key.split('.')

        # Navigate to the correct nested dict
        current = config_dict
        for part in parts[:-1]:
            if part not in current:
                # Skip invalid keys silently
                break
            current = current[part]
        else:
            # Set the value
            final_key = parts[-1]
            if final_key in current:
                current[final_key] = value

    # Create new Config from modified dict
    return Config(**config_dict)


def get_default_config() -> Config:
    """Return a Config instance with all defaults.

    Returns:
        Config instance with default values for all settings
    """
    return Config()


def save_config(config: Config, path: Path) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration to save
        path: Output path for YAML file
    """
    config_dict = config.model_dump()

    # Convert Path objects to strings for YAML serialization
    def convert_paths(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: convert_paths(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_paths(item) for item in obj]
        return obj

    config_dict = convert_paths(config_dict)

    with open(path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def validate_config(config: Config) -> List[str]:
    """Validate configuration and return list of warnings/issues.

    Does not raise - returns human-readable messages for any issues found.

    Args:
        config: Configuration to validate

    Returns:
        List of warning/issue messages (empty if all valid)
    """
    warnings = []

    # Check graph path parent directory
    graph_dir = config.graph.path.parent
    if not graph_dir.exists():
        try:
            graph_dir.mkdir(parents=True, exist_ok=True)
            warnings.append(f"Created graph directory: {graph_dir}")
        except PermissionError:
            warnings.append(f"Cannot create graph directory: {graph_dir}")

    # Check cache directory
    if config.cache.enabled:
        cache_dir = config.cache.directory
        if not cache_dir.exists():
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                warnings.append(f"Created cache directory: {cache_dir}")
            except PermissionError:
                warnings.append(f"Cannot create cache directory: {cache_dir}")
        elif not os.access(cache_dir, os.W_OK):
            warnings.append(f"Cache directory not writable: {cache_dir}")

    # Check sampling configuration
    if config.sampling.min_steps == config.sampling.max_steps:
        warnings.append(
            f"min_steps equals max_steps ({config.sampling.min_steps}) - "
            "all chains will have the same length"
        )

    # Check quality configuration
    if config.quality.min_score > 4.0:
        warnings.append(
            f"min_score is very high ({config.quality.min_score}) - "
            "many conversations may be rejected"
        )

    # Check generation configuration
    if config.generation.parallel_workers > 8:
        warnings.append(
            f"High parallel_workers ({config.generation.parallel_workers}) - "
            "may cause rate limiting with LLM APIs"
        )

    return warnings
