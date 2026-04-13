"""Text Embeddings - Generation and caching for endpoint embeddings.

This module provides embedding generation for tool endpoints using either
OpenAI's API or local sentence-transformers models, with persistent caching.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingDimensionError(Exception):
    """Exception raised when embedding dimension doesn't match expected.

    Attributes:
        expected: The expected embedding dimension
        actual: The actual embedding dimension received
    """

    def __init__(self, expected: int, actual: int) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Embedding dimension mismatch: expected {expected}, got {actual}"
        )


class EmbeddingCache:
    """Cache for storing and retrieving embeddings.

    Provides persistent storage of embeddings in .npz format with
    dimension validation to ensure consistency.

    Attributes:
        cache_path: Path to the .npz cache file
        _embeddings: In-memory cache mapping endpoint_id -> embedding
        _dimension: Embedding dimension (set on first add, validated after)

    Example:
        >>> cache = EmbeddingCache(Path(".cache/embeddings.npz"))
        >>> cache.load()
        >>> cache.set("ep_1", np.array([0.1, 0.2, 0.3]))
        >>> embedding = cache.get("ep_1")
        >>> cache.save()
    """

    def __init__(self, cache_path: Path) -> None:
        """Initialize the EmbeddingCache.

        Args:
            cache_path: Path to the .npz file for persistence
        """
        self.cache_path = cache_path
        self._embeddings: Dict[str, np.ndarray] = {}
        self._dimension: Optional[int] = None

    @property
    def dimension(self) -> Optional[int]:
        """Return the embedding dimension, or None if not set."""
        return self._dimension

    def load(self) -> None:
        """Load embeddings from .npz file if it exists."""
        if not self.cache_path.exists():
            logger.debug(f"Cache file not found: {self.cache_path}")
            return

        try:
            data = np.load(self.cache_path)
            self._embeddings = {key: data[key] for key in data.files}
            if self._embeddings:
                # Set dimension from first embedding
                first_key = next(iter(self._embeddings))
                self._dimension = self._embeddings[first_key].shape[0]
            logger.info(f"Loaded {len(self._embeddings)} embeddings from cache")
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            self._embeddings = {}
            self._dimension = None

    def save(self) -> None:
        """Save embeddings to .npz file."""
        # Ensure parent directory exists
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            np.savez(self.cache_path, **self._embeddings)
            logger.info(f"Saved {len(self._embeddings)} embeddings to cache")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
            raise

    def get(self, endpoint_id: str) -> Optional[np.ndarray]:
        """Get cached embedding for an endpoint.

        Args:
            endpoint_id: The endpoint ID to look up

        Returns:
            The embedding array if cached, None otherwise
        """
        return self._embeddings.get(endpoint_id)

    def set(self, endpoint_id: str, embedding: np.ndarray) -> None:
        """Cache an embedding for an endpoint.

        Args:
            endpoint_id: The endpoint ID to cache
            embedding: The embedding array to cache

        Raises:
            EmbeddingDimensionError: If embedding dimension doesn't match
        """
        dim = embedding.shape[0]

        if self._dimension is None:
            self._dimension = dim
        elif dim != self._dimension:
            raise EmbeddingDimensionError(self._dimension, dim)

        self._embeddings[endpoint_id] = embedding

    def has(self, endpoint_id: str) -> bool:
        """Check if an endpoint is cached.

        Args:
            endpoint_id: The endpoint ID to check

        Returns:
            True if the endpoint is cached, False otherwise
        """
        return endpoint_id in self._embeddings

    def get_all(self) -> Dict[str, np.ndarray]:
        """Return all cached embeddings.

        Returns:
            Dictionary mapping endpoint_id to embedding array
        """
        return dict(self._embeddings)

    def clear(self) -> None:
        """Clear all cached embeddings."""
        self._embeddings.clear()
        self._dimension = None

    def __len__(self) -> int:
        """Return the number of cached embeddings."""
        return len(self._embeddings)


def get_embedding_text(endpoint_data: Dict[str, Any]) -> str:
    """Combine endpoint fields into text for embedding.

    Formats endpoint data into a text string suitable for embedding
    generation. Handles missing fields gracefully.

    Args:
        endpoint_data: Dictionary containing endpoint attributes
            (name, description, domain, method, path)

    Returns:
        Formatted text string for embedding

    Example:
        >>> data = {"name": "Get Weather", "description": "Gets weather", "domain": "weather"}
        >>> get_embedding_text(data)
        'Get Weather: Gets weather. Domain: weather.'
    """
    parts = []

    name = endpoint_data.get("name", "")
    description = endpoint_data.get("description", "")
    domain = endpoint_data.get("domain", "")
    method = endpoint_data.get("method", "")
    path = endpoint_data.get("path", "")

    # Build the text string
    if name:
        if description:
            parts.append(f"{name}: {description}")
        else:
            parts.append(name)

    if domain:
        parts.append(f"Domain: {domain}")

    if method:
        parts.append(f"Method: {method}")

    if path:
        parts.append(f"Path: {path}")

    return ". ".join(parts) + "." if parts else ""


class EmbeddingGenerator:
    """Generates embeddings for endpoint text using OpenAI or local models.

    Supports both OpenAI's embedding API and local sentence-transformers
    models for offline use. Integrates with EmbeddingCache for efficiency.

    Attributes:
        model: The embedding model name
        cache: Optional EmbeddingCache for caching results
        use_openai: Whether to use OpenAI API (True) or local models (False)
        show_progress: Whether to display tqdm progress bars

    Example:
        >>> generator = EmbeddingGenerator(model="text-embedding-3-small")
        >>> embedding = generator.generate_embedding("Get weather data")
        >>> embeddings = generator.generate_for_endpoints(["ep_1", "ep_2"], client)
    """

    # Batch size for OpenAI API requests
    OPENAI_BATCH_SIZE = 100

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        cache: Optional[EmbeddingCache] = None,
        use_openai: bool = True,
        show_progress: bool = True,
    ) -> None:
        """Initialize the EmbeddingGenerator.

        Args:
            model: Embedding model name (OpenAI model or sentence-transformers model)
            cache: Optional EmbeddingCache instance for caching
            use_openai: If True, use OpenAI API; if False, use sentence-transformers
            show_progress: Whether to display tqdm progress bars
        """
        self.model = model
        self.cache = cache
        self.use_openai = use_openai
        self.show_progress = show_progress

        self._openai_client = None
        self._local_model = None

        # Only initialize local model eagerly (it doesn't require credentials)
        # OpenAI client is lazy-initialized on first use
        if not use_openai:
            self._init_local()

    def _get_openai_client(self):
        """Get or create OpenAI client (lazy initialization)."""
        if self._openai_client is None:
            try:
                from openai import OpenAI

                self._openai_client = OpenAI()
                logger.debug("OpenAI client initialized")
            except ImportError:
                raise ImportError(
                    "openai package not installed. Install with: pip install openai"
                )
        return self._openai_client

    def _init_local(self) -> None:
        """Initialize local sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer

            self._local_model = SentenceTransformer(self.model)
            logger.debug(f"Local model '{self.model}' initialized")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

    def get_embedding_text(self, endpoint_data: Dict[str, Any]) -> str:
        """Combine endpoint fields into text for embedding.

        This is a method wrapper around the module-level function
        for convenience.

        Args:
            endpoint_data: Dictionary containing endpoint attributes

        Returns:
            Formatted text string for embedding
        """
        return get_embedding_text(endpoint_data)

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: The text to embed

        Returns:
            Embedding as numpy array
        """
        if self.use_openai:
            return self._generate_openai([text])[0]
        else:
            return self._generate_local([text])[0]

    def generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings as numpy arrays
        """
        if not texts:
            return []

        if self.use_openai:
            return self._generate_openai(texts)
        else:
            return self._generate_local(texts)

    def _generate_openai(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings using OpenAI API.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings as numpy arrays
        """
        client = self._get_openai_client()
        response = client.embeddings.create(
            model=self.model,
            input=texts,
        )

        # Extract embeddings from response, maintaining order
        embeddings = [np.array(item.embedding, dtype=np.float32) for item in response.data]
        return embeddings

    def _generate_local(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings using local sentence-transformers.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings as numpy arrays
        """
        embeddings = self._local_model.encode(texts, convert_to_numpy=True)
        # Ensure float32 and return as list
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.astype(np.float32)
            return [embeddings[i] for i in range(len(texts))]
        return [np.array(emb, dtype=np.float32) for emb in embeddings]

    def generate_for_endpoints(
        self,
        endpoint_ids: List[str],
        client: "GraphClient",  # noqa: F821 - imported at runtime
    ) -> Dict[str, np.ndarray]:
        """Generate embeddings for multiple endpoints.

        This is the main method for batch embedding generation. It:
        1. Checks cache for existing embeddings
        2. Generates embeddings only for uncached endpoints
        3. Updates cache with new embeddings

        Args:
            endpoint_ids: List of endpoint IDs to generate embeddings for
            client: GraphClient instance to fetch endpoint data

        Returns:
            Dictionary mapping endpoint_id to embedding array
        """
        from tqdm import tqdm

        result: Dict[str, np.ndarray] = {}

        # Separate cached and uncached endpoints
        uncached_ids = []
        for ep_id in endpoint_ids:
            if self.cache and self.cache.has(ep_id):
                result[ep_id] = self.cache.get(ep_id)
            else:
                uncached_ids.append(ep_id)

        if not uncached_ids:
            logger.debug("All endpoints already cached")
            return result

        logger.info(f"Generating embeddings for {len(uncached_ids)} endpoints")

        # Prepare texts for uncached endpoints
        texts = []
        valid_ids = []
        for ep_id in uncached_ids:
            endpoint_data = client.get_endpoint_by_id(ep_id)
            if endpoint_data:
                text = self.get_embedding_text(endpoint_data)
                texts.append(text)
                valid_ids.append(ep_id)
            else:
                logger.warning(f"Endpoint not found: {ep_id}")

        if not texts:
            return result

        # Generate embeddings in batches
        all_embeddings: List[np.ndarray] = []

        if self.use_openai:
            # Process in batches for OpenAI
            batches = [
                texts[i : i + self.OPENAI_BATCH_SIZE]
                for i in range(0, len(texts), self.OPENAI_BATCH_SIZE)
            ]

            batch_iter = batches
            if self.show_progress:
                batch_iter = tqdm(batches, desc="Generating embeddings...", unit="batch")

            for batch in batch_iter:
                batch_embeddings = self._generate_openai(batch)
                all_embeddings.extend(batch_embeddings)
        else:
            # Local model can handle all at once
            if self.show_progress:
                logger.info("Generating embeddings with local model...")
            all_embeddings = self._generate_local(texts)

        # Store results and update cache
        for ep_id, embedding in zip(valid_ids, all_embeddings):
            result[ep_id] = embedding
            if self.cache:
                self.cache.set(ep_id, embedding)

        return result
