"""
Embedding generator for test fixtures.

Generates embeddings using sentence-transformers models and populates
IRIS database tables with vector embeddings.
"""

from typing import List, Optional, Any, Dict
import numpy as np


# ==============================================================================
# EXCEPTION CLASSES
# ==============================================================================


class ModelLoadError(Exception):
    """Raised when embedding model fails to load."""

    def __init__(self, model_name: str, reason: str):
        self.model_name = model_name
        self.reason = reason
        super().__init__(f"Failed to load model '{model_name}': {reason}")


class DimensionMismatchError(Exception):
    """Raised when embedding dimension doesn't match model."""

    def __init__(self, model_name: str, expected: int, actual: int):
        self.model_name = model_name
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Model '{model_name}' produces {actual}-dimensional embeddings, "
            f"but {expected} dimensions were requested"
        )


# ==============================================================================
# EMBEDDING GENERATOR
# ==============================================================================

# Global model cache to avoid reloading models
_MODEL_CACHE: Dict[str, Any] = {}


class EmbeddingGenerator:
    """
    Generate embeddings using sentence-transformers models.

    This class provides a simple interface for generating embeddings and
    populating IRIS database tables with vector embeddings.

    Features:
    - Automatic model loading and caching
    - Batch processing for efficiency
    - NULL/empty text handling (zero vectors)
    - L2 normalization support
    - CPU/CUDA device support
    - IRIS TO_VECTOR() integration

    Example:
        >>> generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
        >>> texts = ["Hello world", "Test text"]
        >>> embeddings = generator.generate_embeddings(texts)
        >>> print(embeddings.shape)
        (2, 384)

        >>> # Populate database table
        >>> generator.populate_table_embeddings(
        ...     connection=conn,
        ...     table_name="RAG.SourceDocuments",
        ...     text_column="content",
        ...     embedding_column="embedding"
        ... )
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        dimension: int = 384,
        batch_size: int = 32,
        device: Optional[str] = None,
        normalize: bool = False,
    ):
        """
        Initialize EmbeddingGenerator.

        Args:
            model_name: Sentence-transformers model name
            dimension: Expected embedding dimension
            batch_size: Batch size for processing
            device: Device to use ('cpu', 'cuda', or None for auto-detect)
            normalize: Whether to L2-normalize embeddings

        Raises:
            ModelLoadError: Model failed to load
            DimensionMismatchError: Dimension mismatch with model
        """
        self.model_name = model_name
        self.dimension = dimension
        self.batch_size = batch_size
        self.normalize = normalize

        # Auto-detect device if not specified
        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        self.device = device

        # Load model (with caching)
        self._model = self._load_model()

        # Validate dimension matches model output
        self._validate_dimension()

    def generate_embeddings(self, texts: List[Optional[str]]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings (None/empty -> zero vector)

        Returns:
            Numpy array of shape (len(texts), dimension)
        """
        # Handle empty input
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        # Process texts in batches
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            # Replace None/empty with empty string for model
            clean_batch = []
            null_indices = []

            for j, text in enumerate(batch):
                if text is None or (isinstance(text, str) and text.strip() == ""):
                    clean_batch.append("")  # Placeholder
                    null_indices.append(j)
                else:
                    clean_batch.append(text)

            # Generate embeddings for batch
            batch_embeddings = self._model.encode(
                clean_batch,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )

            # Replace null/empty embeddings with zero vectors
            for idx in null_indices:
                batch_embeddings[idx] = np.zeros(self.dimension, dtype=np.float32)

            all_embeddings.append(batch_embeddings)

        # Concatenate all batches
        embeddings = np.vstack(all_embeddings)

        return embeddings.astype(np.float32)

    def populate_table_embeddings(
        self,
        connection: Any,
        table_name: str,
        text_column: str,
        embedding_column: str,
        batch_size: Optional[int] = None,
        id_column: str = "id",
    ) -> int:
        """
        Populate embedding column in IRIS table.

        This method fetches rows from the table in batches, generates embeddings,
        and updates the embedding column using IRIS's TO_VECTOR() function.

        Args:
            connection: IRIS database connection
            table_name: Full table name (e.g., "RAG.SourceDocuments")
            text_column: Column containing text to embed
            embedding_column: Column to store embeddings
            batch_size: Batch size for fetching/updating (default: self.batch_size)
            id_column: Primary key column for updates

        Returns:
            Total number of rows updated

        Raises:
            Exception: Database operation failed

        Note:
            Uses TO_VECTOR() function for IRIS compatibility. This is required
            because IRIS VECTOR columns cannot be directly inserted via SQL without
            the TO_VECTOR() wrapper.
        """
        if batch_size is None:
            batch_size = self.batch_size

        cursor = connection.cursor()
        total_updated = 0

        try:
            # Get total row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            total_rows = cursor.fetchone()[0]

            # Process in batches
            offset = 0

            while offset < total_rows:
                # Fetch batch of rows
                cursor.execute(
                    f"SELECT {id_column}, {text_column} FROM {table_name} "
                    f"ORDER BY {id_column} LIMIT {batch_size} OFFSET {offset}"
                )
                rows = cursor.fetchall()

                if not rows:
                    break

                # Extract IDs and texts
                ids = [row[0] for row in rows]
                texts = [row[1] for row in rows]

                # Generate embeddings
                embeddings = self.generate_embeddings(texts)

                # Update database using TO_VECTOR()
                for row_id, embedding in zip(ids, embeddings):
                    # Convert numpy array to list, then to string
                    embedding_list = embedding.tolist()
                    embedding_str = "[" + ",".join(str(x) for x in embedding_list) + "]"

                    # Use TO_VECTOR() function for IRIS compatibility
                    update_sql = f"""
                        UPDATE {table_name}
                        SET {embedding_column} = TO_VECTOR(?, FLOAT, {self.dimension})
                        WHERE {id_column} = ?
                    """

                    cursor.execute(update_sql, [embedding_str, row_id])
                    total_updated += 1

                # Commit batch
                connection.commit()

                # Move to next batch
                offset += batch_size

            return total_updated

        except Exception as e:
            # Rollback on error
            connection.rollback()
            raise Exception(f"Failed to populate embeddings for {table_name}: {e}")

        finally:
            cursor.close()

    # ==========================================================================
    # PRIVATE HELPER METHODS
    # ==========================================================================

    def _load_model(self) -> Any:
        """
        Load sentence-transformers model with caching.

        Returns:
            SentenceTransformer model instance

        Raises:
            ModelLoadError: Model loading failed
        """
        # Check cache first
        cache_key = f"{self.model_name}:{self.device}"
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]

        # Import sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ModelLoadError(
                self.model_name,
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers",
            )

        # Load model
        try:
            model = SentenceTransformer(self.model_name, device=self.device)
            _MODEL_CACHE[cache_key] = model
            return model
        except Exception as e:
            raise ModelLoadError(self.model_name, str(e))

    def _validate_dimension(self) -> None:
        """
        Validate that model dimension matches requested dimension.

        Raises:
            DimensionMismatchError: Dimension mismatch
        """
        # Generate a test embedding to check dimension
        test_embedding = self._model.encode(
            ["test"], convert_to_numpy=True, show_progress_bar=False
        )
        actual_dim = test_embedding.shape[1]

        if actual_dim != self.dimension:
            raise DimensionMismatchError(self.model_name, self.dimension, actual_dim)
