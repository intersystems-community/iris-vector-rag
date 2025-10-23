"""
Embedding manager with fallback support for RAG templates.

This module provides a unified interface for embedding generation with support
for multiple backends and graceful fallback mechanisms.
"""

import logging
from typing import Callable, Dict, List, Optional

from ..config.manager import ConfigurationManager

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Manages embedding generation with multiple backends and fallback support.

    Provides a unified interface for generating embeddings from text, with
    automatic fallback to alternative backends if the primary fails.
    """

    def __init__(self, config_manager: ConfigurationManager):
        """
        Initialize the embedding manager.

        Args:
            config_manager: Configuration manager for embedding settings
        """
        self.config_manager = config_manager
        self.embedding_config = self.config_manager.get("embeddings", {})

        # Get primary and fallback backends
        self.primary_backend = self.embedding_config.get(
            "primary_backend", "sentence_transformers"
        )
        self.fallback_backends = self.embedding_config.get(
            "fallback_backends", ["openai"]
        )

        # Cache for loaded embedding functions
        self._embedding_functions: Dict[str, Callable] = {}

        # Initialize primary backend
        self._initialize_backend(self.primary_backend)

    def _initialize_backend(self, backend_name: str) -> bool:
        """
        Initialize a specific embedding backend.

        Args:
            backend_name: Name of the backend to initialize

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            if backend_name == "sentence_transformers":
                self._embedding_functions[backend_name] = (
                    self._create_sentence_transformers_function()
                )
            elif backend_name == "openai":
                self._embedding_functions[backend_name] = self._create_openai_function()
            elif backend_name == "huggingface":
                self._embedding_functions[backend_name] = (
                    self._create_huggingface_function()
                )
            else:
                logger.warning(f"Unknown embedding backend: {backend_name}")
                return False

            logger.info(f"Successfully initialized embedding backend: {backend_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize embedding backend {backend_name}: {e}")
            return False

    def _create_sentence_transformers_function(self) -> Callable:
        """Create sentence transformers embedding function."""
        try:
            from sentence_transformers import SentenceTransformer

            model_name = self.embedding_config.get("sentence_transformers", {}).get(
                "model_name", "all-MiniLM-L6-v2"
            )
            # Get device from config (default to 'cpu' to avoid GPU contention)
            device = self.embedding_config.get("sentence_transformers", {}).get(
                "device", "cpu"
            )
            model = SentenceTransformer(model_name, device=device)
            logger.info(f"✅ SentenceTransformer initialized on device: {device}")

            def embed_texts(texts: List[str]) -> List[List[float]]:
                embeddings = model.encode(texts, convert_to_tensor=False)
                return embeddings.tolist()

            return embed_texts

        except ImportError:
            logger.error(
                "sentence-transformers not available. Install with: pip install sentence-transformers"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to create sentence transformers function: {e}")
            raise

    def _create_openai_function(self) -> Callable:
        """Create OpenAI embedding function."""
        try:
            import openai

            openai_config = self.embedding_config.get("openai", {})
            api_key = openai_config.get("api_key") or self.config_manager.get(
                "openai:api_key"
            )
            model_name = openai_config.get("model_name", "text-embedding-ada-002")

            if not api_key:
                raise ValueError("OpenAI API key not found in configuration")

            client = openai.OpenAI(api_key=api_key)

            def embed_texts(texts: List[str]) -> List[List[float]]:
                response = client.embeddings.create(input=texts, model=model_name)
                return [embedding.embedding for embedding in response.data]

            return embed_texts

        except ImportError:
            logger.error("openai not available. Install with: pip install openai")
            raise
        except Exception as e:
            logger.error(f"Failed to create OpenAI function: {e}")
            raise

    def _create_huggingface_function(self) -> Callable:
        """Create Hugging Face embedding function."""
        try:
            import torch

            from common.huggingface_utils import download_huggingface_model

            hf_config = self.embedding_config.get("huggingface", {})
            model_name = hf_config.get(
                "model_name", "sentence-transformers/all-MiniLM-L6-v2"
            )

            tokenizer, model = download_huggingface_model(model_name)

            def embed_texts(texts: List[str]) -> List[List[float]]:
                # Tokenize and encode
                encoded_input = tokenizer(
                    texts, padding=True, truncation=True, return_tensors="pt"
                )

                # Generate embeddings
                with torch.no_grad():
                    model_output = model(**encoded_input)
                    # Use mean pooling
                    embeddings = model_output.last_hidden_state.mean(dim=1)

                return embeddings.tolist()

            return embed_texts

        except ImportError:
            logger.error(
                "transformers not available. Install with: pip install transformers torch"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to create Hugging Face function: {e}")
            raise

    def _create_fallback_function(self) -> Callable:
        """Create a simple fallback embedding function."""

        def embed_texts(texts: List[str]) -> List[List[float]]:
            """
            Simple fallback that creates basic embeddings based on text length and hash.
            This is not suitable for production but allows the system to continue functioning.
            """
            import hashlib

            embeddings = []
            for text in texts:
                # Handle None or empty text
                if text is None:
                    text = ""
                elif not isinstance(text, str):
                    text = str(text)

                # Create a simple embedding based on text characteristics
                text_hash = hashlib.md5(text.encode()).hexdigest()

                # Convert hash to numbers and normalize
                hash_numbers = [
                    int(text_hash[i : i + 2], 16) for i in range(0, len(text_hash), 2)
                ]

                # Pad or truncate to desired dimension (get from config or use 384 fallback)
                target_dim = self.embedding_config.get("dimension", 384)
                while len(hash_numbers) < target_dim:
                    hash_numbers.extend(hash_numbers[: target_dim - len(hash_numbers)])
                hash_numbers = hash_numbers[:target_dim]

                # Normalize to [-1, 1] range
                normalized = [(x - 127.5) / 127.5 for x in hash_numbers]
                embeddings.append(normalized)

            logger.warning(f"Using fallback embeddings for {len(texts)} texts")
            return embeddings

        return embed_texts

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as list of floats
        """
        embeddings = self.embed_texts([text])
        return embeddings[0]

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text (alias for embed_text for compatibility).

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as list of floats
        """
        return self.embed_text(text)

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (alias for embed_texts for HybridGraphRAG compatibility).

        Args:
            texts: List of input texts to embed

        Returns:
            List of embedding vectors
        """
        return self.embed_texts(texts)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with fallback support.

        Args:
            texts: List of input texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Try primary backend first
        if self.primary_backend in self._embedding_functions:
            try:
                return self._embedding_functions[self.primary_backend](texts)
            except Exception as e:
                logger.warning(f"Primary backend {self.primary_backend} failed: {e}")

        # Try fallback backends
        for backend_name in self.fallback_backends:
            if backend_name not in self._embedding_functions:
                if not self._initialize_backend(backend_name):
                    continue

            try:
                logger.info(f"Using fallback backend: {backend_name}")
                return self._embedding_functions[backend_name](texts)
            except Exception as e:
                logger.warning(f"Fallback backend {backend_name} failed: {e}")
                continue

        # If all backends fail, use simple fallback
        logger.warning("All embedding backends failed, using simple fallback")
        fallback_func = self._create_fallback_function()
        return fallback_func(texts)

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by the current backend.

        Returns:
            Embedding dimension
        """
        # Try to get dimension from config first
        dimension = self.embedding_config.get("dimension")
        if dimension:
            return dimension

        # Try to get from embedding config's model mapping
        model_name = self.embedding_config.get("sentence_transformers", {}).get(
            "model_name", "all-MiniLM-L6-v2"
        )

        # Use direct model-to-dimension mapping instead of dimension utils
        known_dimensions = {
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "bert-base-uncased": 768,
            "bert-large-uncased": 1024,
        }

        if model_name in known_dimensions:
            return known_dimensions[model_name]

        # Otherwise, generate a test embedding to determine dimension
        try:
            test_embedding = self.embed_text("test")
            return len(test_embedding)
        except Exception as e:
            # HARD FAIL - no fallback dimensions to hide configuration issues
            error_msg = f"CRITICAL: Cannot determine embedding dimension: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def is_available(self, backend_name: Optional[str] = None) -> bool:
        """
        Check if a specific backend or any backend is available.

        Args:
            backend_name: Optional specific backend to check

        Returns:
            True if backend(s) available, False otherwise
        """
        if backend_name:
            return backend_name in self._embedding_functions

        # Check if any backend is available
        return len(self._embedding_functions) > 0

    def get_available_backends(self) -> List[str]:
        """
        Get list of currently available backends.

        Returns:
            List of available backend names
        """
        return list(self._embedding_functions.keys())

    def switch_backend(self, backend_name: str) -> bool:
        """
        Switch to a different primary backend.

        Args:
            backend_name: Name of backend to switch to

        Returns:
            True if switch successful, False otherwise
        """
        if backend_name not in self._embedding_functions:
            if not self._initialize_backend(backend_name):
                return False

        self.primary_backend = backend_name
        logger.info(f"Switched to primary backend: {backend_name}")
        return True
