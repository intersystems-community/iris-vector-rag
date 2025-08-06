# common/utils.py
import time
from dataclasses import dataclass, field
from typing import List, Callable, Any, Optional, Dict, Tuple # Added Tuple
# import sqlalchemy  # Removed - not needed for IRIS
import os
import logging # Added for logger usage in get_llm_func
import numpy as np
import yaml # Added for config loading
from pathlib import Path # Added for config path

logger = logging.getLogger(__name__) # Added for logger usage

# --- Config Loading ---
_config = None

def load_config():
    global _config
    if _config is None:
        try:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"
            with open(config_path, 'r') as f:
                _config = yaml.safe_load(f)
            logger.info(f"Successfully loaded config from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load config/config.yaml: {e}. Using default values.")
            _config = {} # Ensure _config is not None
    return _config

def get_config_value(key_path: str, default: Any = None) -> Any:
    config = load_config()
    keys = key_path.split('.')
    value = config
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        logger.warning(f"Config key '{key_path}' not found. Returning default: {default}")
        return default

# --- Dataclasses ---
@dataclass
class Document:
    id: str
    content: str
    score: Optional[float] = None # For similarity score from retrieval
    metadata: Optional[Dict[str, Any]] = None # For title or other metadata
    embedding: Optional[List[float]] = field(default=None, repr=False) # Standard document embedding
    # For ColBERT or other token-level models
    colbert_tokens: Optional[List[str]] = field(default=None, repr=False)
    colbert_token_embeddings: Optional[List[List[float]]] = field(default=None, repr=False) # Raw token embeddings
    colbert_compressed_embeddings: Optional[List[Any]] = field(default=None, repr=False) # Compressed + scale factors

    def to_dict(self, include_embeddings: bool = False) -> Dict[str, Any]:
        data = {
            "id": self.id,
            "content": self.content,
            "score": float(self.score) if self.score is not None else None,
        }
        if include_embeddings:
            data["embedding"] = self.embedding
            # Potentially add colbert fields if needed for serialization
            # data["colbert_tokens"] = self.colbert_tokens 
            # data["colbert_token_embeddings"] = self.colbert_token_embeddings
        return data

# --- Model and Connector Wrappers ---

_llm_instance = None
_current_llm_key = None # For caching LLM instance

DEFAULT_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIMENSION = 384

# New pure HuggingFace embedder
_hf_embedder_cache = {}

def build_hf_embedder(model_name: str):
    """
    Builds an embedding function using HuggingFace transformers directly.
    Includes tokenization, model inference, mean pooling, and normalization.
    Caches tokenizer and model per model_name.
    """
    global _hf_embedder_cache
    import torch
    import functools
    from transformers import AutoTokenizer, AutoModel

    if model_name not in _hf_embedder_cache:
        from common.huggingface_utils import download_huggingface_model
        print(f"Initializing HF embedder for model: {model_name}")
        tokenizer, model = download_huggingface_model(model_name)
        model.eval() # Set to evaluation mode
        # Consider model.to(device) if GPU is available/desired
        _hf_embedder_cache[model_name] = (tokenizer, model)
    else:
        print(f"Using cached HF embedder for model: {model_name}")
    
    tokenizer, model = _hf_embedder_cache[model_name]

    # Removed LRU cache to prevent memory leaks during bulk operations
    # Cache causes indefinite memory growth when processing thousands of unique texts
    def _embed_single_text(text: str) -> List[float]:
        with torch.no_grad():
            # Validate input text
            if not text or not text.strip():
                logger.warning("Empty or whitespace-only text provided for embedding")
                # Use configured dimension for zero vector
                dimension = get_config_value("embedding_model.dimension", DEFAULT_EMBEDDING_DIMENSION)
                return [0.0] * dimension
            
            try:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
                outputs = model(**inputs)
                token_embeddings = outputs.last_hidden_state
                attention_mask = inputs.attention_mask
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                pooled_embedding = sum_embeddings / sum_mask
                normalized_embedding = torch.nn.functional.normalize(pooled_embedding, p=2, dim=1)
                
                # Convert to numpy for NaN/inf checking
                embedding_array = normalized_embedding[0].cpu().numpy()
                
                # Check for NaN or inf values and fix them
                if np.any(np.isnan(embedding_array)) or np.any(np.isinf(embedding_array)):
                    logger.warning(f"NaN/inf values detected in embedding for text: {text[:50]}...")
                    embedding_array = np.nan_to_num(embedding_array, nan=0.0, posinf=1.0, neginf=-1.0)
                    # Re-normalize after fixing
                    norm = np.linalg.norm(embedding_array)
                    if norm > 0:
                        embedding_array = embedding_array / norm
                    else:
                        embedding_array = np.zeros_like(embedding_array)
                
                return embedding_array.tolist()
                
            except Exception as e:
                logger.error(f"Error generating embedding for text '{text[:50]}...': {e}")
                dimension = get_config_value("embedding_model.dimension", DEFAULT_EMBEDDING_DIMENSION)
                return [0.0] * dimension

    def embedding_func_hf(texts: List[str]) -> List[List[float]]:
        # Add memory pressure detection for bulk operations
        if len(texts) > 10:  # Bulk operation detected
            try:
                import psutil
                import gc
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                
                # If memory usage exceeds 2GB, force cleanup
                if memory_mb > 2048:
                    logger.warning(f"High memory usage detected: {memory_mb:.1f} MB. Forcing cleanup...")
                    gc.collect()
                    # Clear torch cache if available
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except ImportError:
                        pass
            except ImportError:
                pass  # psutil not available
        
        return [_embed_single_text(t) for t in texts]

    return embedding_func_hf


def get_embedding_func(model_name_override: Optional[str] = None, provider: Optional[str] = None, mock: bool = False) -> Callable:
    """
    Returns a function that takes either a single text string or a list of texts and returns embeddings.
    Reads model name from config/config.yaml, with override option.
    Supports a "stub" provider or mock=True for testing without real models.
    
    The returned function handles both:
    - Single string: embedding_func("text") -> List[float]
    - List of strings: embedding_func(["text1", "text2"]) -> List[List[float]]
    """
    effective_model_name = model_name_override or get_config_value("embedding_model.name", DEFAULT_EMBEDDING_MODEL_NAME)
    dimension = get_config_value("embedding_model.dimension", DEFAULT_EMBEDDING_DIMENSION)

    if mock or provider == "stub" or effective_model_name == "stub":
        logger.info(f"Using stub embedding function with dimension {dimension}.")
        def stub_embed_texts(texts) -> List[List[float]]:
            # Handle both single string and list of strings
            if isinstance(texts, str):
                return [(len(texts) % 100) * 0.01] * dimension
            return [[(len(text) % 100) * 0.01] * dimension for text in texts]
        return stub_embed_texts
    
    logger.info(f"Using pure HuggingFace embedder for model: {effective_model_name}")
    base_embedder = build_hf_embedder(effective_model_name)
    
    def flexible_embedder(texts):
        """Wrapper that handles both single strings and lists of strings"""
        if isinstance(texts, str):
            # Single string input - return single embedding
            result = base_embedder([texts])
            return result[0] if result else [0.0] * dimension
        else:
            # List input - return list of embeddings
            return base_embedder(texts)
    
    return flexible_embedder


def get_llm_func(provider: str = "openai", model_name: str = "gpt-4.1-mini",
                enable_cache: Optional[bool] = None, **kwargs) -> Callable[[str], str]:
    """
    Returns a function that takes a prompt string and returns an LLM completion string.
    Supports 'openai' or a 'stub' for testing.
    
    Args:
        provider: LLM provider (openai, stub, etc.)
        model_name: Model name
        enable_cache: Override cache enable/disable (None = use config default)
        **kwargs: Additional LLM parameters
    
    Returns:
        LLM function (automatically cached if enabled)
    """
    global _llm_instance, _current_llm_key
    
    # Setup caching if enabled
    cache_enabled = _setup_caching_if_needed(enable_cache)
    
    llm_key = f"{provider}_{model_name}_{cache_enabled}"

    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        try:
            from dotenv import load_dotenv, find_dotenv
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            dotenv_path = os.path.join(project_root, ".env")
            if os.path.exists(dotenv_path):
                logger.info(f"Attempting to load .env file from {dotenv_path} in get_llm_func.")
                load_dotenv(dotenv_path=dotenv_path, override=True)
            else:
                env_path_found = find_dotenv(usecwd=True)
                if env_path_found:
                    logger.info(f"Attempting to load .env file from found path {env_path_found} in get_llm_func.")
                    load_dotenv(dotenv_path=env_path_found, override=True)
                else:
                    logger.warning("No .env file found by find_dotenv() in get_llm_func.")
        except ImportError:
            if not logger.hasHandlers(): logging.basicConfig(level=logging.INFO)
            logger.warning("python-dotenv not installed. Cannot load .env file in get_llm_func.")
        except Exception as e_dotenv:
            if not logger.hasHandlers(): logging.basicConfig(level=logging.INFO)
            logger.warning(f"Error loading .env file in get_llm_func: {e_dotenv}")

    if _llm_instance is None or _current_llm_key != llm_key:
        print(f"Initializing LLM: {provider} - {model_name}")
        if provider == "openai":
            try:
                from langchain_openai import ChatOpenAI
            except ImportError:
                raise ImportError("LangChain OpenAI library not found. Please install with `poetry add langchain-openai`.")
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set or found in .env file.")
            
            _llm_instance = ChatOpenAI(model_name=model_name, openai_api_key=api_key, **kwargs)
            _current_llm_key = llm_key

        elif provider == "stub":
            class StubLLM:
                def __init__(self, model_name, **kwargs):
                    self.model_name = model_name
                def invoke(self, prompt: str, **kwargs) -> Any:
                    response_content = f"Stub LLM response for prompt: '{prompt[:50]}...'"
                    class AIMessage:
                        def __init__(self, content):
                            self.content = content
                    return AIMessage(content=response_content)
            _llm_instance = StubLLM(model_name=model_name, **kwargs)
            _current_llm_key = llm_key 
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def query_llm(prompt: str) -> str:
        response = _llm_instance.invoke(prompt)
        # Ensure the response is always a string
        if hasattr(response, 'content'):
            return str(response.content)
        return str(response)

    return query_llm

# --- ColBERT Specific Encoders ---
# Placeholder for actual ColBERT model loading and encoding
# For now, a mock function.
# A real implementation would use a ColBERT checkpoint.
# Expected output for loader: List[Tuple[str, List[float]]] -> [(token_text, token_embedding_vector), ...]
def get_colbert_doc_encoder_func(model_name: str = None) -> Callable[[str], List[Tuple[str, List[float]]]]:
    """
    Returns a ColBERT document encoder function based on configuration.
    Uses real ColBERT if backend is "native", otherwise falls back to mock.
    """
    # Get ColBERT configuration from config
    colbert_backend = get_config_value("colbert.backend", "native")
    colbert_model = model_name or get_config_value("colbert.model_name", "bert-base-uncased")
    token_dimension = get_config_value("colbert.token_dimension", 768)
    
    if colbert_backend == "native" and colbert_model != "stub_colbert_doc_encoder":
        logger.info(f"Using native ColBERT document encoder with model: {colbert_model}")
        
        try:
            # Import and use real ColBERT interface
            from iris_rag.embeddings.colbert_interface import RAGTemplatesColBERTInterface
            colbert = RAGTemplatesColBERTInterface(
                token_dimension=token_dimension
            )
            
            def real_colbert_doc_encode(text: str) -> List[Tuple[str, List[float]]]:
                # RAGTemplatesColBERTInterface.encode_document returns List[List[float]]
                # We need to convert to List[Tuple[str, List[float]]]
                tokens = text.split()[:100]  # Match the interface's token limit
                embeddings = colbert.encode_document(text)
                
                # Pair tokens with embeddings
                result = []
                for i, embedding in enumerate(embeddings):
                    token_text = tokens[i] if i < len(tokens) else f"token_{i}"
                    result.append((token_text, embedding))
                return result
            
            return real_colbert_doc_encode
            
        except Exception as e:
            logger.warning(f"Failed to initialize real ColBERT encoder: {e}. Falling back to mock.")
    
    # Fallback to mock implementation
    logger.info(f"Using mock ColBERT document encoder: {colbert_model}")

    def mock_colbert_doc_encode(text: str) -> List[Tuple[str, List[float]]]:
        tokens = text.split()[:100] # Limit to first 100 mock tokens
        if not tokens:
            return []
        
        token_embeddings_data = []
        for i, token_str in enumerate(tokens):
            # Create a simple mock embedding based on token index and length
            mock_embedding = [( (i % 10) + len(token_str) % 10 ) * 0.01] * token_dimension
            token_embeddings_data.append((token_str, mock_embedding))
        return token_embeddings_data

    return mock_colbert_doc_encode


def get_colbert_query_encoder_func(model_name: str = None) -> Callable[[str], List[List[float]]]:
    """
    Returns a ColBERT query encoder function based on configuration.
    Uses real ColBERT if backend is "native", otherwise falls back to mock.
    Expected output: List[List[float]] -> [token_embedding_vector, ...]
    """
    # Get ColBERT configuration from config
    colbert_backend = get_config_value("colbert.backend", "native")
    colbert_model = model_name or get_config_value("colbert.model_name", "bert-base-uncased")
    token_dimension = get_config_value("colbert.token_dimension", 768)
    
    if colbert_backend == "native" and colbert_model != "stub_colbert_query_encoder":
        logger.info(f"Using native ColBERT query encoder with model: {colbert_model}")
        
        try:
            # Import and use real ColBERT interface
            from iris_rag.embeddings.colbert_interface import RAGTemplatesColBERTInterface
            colbert = RAGTemplatesColBERTInterface(
                token_dimension=token_dimension
            )
            
            def real_colbert_query_encode(text: str) -> List[List[float]]:
                return colbert.encode_query(text)
            
            return real_colbert_query_encode
            
        except Exception as e:
            logger.warning(f"Failed to initialize real ColBERT query encoder: {e}. Falling back to mock.")
    
    # Fallback to mock implementation
    logger.info(f"Using mock ColBERT query encoder: {colbert_model}")
    
    # Get ColBERT token embedding dimension from config or fallback
    try:
        from iris_rag.storage.schema_manager import SchemaManager
        from common.iris_connection_manager import get_iris_connection
        from iris_rag.config.manager import ConfigurationManager
        
        config_manager = ConfigurationManager('config/config.yaml')
        connection_manager = type('CM', (), {'get_connection': get_iris_connection})()
        schema_manager = SchemaManager(connection_manager, config_manager)
        colbert_token_dimension = schema_manager.get_vector_dimension("DocumentTokenEmbeddings")
        logger.info(f"Using ColBERT token dimension from schema manager: {colbert_token_dimension}D")
    except Exception as e:
        # For mock/stub encoders, use fallback dimension instead of hard failing
        logger.warning(f"Cannot get ColBERT token dimension from schema manager: {e}")
        colbert_token_dimension = token_dimension  # Use the config value as fallback
        logger.info(f"Using fallback ColBERT token dimension: {colbert_token_dimension}D")
    
    logger.info(f"Using mock ColBERT query encoder: {model_name} with {colbert_token_dimension}D embeddings")

    def mock_colbert_query_encode(text: str) -> List[List[float]]:
        tokens = text.split()[:32]  # Limit to first 32 query tokens
        if not tokens:
            return []
        
        import numpy as np
        import hashlib
        
        query_embeddings = []
        # ColBERT token embeddings dimension determined by schema manager
        embedding_dimension = colbert_token_dimension
        
        for i, token_str in enumerate(tokens):
            # Create semantically meaningful embeddings based on token content
            # Use hash of token for deterministic but varied embeddings
            token_hash = int(hashlib.md5(token_str.encode()).hexdigest()[:8], 16)
            np.random.seed(token_hash % 10000)  # Deterministic but varied seed
            
            # Generate diverse embedding with semantic variation
            base_embedding = np.random.randn(embedding_dimension)
            
            # Add position-based variation to distinguish token positions
            position_factor = (i + 1) / len(tokens)  # 0.x to 1.0
            base_embedding += np.random.randn(embedding_dimension) * position_factor * 0.3
            
            # Add token length influence for semantic diversity
            length_factor = min(len(token_str) / 10.0, 1.0)  # 0.0 to 1.0
            base_embedding += np.random.randn(embedding_dimension) * length_factor * 0.2
            
            # Normalize to unit vector (important for cosine similarity)
            norm = np.linalg.norm(base_embedding)
            if norm > 0:
                base_embedding = base_embedding / norm
            
            # Ensure embeddings are diverse (avoid near-identical values)
            mock_embedding = base_embedding.tolist()
            query_embeddings.append(mock_embedding)
            
            # Log first few tokens for debugging
            if i < 3:
                logger.debug(f"ColBERT Mock: Token '{token_str}' -> embedding range [{min(mock_embedding):.4f}, {max(mock_embedding):.4f}]")
        
        logger.info(f"ColBERT Mock: Generated {len(query_embeddings)} diverse token embeddings")
        return query_embeddings

    return mock_colbert_query_encode


def get_colbert_doc_encoder_func_adapted(model_name: str = "stub_colbert_doc_encoder") -> Callable[[str], List[List[float]]]:
    """
    Returns an adapted ColBERT document encoder function that matches the OptimizedColBERT pipeline interface.
    Takes a text string and returns just the token embeddings (without token text).
    Expected output: List[List[float]] -> [token_embedding_vector, ...]
    """
    logger.info(f"Using adapted mock ColBERT document encoder: {model_name}")
    
    # Get the original encoder that returns tuples
    original_encoder = get_colbert_doc_encoder_func(model_name)
    
    def adapted_colbert_doc_encode(text: str) -> List[List[float]]:
        # Get the original output with token text and embeddings
        token_data = original_encoder(text)
        # Extract just the embeddings (second element of each tuple)
        embeddings_only = [embedding for token_text, embedding in token_data]
        return embeddings_only
    
    return adapted_colbert_doc_encode


def get_iris_connector(db_url: Optional[str] = None):
    if db_url is None:
        db_url = os.getenv("IRIS_CONNECTION_URL")
        if not db_url:
            raise ValueError("IRIS_CONNECTION_URL environment variable not set and db_url not provided.")
            
    print(f"Connecting to IRIS at: {db_url}")
    try:
        import sqlalchemy
        engine = sqlalchemy.create_engine(db_url)
        connection = engine.connect()
        return connection
    except Exception as e:
        print(f"Failed to connect to IRIS: {e}")
        raise

def timing_decorator(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Function {func.__name__} executed in {end_time - start_time:.4f} seconds")
        if isinstance(result, dict) and 'latency_ms' not in result:
            result['latency_ms'] = (end_time - start_time) * 1000
        return result
    return wrapper

# ... (Embedded Python specific utilities can remain as they are) ...

_iris_connector_embedded = None
_embedding_model_embedded = None
_llm_embedded = None 

def get_iris_connector_for_embedded():
    global _iris_connector_embedded
    if _iris_connector_embedded is None:
        try:
            try:
                import iris
            except ImportError:
                raise ImportError("IRIS Embedded Python module 'iris' not found. Ensure it is installed in your environment.")
            _iris_connector_embedded = iris.connect()
            print("IRIS Embedded Python: DBAPI connection established.")
        except ImportError:
            print("IRIS Embedded Python: 'iris' module not found.")
            _iris_connector_embedded = None 
        except Exception as e:
            print(f"IRIS Embedded Python: Error connecting to DB: {e}")
            _iris_connector_embedded = None
    return _iris_connector_embedded

def get_embedding_func_for_embedded(model_name_override: Optional[str] = None):
    global _embedding_model_embedded
    if _embedding_model_embedded is None:
        effective_model_name = model_name_override or get_config_value("embedding_model.name", DEFAULT_EMBEDDING_MODEL_NAME)
        dimension = get_config_value("embedding_model.dimension", DEFAULT_EMBEDDING_DIMENSION)
        print(f"IRIS Embedded Python: Loading embedding model {effective_model_name} (dim: {dimension})")
        # This would call build_hf_embedder or similar for embedded context
        _embedding_model_embedded = lambda texts: [[0.1] * dimension for _ in texts]
    return _embedding_model_embedded


def get_llm_func_for_embedded(provider: str = "stub", model_name: str = "stub-model"):
    global _llm_embedded
    if _llm_embedded is None: 
        print(f"IRIS Embedded Python: Initializing LLM {provider} - {model_name}")
        if provider == "stub":
            _llm_embedded = lambda prompt: f"Embedded Stub LLM: {prompt[:30]}"
        else:
            _llm_embedded = lambda prompt: "Error: LLM not configured for embedded"
    return _llm_embedded

def get_colbert_query_encoder():
    """
    Get ColBERT query encoder function.
    For now, this returns the same embedding function as a placeholder.
    In a full implementation, this would use a specialized ColBERT model.
    """
    # For now, use the same embedding function as a placeholder
    # In a real implementation, this would use a ColBERT-specific model
    embedding_func = get_embedding_func()
    
    def colbert_query_encoder(query_text: str):
        """
        Encode a query using ColBERT-style encoding.
        For now, this is a placeholder that uses standard embeddings.
        """
        # In a real ColBERT implementation, this would:
        # 1. Tokenize the query
        # 2. Generate token-level embeddings
        # 3. Return a matrix of embeddings
        
        # Placeholder: return standard embedding
        return embedding_func(query_text)
    
    return colbert_query_encoder


if __name__ == '__main__':
    print("Testing common.utils...")
    doc = Document(id="test_doc_001", content="This is a test document.")
    print(f"Created Document: {doc}")

    try:
        embed_func = get_embedding_func()
        config_model_name = get_config_value("embedding_model.name", DEFAULT_EMBEDDING_MODEL_NAME)
        config_dimension = get_config_value("embedding_model.dimension", DEFAULT_EMBEDDING_DIMENSION)
        print(f"Testing with embedding model: {config_model_name}, dimension: {config_dimension}")
        sample_texts = ["Hello world", "This is a test."]
        embeddings = embed_func(sample_texts)
        print(f"Embeddings generated for {len(embeddings)} texts. First embedding dim: {len(embeddings[0]) if embeddings else 'N/A'}")
        assert len(embeddings) == 2
        if embeddings and embeddings[0] is not None: # Check if embedding was successful
            assert len(embeddings[0]) == config_dimension
    except ImportError as e:
        print(f"Skipping embedding test: {e}")
    except Exception as e:
        print(f"Error during embedding test: {e}")

    try:
        llm_stub_func = get_llm_func(provider="stub")
        response_stub = llm_stub_func("Test prompt for stub LLM")
        print(f"Stub LLM Response: {response_stub}")
        assert "Stub LLM response" in response_stub
    except Exception as e:
        print(f"Error during stub LLM test: {e}")

    if os.getenv("OPENAI_API_KEY"):
        try:
            llm_openai_func = get_llm_func(provider="openai", model_name="gpt-3.5-turbo")
            response_openai = llm_openai_func("What is 1+1?") 
            print(f"OpenAI LLM Response: {response_openai}")
            assert response_openai is not None
        except ImportError as e: print(f"Skipping OpenAI LLM test: {e}")
        except ValueError as e: print(f"Skipping OpenAI LLM test due to config error: {e}")
        except Exception as e: print(f"Error during OpenAI LLM test: {e}")
    else:
        print("Skipping OpenAI LLM test: OPENAI_API_KEY not set.")

    @timing_decorator
    def example_timed_function(duration):
        time.sleep(duration)
        return {"status": "complete", "slept_for": duration}
    timed_result = example_timed_function(0.1)
    print(f"Timed function result: {timed_result}")
    assert "latency_ms" in timed_result
    assert timed_result['latency_ms'] > 0
        
    print("common.utils tests finished.")

def _setup_caching_if_needed(enable_cache: Optional[bool] = None) -> bool:
    """
    Setup LLM caching if needed and return whether caching is enabled.
    
    Args:
        enable_cache: Override cache enable/disable (None = use config default)
    
    Returns:
        True if caching is enabled, False otherwise
    """
    try:
        from common.llm_cache_manager import setup_langchain_cache, is_langchain_cache_configured
        from common.llm_cache_config import load_cache_config
        
        # Load configuration
        config = load_cache_config()
        
        # Determine if caching should be enabled
        cache_enabled = enable_cache if enable_cache is not None else config.enabled
        
        # Setup cache if enabled and not already configured
        if cache_enabled and not is_langchain_cache_configured():
            setup_langchain_cache(config)
            logger.info("LLM caching initialized")
        
        return cache_enabled
        
    except Exception as e:
        logger.warning(f"Failed to setup LLM caching: {e}")
        return False
