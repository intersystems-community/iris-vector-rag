# common/utils.py

import time
from dataclasses import dataclass, field
from typing import List, Callable, Any, Optional, Dict, Tuple # Added Tuple
# import sqlalchemy  # Removed - not needed for IRIS
import os
import logging # Added for logger usage in get_llm_func
import numpy as np

logger = logging.getLogger(__name__) # Added for logger usage

# --- Dataclasses ---
@dataclass
class Document:
    id: str
    content: str
    score: Optional[float] = None # For similarity score from retrieval
    embedding: Optional[List[float]] = field(default=None, repr=False) # Standard document embedding
    # For ColBERT or other token-level models
    colbert_tokens: Optional[List[str]] = field(default=None, repr=False)
    colbert_token_embeddings: Optional[List[List[float]]] = field(default=None, repr=False) # Raw token embeddings
    colbert_compressed_embeddings: Optional[List[Any]] = field(default=None, repr=False) # Compressed + scale factors

    def to_dict(self, include_embeddings: bool = False) -> Dict[str, Any]:
        data = {
            "id": self.id,
            "content": self.content,
            "score": self.score,
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

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dims to match database

# New pure HuggingFace embedder
_hf_embedder_cache = {}

def build_hf_embedder(model_name: str = DEFAULT_EMBEDDING_MODEL):
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
        print(f"Initializing HF embedder for model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval() # Set to evaluation mode
        # Consider model.to(device) if GPU is available/desired
        _hf_embedder_cache[model_name] = (tokenizer, model)
    else:
        print(f"Using cached HF embedder for model: {model_name}")
    
    tokenizer, model = _hf_embedder_cache[model_name]

    @functools.lru_cache(maxsize=128) # Cache individual text embeddings
    def _embed_single_text(text: str) -> List[float]:
        with torch.no_grad():
            # Validate input text
            if not text or not text.strip():
                logger.warning("Empty or whitespace-only text provided for embedding")
                return [0.0] * 768  # Return zero vector for e5-base-v2 dimensions
            
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
                return [0.0] * 768  # Return zero vector on error

    def embedding_func_hf(texts: List[str]) -> List[List[float]]:
        return [_embed_single_text(t) for t in texts]

    return embedding_func_hf


def get_embedding_func(model_name: str = DEFAULT_EMBEDDING_MODEL, provider: Optional[str] = None, mock: bool = False) -> Callable[[List[str]], List[List[float]]]:
    """
    Returns a function that takes a list of texts and returns a list of embeddings.
    Defaults to using build_hf_embedder for real models.
    Supports a "stub" provider or mock=True for testing without real models.
    """
    if mock or provider == "stub" or model_name == "stub":
        logger.info("Using stub embedding function.")
        # The e5-base-v2 model (new default) has 768 dimensions. Stub should match.
        def stub_embed_texts(texts: List[str]) -> List[List[float]]:
            return [[(len(text) % 100) * 0.01] * 768 for text in texts]
        return stub_embed_texts
    
    logger.info(f"Using pure HuggingFace embedder for model: {model_name}")
    return build_hf_embedder(model_name)


def get_llm_func(provider: str = "openai", model_name: str = "gpt-3.5-turbo", **kwargs) -> Callable[[str], str]:
    """
    Returns a function that takes a prompt string and returns an LLM completion string.
    Supports 'openai' or a 'stub' for testing.
    """
    global _llm_instance, _current_llm_key
    
    llm_key = f"{provider}_{model_name}"

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
def get_colbert_doc_encoder_func(model_name: str = "stub_colbert_doc_encoder") -> Callable[[str], List[Tuple[str, List[float]]]]:
    """
    Returns a mock ColBERT document encoder function.
    Takes a text string, "tokenizes" it, and returns mock token embeddings.
    """
    logger.info(f"Using mock ColBERT document encoder: {model_name}")

    def mock_colbert_doc_encode(text: str) -> List[Tuple[str, List[float]]]:
        tokens = text.split()[:100] # Limit to first 100 mock tokens
        if not tokens:
            return []
        
        token_embeddings_data = []
        for i, token_str in enumerate(tokens):
            # Create a simple mock embedding based on token index and length
            mock_embedding = [( (i % 10) + len(token_str) % 10 ) * 0.01] * 128 # 128-dim
            token_embeddings_data.append((token_str, mock_embedding))
        return token_embeddings_data

    return mock_colbert_doc_encode


def get_colbert_query_encoder_func(model_name: str = "stub_colbert_query_encoder") -> Callable[[str], List[List[float]]]:
    """
    Returns a mock ColBERT query encoder function.
    Takes a text string and returns mock query token embeddings.
    Expected output: List[List[float]] -> [token_embedding_vector, ...]
    """
    logger.info(f"Using mock ColBERT query encoder: {model_name}")

    def mock_colbert_query_encode(text: str) -> List[List[float]]:
        tokens = text.split()[:32]  # Limit to first 32 query tokens
        if not tokens:
            return []
        
        query_embeddings = []
        for i, token_str in enumerate(tokens):
            # Create a simple mock embedding based on token index and length
            mock_embedding = [((i % 10) + len(token_str) % 10) * 0.01] * 128  # 128-dim
            query_embeddings.append(mock_embedding)
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
            import intersystems_iris 
            _iris_connector_embedded = intersystems_iris.dbapi.connect() 
            print("IRIS Embedded Python: DBAPI connection established.")
        except ImportError:
            print("IRIS Embedded Python: 'intersystems_iris' module not found.")
            _iris_connector_embedded = None 
        except Exception as e:
            print(f"IRIS Embedded Python: Error connecting to DB: {e}")
            _iris_connector_embedded = None
    return _iris_connector_embedded

def get_embedding_func_for_embedded(model_name: str = DEFAULT_EMBEDDING_MODEL): # Use new default
    global _embedding_model_embedded
    if _embedding_model_embedded is None: 
        print(f"IRIS Embedded Python: Loading embedding model {model_name}")
        # This would call build_hf_embedder or similar for embedded context
        _embedding_model_embedded = lambda texts: [[0.1] * 768 for _ in texts] # Match new default dim
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


if __name__ == '__main__':
    print("Testing common.utils...")
    doc = Document(id="test_doc_001", content="This is a test document.")
    print(f"Created Document: {doc}")

    try:
        embed_func = get_embedding_func() # Uses new default "intfloat/e5-base-v2"
        sample_texts = ["Hello world", "This is a test."]
        embeddings = embed_func(sample_texts)
        print(f"Embeddings generated for {len(embeddings)} texts. First embedding dim: {len(embeddings[0])}")
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 768 # e5-base-v2 has 768 dimensions
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
