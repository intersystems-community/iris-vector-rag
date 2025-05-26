"""
ColBERT Document Encoder

This module implements the token-level embedding generation for documents,
which is a critical part of the ColBERT indexing process.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import re
import logging
import json
from tqdm import tqdm

# Configure logging
logger = logging.getLogger(__name__)

class ColBERTDocEncoder:
    """
    Encoder for generating token-level embeddings for documents in ColBERT.
    
    This class wraps either a real transformer model or provides a mock
    implementation for testing.
    """
    
    def __init__(self, model_name: str = "colbert-ir/colbertv2.0", 
                 device: str = "cpu", 
                 embedding_dim: int = 128,
                 max_doc_length: int = 512,
                 mock: bool = False):
        """
        Initialize the document encoder.
        
        Args:
            model_name: HuggingFace model name to use
            device: Device to run inference on ('cpu' or 'cuda')
            embedding_dim: Dimension of token embeddings
            max_doc_length: Maximum number of tokens to encode per document
            mock: Whether to use a mock implementation for testing
        """
        self.model_name = model_name
        self.device = device
        self.embedding_dim = embedding_dim
        self.max_doc_length = max_doc_length
        self.mock = mock
        
        if not mock:
            try:
                self._initialize_real_model()
            except Exception as e:
                logger.warning(f"Failed to initialize real model: {e}")
                logger.warning("Falling back to mock implementation")
                self.mock = True
    
    def _initialize_real_model(self):
        """
        Initialize the real transformer model for document encoding.
        """
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            logger.info(f"Initializing ColBERT document encoder with {self.model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # Move model to appropriate device
            self.model = self.model.to(self.device)
            
            # Put model in evaluation mode
            self.model.eval()
            
            logger.info("ColBERT document encoder initialized successfully")
            
        except ImportError as e:
            logger.error(f"Required packages not installed: {e}")
            raise
        except Exception as e:
            logger.error(f"Error initializing document encoder: {e}")
            raise
    
    def _tokenize_document(self, document: str) -> Dict[str, Any]:
        """
        Tokenize the document using the appropriate tokenizer.
        
        Args:
            document: Document text
            
        Returns:
            Dictionary of tokenizer outputs
        """
        if self.mock:
            # Simple mock tokenization
            return self._mock_tokenize(document)
        else:
            # Real tokenization
            return self.tokenizer(
                document,
                max_length=self.max_doc_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
    
    def _mock_tokenize(self, text: str) -> Dict[str, Any]:
        """
        Mock tokenization for testing.
        
        Args:
            text: Text to tokenize
            
        Returns:
            Mock tokenizer output
        """
        # Simple whitespace and punctuation tokenization
        tokens = re.findall(r'\w+|[^\w\s]', text.lower())
        
        # Limit to max_doc_length
        tokens = tokens[:self.max_doc_length]
        
        # Create a mock tokenizer output
        mock_output = {
            "input_ids": np.array([[i for i in range(len(tokens))]]),
            "attention_mask": np.array([[1 for _ in range(len(tokens))]]),
            "tokens": tokens  # Not in real tokenizer output, but useful for mock
        }
        
        return mock_output
    
    def _encode_real(self, document: str) -> Tuple[List[str], List[List[float]]]:
        """
        Generate token-level embeddings using the real transformer model.
        
        Args:
            document: Document text
            
        Returns:
            Tuple of (list of tokens, list of token embeddings)
        """
        import torch
        
        # Tokenize the document
        inputs = self._tokenize_document(document)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        
        # Get token strings for reference
        token_ids = inputs["input_ids"][0].cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        
        # Generate embeddings with no gradients
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Get the last hidden state
            last_hidden_state = outputs.last_hidden_state
            
            # Apply attention mask if available
            if "attention_mask" in inputs:
                mask = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden_state.size()).float()
                last_hidden_state = last_hidden_state * mask
            
            # Convert to numpy and normalize
            token_embeddings = last_hidden_state.cpu().numpy()[0]
            
            # Normalize each embedding
            norms = np.linalg.norm(token_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0  # Avoid division by zero
            token_embeddings = token_embeddings / norms
            
            # Filter out padding tokens using attention mask if available
            if "attention_mask" in inputs:
                mask = inputs["attention_mask"].cpu().numpy()[0]
                tokens = [t for i, t in enumerate(tokens) if mask[i] == 1]
                token_embeddings = np.array([emb for i, emb in enumerate(token_embeddings) if mask[i] == 1])

            # Ensure embeddings are truncated to the specified dimension
            if token_embeddings.shape[1] > self.embedding_dim:
                token_embeddings = token_embeddings[:, :self.embedding_dim]
            elif token_embeddings.shape[1] < self.embedding_dim:
                logger.warning(f"Original ColBERT embedding dimension {token_embeddings.shape[1]} is less than target {self.embedding_dim}. Padding with zeros. This might affect performance.")
                padding_width = self.embedding_dim - token_embeddings.shape[1]
                token_embeddings = np.pad(token_embeddings, ((0,0), (0, padding_width)), 'constant')

            return tokens, token_embeddings.tolist()
    
    def _encode_mock(self, document: str) -> Tuple[List[str], List[List[float]]]:
        """
        Generate mock token-level embeddings for testing.
        
        Args:
            document: Document text
            
        Returns:
            Tuple of (list of tokens, list of token embeddings)
        """
        # Get the tokens
        tokenizer_output = self._tokenize_document(document)
        tokens = tokenizer_output.get("tokens", [])
        
        # Generate a mock embedding for each token
        token_embeddings = []
        for i, token in enumerate(tokens):
            # Create a deterministic but "unique" embedding based on token and position
            np.random.seed((hash(token) + i) % 10000)
            embedding = np.random.randn(self.embedding_dim)
            
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
            token_embeddings.append(embedding.tolist())
        
        return tokens, token_embeddings
    
    def encode(self, document: str) -> Tuple[List[str], List[List[float]]]:
        """
        Generate token-level embeddings for the document.
        
        Args:
            document: Document text
            
        Returns:
            Tuple of (list of tokens, list of token embeddings)
        """
        try:
            if self.mock:
                return self._encode_mock(document)
            else:
                return self._encode_real(document)
        except Exception as e:
            logger.error(f"Error encoding document: {e}")
            
            # Fall back to mock in case of error
            if not self.mock:
                logger.warning("Falling back to mock implementation")
                return self._encode_mock(document)
            
            # If already using mock, return empty list as last resort
            return [], []
    
    def __call__(self, document: str) -> List[List[float]]:
        """
        Make the encoder callable as a function.
        
        Args:
            document: Document text
            
        Returns:
            List of token embeddings (without tokens)
        """
        _, embeddings = self.encode(document)
        return embeddings


def get_colbert_doc_encoder(model_name: str = "colbert-ir/colbertv2.0", 
                          device: str = "cpu",
                          embedding_dim: int = 128,
                          mock: bool = False) -> callable:
    """
    Get a function that generates ColBERT document token embeddings.
    
    Args:
        model_name: HuggingFace model name to use
        device: Device to run inference on ('cpu' or 'cuda')
        embedding_dim: Dimension of token embeddings
        mock: Whether to use a mock implementation for testing
        
    Returns:
        Function that takes a document string and returns token embeddings
    """
    encoder = ColBERTDocEncoder(
        model_name=model_name,
        device=device,
        embedding_dim=embedding_dim,
        mock=mock
    )
    
    return encoder.encode


def generate_token_embeddings_for_documents(
    documents: List[Dict[str, Any]], 
    output_file: str = None,
    batch_size: int = 16,
    model_name: str = "colbert-ir/colbertv2.0",
    device: str = "cpu",
    mock: bool = False
) -> List[Dict[str, Any]]:
    """
    Generate token-level embeddings for a collection of documents.
    
    Args:
        documents: List of document dictionaries, each with 'id' and 'content' keys
        output_file: Optional file path to save the embeddings to
        batch_size: Number of documents to process at once
        model_name: HuggingFace model name to use
        device: Device to run inference on ('cpu' or 'cuda')
        mock: Whether to use a mock implementation for testing
        
    Returns:
        List of documents with added 'tokens' and 'token_embeddings' keys
    """
    # Create encoder
    encoder = ColBERTDocEncoder(
        model_name=model_name,
        device=device,
        mock=mock
    )
    
    logger.info(f"Generating token embeddings for {len(documents)} documents")
    
    # Process documents in batches
    enriched_documents = []
    for i in tqdm(range(0, len(documents), batch_size), desc="Processing document batches"):
        batch = documents[i:i+batch_size]
        
        for doc in batch:
            try:
                # Generate token embeddings
                tokens, token_embeddings = encoder.encode(doc["content"])
                
                # Add to document
                doc["tokens"] = tokens
                doc["token_embeddings"] = token_embeddings
                
                enriched_documents.append(doc)
            except Exception as e:
                logger.error(f"Error processing document {doc.get('id', 'unknown')}: {e}")
    
    # Save to file if requested
    if output_file:
        try:
            with open(output_file, "w") as f:
                json.dump(enriched_documents, f)
            logger.info(f"Saved embeddings to {output_file}")
        except Exception as e:
            logger.error(f"Error saving embeddings to {output_file}: {e}")
    
    return enriched_documents


if __name__ == "__main__":
    # Example usage:
    logging.basicConfig(level=logging.INFO)
    
    # Sample documents
    sample_docs = [
        {"id": "doc1", "content": "This is a sample document about AI technology."},
        {"id": "doc2", "content": "ColBERT uses token-level embeddings for better retrieval."}
    ]
    
    # Generate token embeddings with mock encoder
    enriched_docs = generate_token_embeddings_for_documents(
        sample_docs,
        output_file="sample_token_embeddings.json",
        mock=True
    )
    
    # Print stats
    for doc in enriched_docs:
        print(f"Document {doc['id']}:")
        print(f"  Tokens: {len(doc['tokens'])}")
        print(f"  Token embeddings: {len(doc['token_embeddings'])}x{len(doc['token_embeddings'][0]) if doc['token_embeddings'] else 0}")
