# Custom Reranker Implementation Guide

## Table of Contents

1. [Understanding Rerankers](#understanding-rerankers)
2. [Architecture Overview](#architecture-overview)
3. [Implementation Guide](#implementation-guide)
4. [Code Templates](#code-templates)
5. [Registration and Integration](#registration-and-integration)
6. [Testing Your Reranker](#testing-your-reranker)
7. [Advanced Topics](#advanced-topics)
8. [Real-World Examples](#real-world-examples)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)
11. [Implementation Checklist](#implementation-checklist)

---

## Understanding Rerankers

### What is Reranking?

Reranking is a crucial step in RAG pipelines that improves the relevance ordering of initially retrieved documents. While vector similarity search provides good initial candidates, rerankers use more sophisticated models to better assess query-document relevance.

### Why Reranking Matters

- **Improved Precision**: Cross-encoder models provide better relevance scoring than vector similarity alone
- **Domain Adaptation**: Custom rerankers can be tailored to specific domains or use cases
- **Multi-stage Architecture**: Enables efficient two-stage retrieval (fast vector search + precise reranking)
- **Performance Optimization**: Allows tuning the trade-off between speed and accuracy

### Built-in vs Custom Rerankers

**Built-in Rerankers:**
- `hf_reranker`: HuggingFace cross-encoder using `cross-encoder/ms-marco-MiniLM-L-6-v2`
- PyLate native reranking for ColBERT pipelines
- BM25 + neural hybrid approaches

**When to Implement Custom Rerankers:**
- Domain-specific requirements (medical, legal, technical)
- Performance optimization needs
- Integration with external reranking services
- Multi-modal reranking (text + metadata)
- Custom scoring algorithms

---

## Architecture Overview

### Reranker Function Interface

All rerankers in the framework follow a consistent function signature:

```python
from typing import List, Tuple
from iris_rag.core.models import Document

def my_reranker(query: str, docs: List[Document]) -> List[Tuple[Document, float]]:
    """
    Rerank documents based on query relevance.
    
    Args:
        query: The input query string
        docs: List of documents to rerank
        
    Returns:
        List of (document, score) tuples sorted by relevance
    """
    pass
```

### Pipeline Integration Flow

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Vector Search │───▶│  Initial Results │───▶│  Reranker Func │
│   (k * factor)  │    │  (candidates)    │    │  (custom logic) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌──────────────────┐           ▼
│  Final Results  │◀───│   Top-K Select   │    ┌─────────────────┐
│  (answer gen)   │    │   (best ranked)  │◀───│ Scored Results  │
└─────────────────┘    └──────────────────┘    │ [(doc, score)]  │
                                               └─────────────────┘
```

### Configuration Structure

Rerankers integrate seamlessly with the pipeline configuration system:

```yaml
# config/pipelines.yaml
pipelines:
  - name: "CustomReranking"
    module: "my_pipelines.custom_rerank"
    class: "CustomRerankingPipeline"
    enabled: true
    params:
      top_k: 5
      reranker_model: "my-custom-model"
      rerank_factor: 3
      custom_param: "value"
```

---

## Implementation Guide

### Step 1: Create Reranker Function

First, implement your custom reranker function:

```python
# my_rerankers/custom_reranker.py
import logging
from typing import List, Tuple
from iris_rag.core.models import Document

logger = logging.getLogger(__name__)

def custom_domain_reranker(query: str, docs: List[Document]) -> List[Tuple[Document, float]]:
    """
    Custom reranker optimized for domain-specific content.
    
    This example implements a hybrid scoring approach combining:
    - Keyword matching
    - Metadata relevance
    - Custom domain logic
    """
    scored_documents = []
    
    # Extract query keywords (simplified)
    query_words = set(query.lower().split())
    
    for doc in docs:
        score = 0.0
        
        # 1. Content relevance (keyword matching)
        content_words = set(doc.page_content.lower().split())
        keyword_overlap = len(query_words & content_words) / len(query_words)
        score += keyword_overlap * 0.4
        
        # 2. Metadata relevance (domain-specific)
        metadata = doc.metadata
        if metadata.get('document_type') == 'primary_source':
            score += 0.3
        if metadata.get('confidence_score', 0) > 0.8:
            score += 0.2
        
        # 3. Custom domain logic
        if 'important_keyword' in doc.page_content.lower():
            score += 0.1
            
        scored_documents.append((doc, score))
    
    # Sort by score (descending)
    scored_documents.sort(key=lambda x: x[1], reverse=True)
    
    logger.debug(f"Reranked {len(docs)} documents with custom logic")
    return scored_documents
```

### Step 2: Create Custom Pipeline

Extend the basic reranking pipeline with your custom reranker:

```python
# my_pipelines/custom_rerank.py
import logging
from typing import List, Dict, Any, Optional, Callable, Tuple
from iris_rag.pipelines.basic_rerank import BasicRAGRerankingPipeline
from iris_rag.core.models import Document
from my_rerankers.custom_reranker import custom_domain_reranker

logger = logging.getLogger(__name__)

class CustomRerankingPipeline(BasicRAGRerankingPipeline):
    """
    Custom reranking pipeline with domain-specific optimization.
    
    Extends BasicRAGRerankingPipeline to use custom reranker function
    and additional configuration options.
    """
    
    def __init__(
        self,
        connection_manager,
        config_manager,
        reranker_func: Optional[Callable[[str, List[Document]], List[Tuple[Document, float]]]] = None,
        **kwargs,
    ):
        # Use custom reranker if none provided
        if reranker_func is None:
            reranker_func = custom_domain_reranker
            
        # Initialize parent with custom reranker
        super().__init__(connection_manager, config_manager, reranker_func, **kwargs)
        
        # Additional custom configuration
        self.custom_config = self.config_manager.get("custom_reranking", {})
        self.enable_metadata_boost = self.custom_config.get("enable_metadata_boost", True)
        
        logger.info(f"Initialized CustomRerankingPipeline with metadata_boost={self.enable_metadata_boost}")
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Override to include custom pipeline information."""
        info = super().get_pipeline_info()
        info.update({
            "pipeline_type": "custom_reranking",
            "custom_features": {
                "metadata_boost": self.enable_metadata_boost,
                "domain_optimized": True
            }
        })
        return info
```

### Step 3: Error Handling and Validation

Implement robust error handling in your reranker:

```python
# my_rerankers/robust_reranker.py
import logging
from typing import List, Tuple
from iris_rag.core.models import Document

logger = logging.getLogger(__name__)

def robust_custom_reranker(query: str, docs: List[Document]) -> List[Tuple[Document, float]]:
    """
    Robust custom reranker with comprehensive error handling.
    """
    try:
        # Input validation
        if not query or not query.strip():
            logger.warning("Empty query provided to reranker")
            return [(doc, 0.5) for doc in docs]  # Default neutral scoring
            
        if not docs:
            logger.warning("No documents provided to reranker")
            return []
        
        # Your reranking logic here
        scored_documents = []
        
        for doc in docs:
            try:
                # Individual document scoring with error handling
                score = calculate_document_score(query, doc)
                scored_documents.append((doc, score))
                
            except Exception as e:
                logger.warning(f"Error scoring document {doc.id}: {e}")
                # Fallback to neutral score
                scored_documents.append((doc, 0.5))
        
        # Sort and validate results
        scored_documents.sort(key=lambda x: x[1], reverse=True)
        
        # Ensure all scores are valid floats
        validated_results = []
        for doc, score in scored_documents:
            if isinstance(score, (int, float)) and not (score != score):  # Check for NaN
                validated_results.append((doc, float(score)))
            else:
                logger.warning(f"Invalid score {score} for document {doc.id}, using 0.5")
                validated_results.append((doc, 0.5))
        
        logger.debug(f"Successfully reranked {len(validated_results)} documents")
        return validated_results
        
    except Exception as e:
        logger.error(f"Critical error in reranker: {e}")
        # Fallback: return documents with neutral scores
        return [(doc, 0.5) for doc in docs]

def calculate_document_score(query: str, doc: Document) -> float:
    """Calculate relevance score for a single document."""
    # Implement your scoring logic here
    # This is a placeholder - replace with actual logic
    return 0.5
```

---

## Code Templates

### Template 1: HuggingFace Cross-Encoder Reranker

```python
# templates/hf_cross_encoder_reranker.py
from typing import List, Tuple
from iris_rag.core.models import Document

def create_hf_cross_encoder_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """
    Factory function to create HuggingFace cross-encoder reranker.
    
    Args:
        model_name: HuggingFace model identifier
        
    Returns:
        Reranker function configured with specified model
    """
    def hf_cross_encoder_reranker(query: str, docs: List[Document]) -> List[Tuple[Document, float]]:
        # Lazy import to avoid startup overhead
        from sentence_transformers import CrossEncoder
        
        # Initialize model (consider caching in production)
        cross_encoder = CrossEncoder(model_name)
        
        # Prepare query-document pairs
        pairs = [(query, doc.page_content) for doc in docs]
        
        # Get relevance scores
        scores = cross_encoder.predict(pairs)
        
        # Combine documents with scores
        scored_docs = list(zip(docs, scores))
        
        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_docs
    
    return hf_cross_encoder_reranker
```

### Template 2: BM25 + Neural Hybrid Reranker

```python
# templates/hybrid_bm25_neural_reranker.py
from typing import List, Tuple
from iris_rag.core.models import Document

def create_hybrid_bm25_neural_reranker(
    neural_weight: float = 0.7,
    bm25_weight: float = 0.3,
    neural_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
):
    """
    Create hybrid reranker combining BM25 and neural scoring.
    
    Args:
        neural_weight: Weight for neural score (0-1)
        bm25_weight: Weight for BM25 score (0-1)
        neural_model: HuggingFace model for neural scoring
    """
    def hybrid_reranker(query: str, docs: List[Document]) -> List[Tuple[Document, float]]:
        from rank_bm25 import BM25Okapi
        from sentence_transformers import CrossEncoder
        import numpy as np
        
        # Prepare documents for BM25
        tokenized_docs = [doc.page_content.lower().split() for doc in docs]
        bm25 = BM25Okapi(tokenized_docs)
        
        # Get BM25 scores
        query_tokens = query.lower().split()
        bm25_scores = bm25.get_scores(query_tokens)
        
        # Normalize BM25 scores to 0-1 range
        if len(bm25_scores) > 0:
            bm25_min, bm25_max = min(bm25_scores), max(bm25_scores)
            if bm25_max > bm25_min:
                bm25_scores = [(score - bm25_min) / (bm25_max - bm25_min) for score in bm25_scores]
            else:
                bm25_scores = [0.5] * len(bm25_scores)
        
        # Get neural scores
        cross_encoder = CrossEncoder(neural_model)
        pairs = [(query, doc.page_content) for doc in docs]
        neural_scores = cross_encoder.predict(pairs)
        
        # Normalize neural scores to 0-1 range (sigmoid)
        neural_scores = 1 / (1 + np.exp(-neural_scores))
        
        # Combine scores
        combined_scores = []
        for i, doc in enumerate(docs):
            combined_score = (
                neural_weight * neural_scores[i] + 
                bm25_weight * bm25_scores[i]
            )
            combined_scores.append((doc, combined_score))
        
        # Sort by combined score
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        
        return combined_scores
    
    return hybrid_reranker
```

### Template 3: Domain-Specific Reranker

```python
# templates/domain_specific_reranker.py
from typing import List, Tuple, Dict, Any
from iris_rag.core.models import Document

class DomainSpecificReranker:
    """Configurable domain-specific reranker."""
    
    def __init__(self, domain_config: Dict[str, Any]):
        self.domain_config = domain_config
        self.domain_keywords = domain_config.get('keywords', [])
        self.metadata_weights = domain_config.get('metadata_weights', {})
        self.scoring_rules = domain_config.get('scoring_rules', {})
    
    def __call__(self, query: str, docs: List[Document]) -> List[Tuple[Document, float]]:
        """Apply domain-specific reranking."""
        scored_docs = []
        
        for doc in docs:
            score = self._calculate_domain_score(query, doc)
            scored_docs.append((doc, score))
        
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs
    
    def _calculate_domain_score(self, query: str, doc: Document) -> float:
        """Calculate domain-specific relevance score."""
        score = 0.0
        
        # Keyword relevance
        query_words = set(query.lower().split())
        content_words = set(doc.page_content.lower().split())
        keyword_overlap = len(query_words & content_words) / len(query_words) if query_words else 0
        score += keyword_overlap * 0.4
        
        # Domain keyword boost
        for keyword in self.domain_keywords:
            if keyword.lower() in doc.page_content.lower():
                score += 0.1
        
        # Metadata scoring
        for metadata_key, weight in self.metadata_weights.items():
            if metadata_key in doc.metadata:
                metadata_value = doc.metadata[metadata_key]
                if isinstance(metadata_value, (int, float)):
                    score += weight * metadata_value
                elif isinstance(metadata_value, str):
                    # String metadata scoring
                    if any(word in metadata_value.lower() for word in query.lower().split()):
                        score += weight
        
        # Custom scoring rules
        for rule_name, rule_config in self.scoring_rules.items():
            rule_score = self._apply_scoring_rule(query, doc, rule_config)
            score += rule_score
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _apply_scoring_rule(self, query: str, doc: Document, rule_config: Dict[str, Any]) -> float:
        """Apply a custom scoring rule."""
        rule_type = rule_config.get('type')
        
        if rule_type == 'length_bonus':
            # Bonus for documents within optimal length range
            min_length = rule_config.get('min_length', 100)
            max_length = rule_config.get('max_length', 1000)
            bonus = rule_config.get('bonus', 0.1)
            
            doc_length = len(doc.page_content)
            if min_length <= doc_length <= max_length:
                return bonus
        
        elif rule_type == 'freshness_bonus':
            # Bonus for recent documents
            import datetime
            from dateutil import parser
            
            if 'date' in doc.metadata:
                try:
                    doc_date = parser.parse(doc.metadata['date'])
                    age_days = (datetime.datetime.now() - doc_date).days
                    max_age = rule_config.get('max_age_days', 30)
                    bonus = rule_config.get('bonus', 0.1)
                    
                    if age_days <= max_age:
                        return bonus * (1 - age_days / max_age)
                except Exception:
                    pass
        
        return 0.0

# Example usage
def create_medical_reranker():
    """Create medical domain reranker."""
    medical_config = {
        'keywords': ['treatment', 'diagnosis', 'therapy', 'clinical', 'medical'],
        'metadata_weights': {
            'evidence_level': 0.2,
            'study_type_score': 0.15,
            'impact_factor': 0.1
        },
        'scoring_rules': {
            'freshness': {
                'type': 'freshness_bonus',
                'max_age_days': 365,
                'bonus': 0.15
            },
            'length': {
                'type': 'length_bonus',
                'min_length': 500,
                'max_length': 5000,
                'bonus': 0.1
            }
        }
    }
    
    return DomainSpecificReranker(medical_config)
```

---

## Registration and Integration

### Step 1: Update Pipeline Configuration

Add your custom pipeline to [`config/pipelines.yaml`](config/pipelines.yaml):

```yaml
# config/pipelines.yaml
pipelines:
  # ... existing pipelines ...
  
  - name: "CustomReranking"
    module: "my_pipelines.custom_rerank"
    class: "CustomRerankingPipeline"
    enabled: true
    params:
      top_k: 5
      reranker_model: "custom-domain-model"
      rerank_factor: 3
      
  - name: "HybridReranking"
    module: "my_pipelines.hybrid_rerank"
    class: "HybridRerankingPipeline"
    enabled: true
    params:
      top_k: 5
      neural_weight: 0.7
      bm25_weight: 0.3
      rerank_factor: 2
```

### Step 2: Configure Application Settings

Update your application configuration:

```yaml
# config/rag_config.yaml
pipelines:
  custom_reranking:
    enable_metadata_boost: true
    confidence_threshold: 0.8
    debug_scoring: false
    
  hybrid_reranking:
    neural_model: "cross-encoder/ms-marco-TinyBERT-L-2"
    use_cache: true
    cache_size: 1000
```

### Step 3: Factory Integration

The pipeline factory automatically discovers and loads your custom pipeline:

```python
# Usage example
from iris_rag.pipelines.factory import PipelineFactory
from iris_rag.pipelines.registry import PipelineRegistry
from iris_rag.config.pipeline_config_service import PipelineConfigService
from iris_rag.utils.module_loader import ModuleLoader

# Initialize framework components
config_service = PipelineConfigService()
module_loader = ModuleLoader()
framework_dependencies = {
    'connection_manager': connection_manager,
    'config_manager': config_manager,
    'llm_func': llm_function,
    'vector_store': None
}

# Create factory and registry
factory = PipelineFactory(config_service, module_loader, framework_dependencies)
registry = PipelineRegistry(factory)

# Register all pipelines (including custom ones)
registry.register_pipelines()

# Use your custom pipeline
custom_pipeline = registry.get_pipeline("CustomReranking")
result = custom_pipeline.query("What is machine learning?", top_k=5)
```

### Step 4: Module Structure

Organize your custom reranker modules:

```
my_project/
├── my_rerankers/
│   ├── __init__.py
│   ├── custom_reranker.py
│   ├── hybrid_reranker.py
│   └── domain_specific.py
├── my_pipelines/
│   ├── __init__.py
│   ├── custom_rerank.py
│   └── hybrid_rerank.py
├── config/
│   ├── pipelines.yaml
│   └── rag_config.yaml
└── tests/
    ├── test_custom_reranker.py
    └── test_custom_pipeline.py
```

---

## Testing Your Reranker

### Unit Tests for Reranker Function

```python
# tests/test_custom_reranker.py
import pytest
from iris_rag.core.models import Document
from my_rerankers.custom_reranker import custom_domain_reranker

class TestCustomReranker:
    
    def test_reranker_basic_functionality(self):
        """Test basic reranker functionality."""
        # Create test documents
        docs = [
            Document(
                page_content="Machine learning is a subset of AI",
                metadata={"document_type": "primary_source", "confidence_score": 0.9}
            ),
            Document(
                page_content="Deep learning uses neural networks",
                metadata={"document_type": "secondary_source", "confidence_score": 0.7}
            ),
            Document(
                page_content="Natural language processing important_keyword",
                metadata={"document_type": "primary_source", "confidence_score": 0.8}
            )
        ]
        
        query = "machine learning AI"
        results = custom_domain_reranker(query, docs)
        
        # Verify results structure
        assert len(results) == 3
        assert all(isinstance(item, tuple) and len(item) == 2 for item in results)
        assert all(isinstance(item[0], Document) for item in results)
        assert all(isinstance(item[1], (int, float)) for item in results)
        
        # Verify sorting (descending by score)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_reranker_empty_input(self):
        """Test reranker with empty inputs."""
        result = custom_domain_reranker("", [])
        assert result == []
        
        result = custom_domain_reranker("query", [])
        assert result == []
    
    def test_reranker_single_document(self):
        """Test reranker with single document."""
        doc = Document(page_content="Test content", metadata={})
        results = custom_domain_reranker("test query", [doc])
        
        assert len(results) == 1
        assert results[0][0] == doc
        assert isinstance(results[0][1], (int, float))
    
    def test_score_calculation_logic(self):
        """Test specific scoring logic."""
        # Document with high metadata relevance
        high_relevance_doc = Document(
            page_content="important_keyword machine learning",
            metadata={"document_type": "primary_source", "confidence_score": 0.9}
        )
        
        # Document with low metadata relevance
        low_relevance_doc = Document(
            page_content="unrelated content",
            metadata={"document_type": "secondary_source", "confidence_score": 0.5}
        )
        
        results = custom_domain_reranker("machine learning", [low_relevance_doc, high_relevance_doc])
        
        # High relevance document should be ranked first
        assert results[0][0] == high_relevance_doc
        assert results[0][1] > results[1][1]
```

### Integration Tests for Pipeline

```python
# tests/test_custom_pipeline.py
import pytest
from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager
from my_pipelines.custom_rerank import CustomRerankingPipeline

class TestCustomRerankingPipeline:
    
    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance for testing."""
        connection_manager = ConnectionManager()
        config_manager = ConfigurationManager()
        return CustomRerankingPipeline(connection_manager, config_manager)
    
    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline is not None
        assert hasattr(pipeline, 'reranker_func')
        assert hasattr(pipeline, 'custom_config')
        
        info = pipeline.get_pipeline_info()
        assert info['pipeline_type'] == 'custom_reranking'
        assert 'custom_features' in info
    
    def test_pipeline_query_execution(self, pipeline):
        """Test end-to-end query execution."""
        # This requires a test database setup
        # Implement based on your testing infrastructure
        pass
    
    def test_reranking_integration(self, pipeline, mock_documents):
        """Test reranking integration within pipeline."""
        query = "test query"
        docs = mock_documents()
        
        # Test internal reranking method
        reranked_docs = pipeline._rerank_documents(query, docs, top_k=3)
        
        assert len(reranked_docs) <= 3
        assert all(isinstance(doc, Document) for doc in reranked_docs)
```

### Performance Benchmarking

```python
# tests/test_reranker_performance.py
import time
import pytest
from iris_rag.core.models import Document
from my_rerankers.custom_reranker import custom_domain_reranker

class TestRerankerPerformance:
    
    @pytest.fixture
    def large_document_set(self):
        """Create large set of documents for performance testing."""
        docs = []
        for i in range(1000):
            docs.append(Document(
                page_content=f"Document {i} content with various keywords and text",
                metadata={"doc_id": i, "document_type": "test"}
            ))
        return docs
    
    def test_reranker_performance_1000_docs(self, large_document_set):
        """Test reranker performance with 1000 documents."""
        query = "test query performance"
        
        start_time = time.time()
        results = custom_domain_reranker(query, large_document_set)
        execution_time = time.time() - start_time
        
        # Performance assertions
        assert execution_time < 5.0  # Should complete within 5 seconds
        assert len(results) == 1000
        
        print(f"Reranked 1000 documents in {execution_time:.2f} seconds")
        print(f"Average time per document: {(execution_time/1000)*1000:.2f} ms")
    
    def test_memory_usage(self, large_document_set):
        """Test memory usage during reranking."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        query = "memory test query"
        results = custom_domain_reranker(query, large_document_set)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory should not increase excessively
        assert memory_increase < 100  # Less than 100MB increase
        
        print(f"Memory increase: {memory_increase:.2f} MB")
```

---

## Advanced Topics

### Async Reranking for Performance

```python
# advanced/async_reranking_pipeline.py
import asyncio
from typing import List, Dict, Any
from iris_rag.pipelines.basic_rerank import BasicRAGRerankingPipeline
from iris_rag.core.models import Document

class AsyncRerankingPipeline(BasicRAGRerankingPipeline):
    """Pipeline with async reranking for improved performance."""
    
    async def _async_rerank_documents(self, query_text: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """Async version of document reranking."""
        batch_size = 10
        batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
        
        # Process batches concurrently
        tasks = [self._score_batch_async(query_text, batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks)
        
        # Flatten and sort results
        all_results = []
        for batch_result in batch_results:
            all_results.extend(batch_result)
        
        all_results.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in all_results[:top_k]]
    
    async def _score_batch_async(self, query: str, docs: List[Document]):
        """Score a batch of documents asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.reranker_func, query, docs)
```

### GPU Acceleration

```python
# advanced/gpu_reranker.py
import torch
from typing import List, Tuple
from iris_rag.core.models import Document

def create_gpu_accelerated_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Create GPU-accelerated reranker."""
    
    def gpu_reranker(query: str, docs: List[Document]) -> List[Tuple[Document, float]]:
        from sentence_transformers import CrossEncoder
        
        # Check GPU availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model on GPU
        model = CrossEncoder(model_name, device=device)
        
        # Prepare inputs
        pairs = [(query, doc.page_content) for doc in docs]
        
        # Batch processing for GPU efficiency
        batch_size = 32
        all_scores = []
        
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            batch_scores = model.predict(batch_pairs)
            all_scores.extend(batch_scores)
        
        # Combine and sort
        scored_docs = list(zip(docs, all_scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_docs
    
    return gpu_reranker
```

### Caching Strategies

```python
# advanced/cached_reranker.py
import hashlib
import pickle
from typing import List, Tuple, Optional
from iris_rag.core.models import Document

class CachedReranker:
    """Reranker with intelligent caching."""
    
    def __init__(self, base_reranker, cache_size: int = 1000, ttl_seconds: int = 3600):
        self.base_reranker = base_reranker
        self.cache = {}
        self.cache_order = []
        self.cache_size = cache_size
        self.ttl_seconds = ttl_seconds
    
    def __call__(self, query: str, docs: List[Document]) -> List[Tuple[Document, float]]:
        # Create cache key
        doc_contents = [doc.page_content for doc in docs]
        cache_key = self._create_cache_key(query, doc_contents)
        
        # Check cache
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Compute result
        result = self.base_reranker(query, docs)
        
        # Store in cache
        self._store_in_cache(cache_key, result)
        
        return result
    
    def _create_cache_key(self, query: str, doc_contents: List[str]) -> str:
        """Create deterministic cache key."""
        content = query + "|".join(sorted(doc_contents))
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_from_cache(self, key: str) -> Optional[List[Tuple[Document, float]]]:
        """Retrieve from cache if valid."""
        import time
        
        if key in self.cache:
            result, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                # Move to end (LRU)
                self.cache_order.remove(key)
                self.cache_order.append(key)
                return result
            else:
                # Expired
                del self.cache[key]
                self.cache_order.remove(key)
        
        return None
    
    def _store_in_cache(self, key: str, result: List[Tuple[Document, float]]) -> None:
        """Store result in cache with LRU eviction."""
        import time
        
        # Evict if cache is full
        if len(self.cache) >= self.cache_size:
            oldest_key = self.cache_order.pop(0)
            del self.cache[oldest_key]
        
        # Store new result
        self.cache[key] = (result, time.time())
        self.cache_order.append(key)
```

---

## Real-World Examples

### Example 1: Medical Literature Reranker

```python
# examples/medical_reranker.py
from typing import List, Tuple
from iris_rag.core.models import Document

def medical_literature_reranker(query: str, docs: List[Document]) -> List[Tuple[Document, float]]:
    """
    Specialized reranker for medical literature.
    
    Considers:
    - Medical terminology matching
    - Evidence level
    - Publication type
    - Journal impact factor
    """
    medical_terms = extract_medical_terms(query)
    scored_docs = []
    
    for doc in docs:
        score = 0.0
        metadata = doc.metadata
        content = doc.page_content.lower()
        
        # Medical terminology relevance (40%)
        term_score = calculate_medical_term_overlap(medical_terms, content)
        score += 0.4 * term_score
        
        # Evidence level (30%)
        evidence_level = metadata.get('evidence_level', 'C')
        evidence_scores = {'A': 1.0, 'B': 0.7, 'C': 0.4, 'D': 0.2}
        score += 0.3 * evidence_scores.get(evidence_level, 0.2)
        
        # Publication type (20%)
        pub_type = metadata.get('publication_type', '')
        if pub_type == 'systematic_review':
            score += 0.2
        elif pub_type == 'randomized_trial':
            score += 0.18
        elif pub_type == 'cohort_study':
            score += 0.15
        elif pub_type == 'case_report':
            score += 0.05
        
        # Journal impact factor (10%)
        impact_factor = metadata.get('impact_factor', 0)
        if impact_factor > 10:
            score += 0.1
        elif impact_factor > 5:
            score += 0.07
        elif impact_factor > 2:
            score += 0.05
        
        scored_docs.append((doc, score))
    
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return scored_docs

def extract_medical_terms(query: str) -> List[str]:
    """Extract medical terms from query using medical ontology."""
    # Simplified implementation - in practice, use medical NLP libraries
    medical_keywords = [
        'disease', 'treatment', 'diagnosis', 'therapy', 'syndrome',
        'medication', 'clinical', 'patient', 'symptom', 'pathology'
    ]
    
    query_lower = query.lower()
    return [term for term in medical_keywords if term in query_lower]

def calculate_medical_term_overlap(terms: List[str], content: str) -> float:
    """Calculate overlap between medical terms and content."""
    if not terms:
        return 0.5
    
    found_terms = sum(1 for term in terms if term in content)
    return found_terms / len(terms)
```

### Example 2: Legal Document Reranker

```python
# examples/legal_reranker.py
import re
from typing import List, Tuple
from iris_rag.core.models import Document

def legal_document_reranker(query: str, docs: List[Document]) -> List[Tuple[Document, float]]:
    """
    Specialized reranker for legal documents.
    
    Considers:
    - Legal citation matching
    - Jurisdiction relevance
    - Document type hierarchy
    - Precedent value
    """
    scored_docs = []
    
    for doc in docs:
        score = 0.0
        metadata = doc.metadata
        content = doc.page_content
        
        # Legal citation relevance (35%)
        citation_score = calculate_citation_relevance(query, content)
        score += 0.35 * citation_score
        
        # Jurisdiction matching (25%)
        jurisdiction_score = calculate_jurisdiction_relevance(query, metadata)
        score += 0.25 * jurisdiction_score
        
        # Document type hierarchy (25%)
        doc_type_score = calculate_document_type_score(metadata)
        score += 0.25 * doc_type_score
        
        # Precedent value (15%)
        precedent_score = calculate_precedent_value(metadata)
        score += 0.15 * precedent_score
        
        scored_docs.append((doc, score))
    
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return scored_docs

def calculate_citation_relevance(query: str, content: str) -> float:
    """Calculate relevance based on legal citation patterns."""
    # Extract case citations from query
    query_citations = extract_legal_citations(query)
    content_citations = extract_legal_citations(content)
    
    if not query_citations:
        return 0.5  # Neutral if no citations in query
    
    # Calculate citation overlap
    common_citations = set(query_citations) & set(content_citations)
    return len(common_citations) / len(query_citations)

def extract_legal_citations(text: str) -> List[str]:
    """Extract legal citations using regex patterns."""
    patterns = [
        r'\d+\s+U\.S\.\s+\d+',  # U.S. Supreme Court
        r'\d+\s+F\.\d+d\s+\d+',  # Federal Reporter
        r'\d+\s+S\.Ct\.\s+\d+',  # Supreme Court Reporter
    ]
    
    citations = []
    for pattern in patterns:
        citations.extend(re.findall(pattern, text))
    
    return citations
```

---

## Best Practices

### Performance Optimization

1. **Lazy Loading**: Load models only when needed
2. **Batch Processing**: Process documents in batches for GPU efficiency
3. **Caching**: Cache frequently used models and results
4. **Early Stopping**: Implement score thresholds to avoid processing irrelevant documents

### Error Handling and Robustness

1. **Input Validation**: Always validate inputs
2. **Graceful Degradation**: Provide fallbacks when reranking fails
3. **Logging**: Log performance metrics and errors
4. **Resource Management**: Properly manage GPU memory and model loading

### Configuration Management

1. **Environment-specific configs**: Different settings for dev/staging/prod
2. **Parameter validation**: Validate configuration parameters
3. **Hot reloading**: Support configuration updates without restart

### Monitoring and Metrics

1. **Performance Metrics**: Track latency, throughput, memory usage
2. **Quality Metrics**: Monitor relevance scores and user feedback
3. **System Health**: Monitor GPU utilization, error rates

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: ImportError when loading custom reranker

**Problem**: `ModuleNotFoundError: No module named 'my_rerankers'`

**Solution**:
```python
# Ensure your module is in Python path
import sys
sys.path.append('/path/to/your/project')

# Or use relative imports in package structure
from .my_rerankers.custom_reranker import custom_domain_reranker
```

#### Issue 2: GPU out of memory errors

**Problem**: `CUDA out of memory` when processing large document sets

**Solution**:
```python
# Reduce batch size
def gpu_safe_reranker(query: str, docs: List[Document]) -> List[Tuple[Document, float]]:
    batch_size = 8  # Reduce from default 32
    # ... rest of implementation
    
# Or implement gradient accumulation
import torch

def memory_efficient_reranker(query: str, docs: List[Document]) -> List[Tuple[Document, float]]:
    with torch.no_grad():  # Disable gradient computation
        # Your reranking logic here
        pass
```

#### Issue 3: Slow reranking performance

**Problem**: Reranking takes too long for production use

**Solutions**:
```python
# 1. Use smaller/faster models
faster_reranker = create_hf_cross_encoder_reranker("cross-encoder/ms-marco-TinyBERT-L-2")

# 2. Implement early stopping
def fast_reranker_with_threshold(query: str, docs: List[Document], threshold: float = 0.8):
    results = []
    for doc in docs:
        score = calculate_score(query, doc)
        if score >= threshold:
            results.append((doc, score))
        elif len(results) >= 10:  # Early stopping
            break
    return results
```

#### Issue 4: Configuration not being loaded

**Problem**: Custom configuration parameters not being recognized

**Solution**:
```python
# Check configuration file path and format
def debug_config_loading():
    from iris_rag.config.manager import ConfigurationManager
    
    config_manager = ConfigurationManager()
    
    # Debug config loading
    all_config = config_manager.get_all()
    print(f"Loaded config: {all_config}")
    
    # Check specific sections
    pipeline_config = config_manager.get("pipelines:custom_reranking", {})
    print(f"Pipeline config: {pipeline_config}")
```

---

## Implementation Checklist

Use this checklist to ensure your custom reranker implementation is complete:

### Design Phase
- [ ] Define reranker requirements and use cases
- [ ] Choose appropriate scoring approach (neural, hybrid, rule-based)
- [ ] Design error handling and fallback strategies
- [ ] Plan performance optimization approach

### Implementation Phase
- [ ] Create reranker function with correct signature
- [ ] Implement robust error handling
- [ ] Add input validation and type checking
- [ ] Create custom pipeline class (if needed)
- [ ] Add configuration support

### Integration Phase
- [ ] Update pipeline configuration files
- [ ] Add module to Python path
- [ ] Test factory registration and loading
- [ ] Verify configuration loading

### Testing Phase
- [ ] Write unit tests for reranker function
- [ ] Create integration tests for pipeline
- [ ] Add performance benchmarks
- [ ] Test error handling scenarios
- [ ] Validate against baseline rerankers

### Deployment Phase
- [ ] Add monitoring and logging
- [ ] Configure production settings
- [ ] Set up performance alerts
- [ ] Document deployment procedures

### Maintenance Phase
- [ ] Monitor performance metrics
- [ ] Track quality metrics
- [ ] Plan model updates and improvements
- [ ] Maintain documentation

---

## FAQ

**Q: Can I use multiple rerankers in sequence?**
A: Yes, implement a multi-stage reranking pipeline that applies multiple rerankers sequentially, each filtering to fewer documents.

**Q: How do I handle different document types in one reranker?**
A: Use metadata-based routing within your reranker function to apply different scoring logic based on document type.

**Q: Can I use external APIs in my reranker?**
A: Yes, but implement proper timeout handling, retries, and fallbacks for network failures.

**Q: How do I tune reranker performance?**
A: Use A/B testing with relevance metrics, benchmark against baseline rerankers, and optimize based on your specific use case.

**Q: Can I cache reranker results?**
A: Yes, implement intelligent caching based on query and document content hashes, with appropriate TTL values.

---

## Additional Resources

- [HuggingFace Cross-Encoders](https://huggingface.co/cross-encoder)
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [IRIS RAG Framework Documentation](iris_rag/README.md)
- [Pipeline Configuration Guide](config/README.md)

---

**Last Updated**: November 2024  
**Version**: 1.0