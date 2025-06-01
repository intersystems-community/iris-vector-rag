"""
Comprehensive tests for the enhanced chunking system integration with RAG techniques.

This test suite validates:
1. Enhanced chunking strategies work correctly
2. Integration with all 7 RAG techniques
3. Performance at scale with 1000+ documents
4. Quality metrics and biomedical optimization
5. Database storage and retrieval
"""

import pytest
import sys
import os
import json
import time
import statistics
from typing import List, Dict, Any

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from chunking.enhanced_chunking_service import ( # Path remains correct
    EnhancedDocumentChunkingService,
    TokenEstimator,
    BiomedicalSemanticAnalyzer,
    RecursiveChunkingStrategy,
    SemanticChunkingStrategy,
    AdaptiveChunkingStrategy,
    HybridChunkingStrategy,
    ChunkingQuality
)
from src.common.iris_connector import get_iris_connection # Updated import
from src.common.embedding_utils import get_embedding_model # Updated import

# Import all RAG techniques for integration testing
from src.deprecated.basic_rag.pipeline import run_basic_rag # Updated import
from src.experimental.hyde.pipeline import run_hyde_rag # Updated import
from src.experimental.crag.pipeline import run_crag # Updated import
from src.working.colbert.pipeline import run_colbert_rag # Updated import
from src.experimental.noderag.pipeline import run_noderag # Updated import
from src.experimental.graphrag.pipeline import run_graphrag # Updated import
from src.experimental.hybrid_ifind_rag.pipeline import run_hybrid_ifind_rag # Updated import

class TestEnhancedChunkingCore:
    """Test core enhanced chunking functionality."""
    
    @pytest.fixture
    def chunking_service(self):
        """Create enhanced chunking service for testing."""
        embedding_model = get_embedding_model(mock=True)
        # Create a function wrapper for the model
        def embedding_func(texts):
            return embedding_model.embed_documents(texts)
        return EnhancedDocumentChunkingService(embedding_func=embedding_func)
    
    @pytest.fixture
    def biomedical_sample_text(self):
        """Sample biomedical text for testing."""
        return """
        Diabetes mellitus is a group of metabolic disorders characterized by high blood sugar levels over a prolonged period.
        Symptoms often include frequent urination, increased thirst, and increased appetite. If left untreated, diabetes can cause many health complications.
        
        Type 1 diabetes results from the pancreas's failure to produce enough insulin due to loss of beta cells.
        This form was previously referred to as "insulin-dependent diabetes mellitus" (IDDM) or "juvenile diabetes".
        The cause is unknown. Type 2 diabetes begins with insulin resistance, a condition in which cells fail to respond to insulin properly.
        
        As the disease progresses, a lack of insulin may also develop (Fig. 1). This form was previously referred to as "non insulin-dependent diabetes mellitus" (NIDDM) or "adult-onset diabetes".
        The most common cause is a combination of excessive body weight and insufficient exercise.
        
        Gestational diabetes is the third main form, and occurs when pregnant women without a previous history of diabetes develop high blood sugar levels.
        Treatment may include dietary changes, blood glucose monitoring, and in some cases, insulin may be required.
        
        Several studies have shown that metformin vs. placebo significantly reduces the risk of developing type 2 diabetes (p < 0.001).
        The UKPDS study demonstrated that intensive glucose control reduces microvascular complications by 25% (95% CI: 7-40%).
        """
    
    def test_token_estimator_accuracy(self):
        """Test token estimation accuracy with biomedical text."""
        estimator = TokenEstimator()
        
        # Test cases with known token counts (approximate)
        test_cases = [
            ("Simple sentence.", 3),
            ("Diabetes mellitus is a metabolic disorder.", 8),
            ("The p-value was < 0.05 indicating statistical significance.", 12),
            ("Fig. 1 shows the correlation between HbA1c and glucose levels.", 13),
            ("Patients received metformin 500mg twice daily vs. placebo.", 11)
        ]
        
        for text, expected_tokens in test_cases:
            estimated = estimator.estimate_tokens(text)
            # Allow 20% variance for estimation
            assert abs(estimated - expected_tokens) <= max(2, expected_tokens * 0.2), \
                f"Token estimation for '{text}': expected ~{expected_tokens}, got {estimated}"
    
    def test_biomedical_semantic_analyzer(self):
        """Test biomedical semantic analysis capabilities."""
        analyzer = BiomedicalSemanticAnalyzer()
        
        # Test boundary strength analysis
        current_sent = "The study included 100 patients with type 2 diabetes."
        next_sent_weak = "All patients were between 18 and 65 years old."
        next_sent_strong = "However, the control group showed different characteristics."
        
        weak_boundary = analyzer.analyze_boundary_strength(current_sent, next_sent_weak)
        strong_boundary = analyzer.analyze_boundary_strength(current_sent, next_sent_strong)
        
        assert strong_boundary > weak_boundary, "Strong boundary should have higher score than weak boundary"
        assert 0 <= weak_boundary <= 1, "Boundary strength should be between 0 and 1"
        assert 0 <= strong_boundary <= 1, "Boundary strength should be between 0 and 1"
    
    def test_recursive_chunking_strategy(self, biomedical_sample_text):
        """Test recursive chunking strategy."""
        strategy = RecursiveChunkingStrategy(chunk_size=200, chunk_overlap=20)
        chunks = strategy.chunk(biomedical_sample_text, "test_doc")
        
        assert len(chunks) > 1, "Should create multiple chunks for long text"
        
        for chunk in chunks:
            assert chunk.metrics.token_count <= 250, f"Chunk exceeds token limit: {chunk.metrics.token_count}"
            assert len(chunk.text.strip()) > 0, "Chunk should not be empty"
            assert chunk.chunk_type == "recursive", "Chunk type should be recursive"
            assert chunk.strategy_name == "recursive", "Strategy name should be recursive"
    
    def test_semantic_chunking_strategy(self, biomedical_sample_text):
        """Test semantic chunking strategy."""
        strategy = SemanticChunkingStrategy(target_chunk_size=300, boundary_threshold=0.5)
        chunks = strategy.chunk(biomedical_sample_text, "test_doc")
        
        assert len(chunks) > 0, "Should create at least one chunk"
        
        for chunk in chunks:
            assert chunk.metrics.semantic_coherence_score >= 0, "Coherence score should be non-negative"
            assert chunk.chunk_type == "semantic", "Chunk type should be semantic"
            assert chunk.strategy_name == "semantic", "Strategy name should be semantic"
            assert "sentence_boundaries" in chunk.metadata, "Should include sentence boundary metadata"
    
    def test_adaptive_chunking_strategy(self, biomedical_sample_text):
        """Test adaptive chunking strategy."""
        strategy = AdaptiveChunkingStrategy()
        chunks = strategy.chunk(biomedical_sample_text, "test_doc")
        
        assert len(chunks) > 0, "Should create at least one chunk"
        
        for chunk in chunks:
            assert chunk.chunk_type == "adaptive", "Chunk type should be adaptive"
            assert chunk.strategy_name == "adaptive", "Strategy name should be adaptive"
            assert "selected_strategy" in chunk.metadata, "Should include selected strategy metadata"
            assert "document_analysis" in chunk.metadata, "Should include document analysis metadata"
    
    def test_hybrid_chunking_strategy(self, biomedical_sample_text):
        """Test hybrid chunking strategy."""
        strategy = HybridChunkingStrategy(primary_strategy="semantic", fallback_strategy="recursive")
        chunks = strategy.chunk(biomedical_sample_text, "test_doc")
        
        assert len(chunks) > 0, "Should create at least one chunk"
        
        for chunk in chunks:
            assert chunk.chunk_type == "hybrid", "Chunk type should be hybrid"
            assert chunk.strategy_name == "hybrid", "Strategy name should be hybrid"
            assert "primary_strategy" in chunk.metadata, "Should include primary strategy metadata"
    
    def test_chunking_service_integration(self, chunking_service, biomedical_sample_text):
        """Test enhanced chunking service integration."""
        # Test all available strategies
        strategies = ["recursive", "semantic", "adaptive", "hybrid"]
        
        for strategy in strategies:
            chunks = chunking_service.chunk_document("test_doc", biomedical_sample_text, strategy)
            
            assert len(chunks) > 0, f"Strategy {strategy} should create chunks"
            
            for chunk in chunks:
                assert "chunk_id" in chunk, "Chunk should have ID"
                assert "chunk_metadata" in chunk, "Chunk should have metadata"
                assert "embedding_str" in chunk, "Chunk should have embedding"
                
                # Validate metadata structure
                metadata = json.loads(chunk["chunk_metadata"])
                assert "chunk_metrics" in metadata, "Should include chunk metrics"
                assert "biomedical_optimized" in metadata, "Should indicate biomedical optimization"
    
    def test_chunking_effectiveness_analysis(self, chunking_service, biomedical_sample_text):
        """Test chunking effectiveness analysis."""
        analysis = chunking_service.analyze_chunking_effectiveness(
            "test_doc", biomedical_sample_text, ["recursive", "semantic", "adaptive"]
        )
        
        assert "document_info" in analysis, "Should include document info"
        assert "strategy_analysis" in analysis, "Should include strategy analysis"
        assert "recommendations" in analysis, "Should include recommendations"
        
        # Validate document info
        doc_info = analysis["document_info"]
        assert doc_info["estimated_tokens"] > 0, "Should estimate tokens"
        assert doc_info["biomedical_density"] >= 0, "Should calculate biomedical density"
        
        # Validate strategy analysis
        for strategy in ["recursive", "semantic", "adaptive"]:
            if strategy in analysis["strategy_analysis"]:
                strategy_metrics = analysis["strategy_analysis"][strategy]
                assert "chunk_count" in strategy_metrics, "Should include chunk count"
                assert "quality_score" in strategy_metrics, "Should include quality score"
                assert "processing_time_ms" in strategy_metrics, "Should include processing time"
        
        # Validate recommendations
        recommendations = analysis["recommendations"]
        assert "recommended_strategy" in recommendations, "Should recommend a strategy"
        assert "reason" in recommendations, "Should provide reason for recommendation"

class TestRAGIntegration:
    """Test integration of enhanced chunking with all RAG techniques."""
    
    @pytest.fixture
    def chunking_service(self):
        """Create enhanced chunking service for testing."""
        embedding_model = get_embedding_model(mock=True)
        # Create a function wrapper for the model
        def embedding_func(texts):
            return embedding_model.embed_documents(texts)
        return EnhancedDocumentChunkingService(embedding_func=embedding_func)
    
    @pytest.fixture
    def sample_documents(self):
        """Get sample documents from database for testing."""
        connection = get_iris_connection()
        cursor = connection.cursor()
        
        try:
            cursor.execute("""
                SELECT TOP 10 doc_id, title, text_content
                FROM RAG.SourceDocuments
                WHERE text_content IS NOT NULL
                AND LENGTH(text_content) > 500
                ORDER BY doc_id
            """)
            
            documents = cursor.fetchall()
            return documents
        finally:
            cursor.close()
            connection.close()
    
    def test_chunking_with_basic_rag(self, chunking_service, sample_documents):
        """Test enhanced chunking integration with BasicRAG."""
        if not sample_documents:
            pytest.skip("No sample documents available")
        
        doc_id, title, text_content = sample_documents[0]
        
        # Create chunks using enhanced chunking
        chunks = chunking_service.chunk_document(doc_id, text_content, "adaptive")
        assert len(chunks) > 0, "Should create chunks"
        
        # Store chunks temporarily for testing
        success = chunking_service.store_chunks(chunks)
        assert success, "Should successfully store chunks"
        
        # Test BasicRAG with chunked documents
        try:
            iris_connector = get_iris_connection()
            embedding_model = get_embedding_model(mock=True)
            def embedding_func(texts):
                return embedding_model.embed_documents(texts)
            
            result = run_basic_rag(
                query="What is the main finding of this study?",
                iris_connector=iris_connector,
                embedding_func=embedding_func,
                top_k=5
            )
            
            assert "answer" in result, "BasicRAG should return answer"
            assert "retrieved_documents" in result, "BasicRAG should return retrieved documents"
            
        except Exception as e:
            pytest.fail(f"BasicRAG integration failed: {e}")
    
    def test_chunking_with_hyde_rag(self, chunking_service, sample_documents):
        """Test enhanced chunking integration with HyDE RAG."""
        if not sample_documents:
            pytest.skip("No sample documents available")
        
        doc_id, title, text_content = sample_documents[0]
        
        # Create chunks using semantic strategy (good for HyDE)
        chunks = chunking_service.chunk_document(doc_id, text_content, "semantic")
        assert len(chunks) > 0, "Should create chunks"
        
        # Test HyDE RAG
        try:
            iris_connector = get_iris_connection()
            embedding_model = get_embedding_model(mock=True)
            def embedding_func(texts):
                return embedding_model.embed_documents(texts)
            
            result = run_hyde_rag(
                query="What are the clinical implications?",
                iris_connector=iris_connector,
                embedding_func=embedding_func,
                top_k=3
            )
            
            assert "answer" in result, "HyDE RAG should return answer"
            assert "retrieved_documents" in result, "HyDE RAG should return retrieved documents"
            
        except Exception as e:
            pytest.fail(f"HyDE RAG integration failed: {e}")
    
    def test_chunking_with_crag(self, chunking_service, sample_documents):
        """Test enhanced chunking integration with CRAG."""
        if not sample_documents:
            pytest.skip("No sample documents available")
        
        doc_id, title, text_content = sample_documents[0]
        
        # Create chunks using hybrid strategy
        chunks = chunking_service.chunk_document(doc_id, text_content, "hybrid")
        assert len(chunks) > 0, "Should create chunks"
        
        # Test CRAG
        try:
            iris_connector = get_iris_connection()
            embedding_model = get_embedding_model(mock=True)
            def embedding_func(texts):
                return embedding_model.embed_documents(texts)
            
            result = run_crag(
                query="What methodology was used?",
                iris_connector=iris_connector,
                embedding_func=embedding_func,
                top_k=5
            )
            
            assert "answer" in result, "CRAG should return answer"
            assert "retrieved_documents" in result, "CRAG should return retrieved documents"
            
        except Exception as e:
            pytest.fail(f"CRAG integration failed: {e}")
    
    def test_chunking_with_colbert(self, chunking_service, sample_documents):
        """Test enhanced chunking integration with ColBERT."""
        if not sample_documents:
            pytest.skip("No sample documents available")
        
        doc_id, title, text_content = sample_documents[0]
        
        # Create chunks using recursive strategy (good for ColBERT)
        chunks = chunking_service.chunk_document(doc_id, text_content, "recursive")
        assert len(chunks) > 0, "Should create chunks"
        
        # Test ColBERT
        try:
            iris_connector = get_iris_connection()
            embedding_model = get_embedding_model(mock=True)
            def embedding_func(texts):
                return embedding_model.embed_documents(texts)
            
            result = run_colbert_rag(
                query="What are the results?",
                iris_connector=iris_connector,
                embedding_func=embedding_func,
                top_k=5
            )
            
            assert "answer" in result, "ColBERT should return answer"
            assert "retrieved_documents" in result, "ColBERT should return retrieved documents"
            
        except Exception as e:
            pytest.fail(f"ColBERT integration failed: {e}")
    
    def test_chunking_with_noderag(self, chunking_service, sample_documents):
        """Test enhanced chunking integration with NodeRAG."""
        if not sample_documents:
            pytest.skip("No sample documents available")
        
        doc_id, title, text_content = sample_documents[0]
        
        # Create chunks using adaptive strategy
        chunks = chunking_service.chunk_document(doc_id, text_content, "adaptive")
        assert len(chunks) > 0, "Should create chunks"
        
        # Test NodeRAG
        try:
            iris_connector = get_iris_connection()
            embedding_model = get_embedding_model(mock=True)
            def embedding_func(texts):
                return embedding_model.embed_documents(texts)
            
            result = run_noderag(
                query="What are the key findings?",
                iris_connector=iris_connector,
                embedding_func=embedding_func,
                top_k=10
            )
            
            assert "answer" in result, "NodeRAG should return answer"
            assert "retrieved_documents" in result, "NodeRAG should return retrieved documents"
            
        except Exception as e:
            pytest.fail(f"NodeRAG integration failed: {e}")
    
    def test_chunking_with_graphrag(self, chunking_service, sample_documents):
        """Test enhanced chunking integration with GraphRAG."""
        if not sample_documents:
            pytest.skip("No sample documents available")
        
        doc_id, title, text_content = sample_documents[0]
        
        # Create chunks using semantic strategy (good for graph relationships)
        chunks = chunking_service.chunk_document(doc_id, text_content, "semantic")
        assert len(chunks) > 0, "Should create chunks"
        
        # Test GraphRAG
        try:
            iris_connector = get_iris_connection()
            embedding_model = get_embedding_model(mock=True)
            def embedding_func(texts):
                return embedding_model.embed_documents(texts)
            
            result = run_graphrag(
                query="What relationships exist in the data?",
                iris_connector=iris_connector,
                embedding_func=embedding_func,
                top_k=10
            )
            
            assert "answer" in result, "GraphRAG should return answer"
            assert "retrieved_documents" in result, "GraphRAG should return retrieved documents"
            
        except Exception as e:
            pytest.fail(f"GraphRAG integration failed: {e}")
    
    def test_chunking_with_hybrid_ifind(self, chunking_service, sample_documents):
        """Test enhanced chunking integration with Hybrid iFind RAG."""
        if not sample_documents:
            pytest.skip("No sample documents available")
        
        doc_id, title, text_content = sample_documents[0]
        
        # Create chunks using hybrid strategy
        chunks = chunking_service.chunk_document(doc_id, text_content, "hybrid")
        assert len(chunks) > 0, "Should create chunks"
        
        # Test Hybrid iFind RAG
        try:
            iris_connector = get_iris_connection()
            embedding_model = get_embedding_model(mock=True)
            def embedding_func(texts):
                return embedding_model.embed_documents(texts)
            
            result = run_hybrid_ifind_rag(
                query="What are the main conclusions?",
                iris_connector=iris_connector,
                embedding_func=embedding_func,
                top_k=5
            )
            
            assert "answer" in result, "Hybrid iFind RAG should return answer"
            assert "retrieved_documents" in result, "Hybrid iFind RAG should return retrieved documents"
            
        except Exception as e:
            pytest.fail(f"Hybrid iFind RAG integration failed: {e}")

class TestScalePerformance:
    """Test enhanced chunking performance at scale."""
    
    @pytest.fixture
    def chunking_service(self):
        """Create enhanced chunking service for testing."""
        embedding_model = get_embedding_model(mock=True)
        # Create a function wrapper for the model
        def embedding_func(texts):
            return embedding_model.embed_documents(texts)
        return EnhancedDocumentChunkingService(embedding_func=embedding_func)
    
    def test_chunking_1000_documents(self, chunking_service):
        """Test chunking performance with 1000+ documents."""
        # Test with different batch sizes and strategies
        strategies_to_test = ["adaptive", "recursive", "semantic"]
        
        for strategy in strategies_to_test:
            start_time = time.time()
            
            results = chunking_service.process_documents_at_scale(
                limit=1000,
                strategy_names=[strategy],
                batch_size=50
            )
            
            processing_time = time.time() - start_time
            
            # Validate results
            assert results["processed_documents"] > 0, f"Should process documents with {strategy}"
            assert results["total_chunks_created"] > 0, f"Should create chunks with {strategy}"
            
            # Performance assertions
            docs_per_second = results["performance_metrics"]["documents_per_second"]
            assert docs_per_second > 0.1, f"Processing rate too slow for {strategy}: {docs_per_second} docs/sec"
            
            # Quality assertions
            avg_coherence = results["quality_metrics"]["avg_semantic_coherence"]
            assert avg_coherence >= 0, f"Invalid coherence score for {strategy}: {avg_coherence}"
            
            print(f"\n{strategy} Strategy Performance:")
            print(f"  Documents processed: {results['processed_documents']}")
            print(f"  Chunks created: {results['total_chunks_created']}")
            print(f"  Processing rate: {docs_per_second:.2f} docs/sec")
            print(f"  Average coherence: {avg_coherence:.3f}")
            print(f"  Total time: {processing_time:.2f}s")
    
    def test_chunking_quality_metrics(self, chunking_service):
        """Test quality metrics across different document types."""
        connection = get_iris_connection()
        cursor = connection.cursor()
        
        try:
            # Get diverse document sample
            cursor.execute("""
                SELECT TOP 100 doc_id, title, text_content
                FROM RAG.SourceDocuments
                WHERE text_content IS NOT NULL
                AND LENGTH(text_content) BETWEEN 1000 AND 5000
                ORDER BY RANDOM()
            """)
            
            documents = cursor.fetchall()
            
            if not documents:
                pytest.skip("No suitable documents for quality testing")
            
            quality_results = {
                "recursive": [],
                "semantic": [],
                "adaptive": [],
                "hybrid": []
            }
            
            for doc_id, title, text_content in documents[:20]:  # Test subset for speed
                analysis = chunking_service.analyze_chunking_effectiveness(
                    doc_id, text_content, list(quality_results.keys())
                )
                
                for strategy, metrics in analysis["strategy_analysis"].items():
                    if "error" not in metrics:
                        quality_results[strategy].append({
                            "quality_score": metrics.get("quality_score", 0),
                            "coherence": metrics.get("avg_semantic_coherence", 0),
                            "biomedical_density": metrics.get("avg_biomedical_density", 0),
                            "processing_time": metrics.get("processing_time_ms", 0)
                        })
            
            # Analyze quality results
            for strategy, results in quality_results.items():
                if results:
                    avg_quality = statistics.mean([r["quality_score"] for r in results])
                    avg_coherence = statistics.mean([r["coherence"] for r in results])
                    avg_processing_time = statistics.mean([r["processing_time"] for r in results])
                    
                    print(f"\n{strategy} Quality Metrics:")
                    print(f"  Average quality score: {avg_quality:.3f}")
                    print(f"  Average coherence: {avg_coherence:.3f}")
                    print(f"  Average processing time: {avg_processing_time:.1f}ms")
                    
                    # Quality assertions
                    assert avg_quality >= 0.3, f"{strategy} quality too low: {avg_quality}"
                    assert avg_processing_time < 5000, f"{strategy} too slow: {avg_processing_time}ms"
        
        finally:
            cursor.close()
            connection.close()
    
    def test_chunking_database_storage(self, chunking_service):
        """Test database storage and retrieval of enhanced chunks."""
        connection = get_iris_connection()
        cursor = connection.cursor()
        
        try:
            # Get a test document
            cursor.execute("""
                SELECT TOP 1 doc_id, text_content
                FROM RAG.SourceDocuments
                WHERE text_content IS NOT NULL
                AND LENGTH(text_content) > 500
            """)
            
            result = cursor.fetchone()
            if not result:
                pytest.skip("No suitable document for storage testing")
            
            doc_id, text_content = result
            
            # Create chunks
            chunks = chunking_service.chunk_document(doc_id, text_content, "adaptive")
            assert len(chunks) > 0, "Should create chunks"
            
            # Store chunks
            success = chunking_service.store_chunks(chunks)
            assert success, "Should successfully store chunks"
            
            # Verify storage
            cursor.execute("""
                SELECT COUNT(*) FROM RAG.DocumentChunks
                WHERE doc_id = ?
            """, (doc_id,))
            
            stored_count = cursor.fetchone()[0]
            assert stored_count == len(chunks), f"Should store all chunks: expected {len(chunks)}, got {stored_count}"
            
            # Test retrieval with metadata
            cursor.execute("""
                SELECT chunk_id, chunk_text, chunk_metadata
                FROM RAG.DocumentChunks
                WHERE doc_id = ?
                ORDER BY chunk_index
            """, (doc_id,))
            
            stored_chunks = cursor.fetchall()
            
            for chunk_id, chunk_text, chunk_metadata in stored_chunks:
                assert len(chunk_text) > 0, "Stored chunk should not be empty"
                
                # Validate metadata
                metadata = json.loads(chunk_metadata)
                assert "chunk_metrics" in metadata, "Should store chunk metrics"
                assert "biomedical_optimized" in metadata, "Should indicate biomedical optimization"
                
                metrics = metadata["chunk_metrics"]
                assert metrics["token_count"] > 0, "Should store token count"
                assert metrics["character_count"] > 0, "Should store character count"
        
        finally:
            # Cleanup test data
            try:
                cursor.execute("DELETE FROM RAG.DocumentChunks WHERE doc_id = ?", (doc_id,))
                connection.commit()
            except:
                pass
            cursor.close()
            connection.close()

if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])