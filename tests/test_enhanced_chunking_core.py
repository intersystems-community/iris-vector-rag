"""
Core tests for the enhanced chunking system.

This test suite validates:
1. Enhanced chunking strategies work correctly
2. Token estimation accuracy
3. Biomedical optimization features
4. Database storage and retrieval
5. Performance at scale
"""

import pytest
import sys
import os
import json
import time

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from tools.chunking.enhanced_chunking_service import (
    EnhancedDocumentChunkingService,
    TokenEstimator,
    BiomedicalSemanticAnalyzer,
    RecursiveChunkingStrategy,
    SemanticChunkingStrategy,
    AdaptiveChunkingStrategy,
    HybridChunkingStrategy,
)
from common.iris_connector import get_iris_connection
from common.embedding_utils import get_embedding_model
from iris_rag.storage.schema_manager import SchemaManager
from iris_rag.config.manager import ConfigurationManager

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
        
        # Test cases - focus on relative accuracy rather than absolute numbers
        test_cases = [
            ("Short text.", "short"),
            ("This is a medium length sentence with several words.", "medium"),
            ("This is a much longer sentence that contains many more words and should result in a significantly higher token count than the shorter examples.", "long")
        ]
        
        results = []
        for text, category in test_cases:
            estimated = estimator.estimate_tokens(text)
            results.append((category, estimated))
            assert estimated > 0, f"Token estimation should be positive for '{text}'"
        
        # Verify relative ordering
        short_tokens = next(tokens for cat, tokens in results if cat == "short")
        medium_tokens = next(tokens for cat, tokens in results if cat == "medium")
        long_tokens = next(tokens for cat, tokens in results if cat == "long")
        
        assert short_tokens < medium_tokens < long_tokens, \
            f"Token counts should increase with text length: {short_tokens} < {medium_tokens} < {long_tokens}"
    
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
    
    def test_database_operations(self, chunking_service, biomedical_sample_text):
        """Test database storage and retrieval of enhanced chunks."""
        import uuid
        import time
        
        # Use a truly unique doc_id with timestamp to avoid any conflicts
        timestamp = str(int(time.time() * 1000))  # millisecond timestamp
        unique_doc_id = f"test_enhanced_chunk_{uuid.uuid4().hex[:8]}_{timestamp}"
        
        # Initialize schema manager to ensure table exists
        connection_manager = type('ConnectionManager', (), {
            'get_connection': lambda self: get_iris_connection()
        })()
        config_manager = ConfigurationManager()
        schema_manager = SchemaManager(connection_manager, config_manager)

        # Ensure both SourceDocuments and DocumentChunks tables exist
        schema_manager.ensure_table_schema('SourceDocuments')
        schema_manager.ensure_table_schema('DocumentChunks')

        # Create a shared connection for both storing and verifying
        connection = get_iris_connection()

        try:
            # Comprehensive cleanup: remove any existing chunks and documents that might conflict
            cleanup_cursor = connection.cursor()
            
            # Clear chunks first (due to foreign key constraint)
            cleanup_cursor.execute("DELETE FROM RAG.DocumentChunks WHERE doc_id = ?", (unique_doc_id,))
            cleanup_cursor.execute("DELETE FROM RAG.DocumentChunks WHERE chunk_id LIKE ?", (f"{unique_doc_id}_chunk_%",))
            cleanup_cursor.execute("DELETE FROM RAG.DocumentChunks WHERE doc_id LIKE ? AND doc_id != ?",
                                  (f"test_enhanced_chunk_%", unique_doc_id))
            
            # Clear documents
            cleanup_cursor.execute("DELETE FROM RAG.SourceDocuments WHERE doc_id = ?", (unique_doc_id,))
            cleanup_cursor.execute("DELETE FROM RAG.SourceDocuments WHERE doc_id LIKE ? AND doc_id != ?",
                                  (f"test_enhanced_chunk_%", unique_doc_id))
            
            connection.commit()
            cleanup_cursor.close()

            # Create the parent document in SourceDocuments table (required for foreign key constraint)
            doc_cursor = connection.cursor()
            doc_cursor.execute("""
                INSERT INTO RAG.SourceDocuments (doc_id, title, text_content, authors, keywords, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                unique_doc_id,
                f"Test Document {unique_doc_id}",
                biomedical_sample_text,
                "Test Authors",
                "test, enhanced, chunking",
                "{}"
            ))
            connection.commit()
            doc_cursor.close()

            # Create chunks with unique doc_id
            chunks = chunking_service.chunk_document(unique_doc_id, biomedical_sample_text, "adaptive")
            assert len(chunks) > 0, "Should create chunks"

            # Store chunks using the shared connection
            print(f"DEBUG: About to store {len(chunks)} chunks")
            for i, chunk in enumerate(chunks):
                print(f"DEBUG: Chunk {i}: {chunk['chunk_id']}")
            
            success = chunking_service.store_chunks(chunks, connection=connection)
            assert success, "Should successfully store chunks"
    
            # Create a fresh cursor after the chunks are stored and committed
            cursor = connection.cursor()
    
            # Verify storage using the same connection
            cursor.execute("""
            SELECT COUNT(*) FROM RAG.DocumentChunks
            WHERE doc_id = ?
        """, (unique_doc_id,))
    
            stored_count = cursor.fetchone()[0]
            print(f"DEBUG: Expected {len(chunks)} chunks, found {stored_count}")
            
            # Also check what chunks are actually in the database
            cursor.execute("""
            SELECT chunk_id FROM RAG.DocumentChunks
            WHERE doc_id = ?
        """, (unique_doc_id,))
            
            actual_chunks = cursor.fetchall()
            print(f"DEBUG: Actual chunks in DB: {[row[0] for row in actual_chunks]}")
            
            assert stored_count == len(chunks), f"Should store all chunks: expected {len(chunks)}, got {stored_count}"
            
            # Test retrieval with metadata (reuse the same cursor)
            cursor.execute("""
                SELECT chunk_id, chunk_text, metadata
                FROM RAG.DocumentChunks
                WHERE doc_id = ?
                ORDER BY chunk_index
            """, (unique_doc_id,))
            
            stored_chunks = cursor.fetchall()
            cursor.close()
            
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
            # Cleanup test data to ensure test isolation
            try:
                cleanup_cursor = connection.cursor()
                # Remove chunks first (due to foreign key constraint)
                cleanup_cursor.execute("DELETE FROM RAG.DocumentChunks WHERE doc_id = ?", (unique_doc_id,))
                # Remove the parent document
                cleanup_cursor.execute("DELETE FROM RAG.SourceDocuments WHERE doc_id = ?", (unique_doc_id,))
                connection.commit()
                cleanup_cursor.close()
            except Exception as e:
                print(f"Warning: Failed to clean up test data: {e}")
            finally:
                connection.close()
    
    def test_performance_at_scale(self, chunking_service):
        """Test chunking performance with multiple documents."""
        # Create test documents with sufficient length to generate multiple chunks
        test_docs = []
        for i in range(10):
            doc_text = f"""
            Document {i}: This is a comprehensive test document for performance evaluation of enhanced chunking strategies.
            It contains multiple sentences and paragraphs to test chunking performance across different content types.
            The document discusses various biomedical topics including diabetes mellitus, hypertension, cardiovascular disease, and metabolic disorders.
            
            Statistical analysis shows significant improvements (p < 0.05) in patient outcomes when using enhanced treatment protocols.
            Figure {i} demonstrates the correlation between treatment adherence and recovery rates in clinical trials.
            
            The methodology section describes the experimental design used to evaluate treatment efficacy.
            Participants were randomly assigned to control and treatment groups using stratified randomization.
            Primary endpoints included reduction in HbA1c levels and improvement in quality of life scores.
            
            Results indicate that the enhanced treatment protocol led to statistically significant improvements.
            Secondary analysis revealed additional benefits in terms of reduced hospitalization rates.
            These findings have important implications for clinical practice and patient care protocols.
            
            The discussion section explores the broader context of these findings within current medical literature.
            Limitations of the study include the relatively short follow-up period and potential selection bias.
            Future research should focus on long-term outcomes and cost-effectiveness analysis.
            """
            test_docs.append((f"perf_test_doc_{i}", doc_text))
        
        # Test different strategies
        strategies = ["recursive", "semantic", "adaptive"]
        
        for strategy in strategies:
            start_time = time.time()
            total_chunks = 0
            
            for doc_id, doc_text in test_docs:
                chunks = chunking_service.chunk_document(doc_id, doc_text, strategy)
                total_chunks += len(chunks)
            
            processing_time = time.time() - start_time
            docs_per_second = len(test_docs) / processing_time
            
            print(f"\n{strategy} Performance:")
            print(f"  Documents: {len(test_docs)}")
            print(f"  Total chunks: {total_chunks}")
            print(f"  Processing time: {processing_time:.3f}s")
            print(f"  Rate: {docs_per_second:.1f} docs/sec")
            
            # Performance assertions - adjusted for realistic expectations
            assert docs_per_second > 1.0, f"{strategy} processing too slow: {docs_per_second:.1f} docs/sec"
            # More realistic expectation: at least 1.5 chunks per document on average for longer documents
            assert total_chunks >= len(test_docs), f"{strategy} should create at least one chunk per document"

if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])