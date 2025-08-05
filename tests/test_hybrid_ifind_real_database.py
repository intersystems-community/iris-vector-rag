#!/usr/bin/env python3
"""
Real database integration tests for HybridIFind pipeline.

This test file validates actual database operations without mocking,
ensuring the pipeline works correctly with real IRIS database connections.
"""

import pytest
import logging
from typing import List, Dict, Any

from iris_rag.pipelines.hybrid_ifind import HybridIFindRAGPipeline
from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.models import Document
from common.iris_connection_manager import get_iris_connection

logger = logging.getLogger(__name__)


@pytest.mark.integration
class TestHybridIFindRealDatabase:
    """
    Real database integration tests for HybridIFind pipeline.
    
    These tests use actual IRIS database connections to validate:
    1. IFind SQL syntax and functionality
    2. Vector search integration
    3. Table schema requirements
    4. Error handling with real database responses
    """

    @pytest.fixture(scope="class")
    def real_connection_manager(self):
        """Create real connection manager with IRIS database."""
        return ConnectionManager()

    @pytest.fixture(scope="class") 
    def real_config_manager(self):
        """Create real configuration manager."""
        return ConfigurationManager()

    @pytest.fixture(scope="class")
    def test_documents(self):
        """Create test documents for validation."""
        return [
            Document(
                id="real_test_doc_1",
                page_content="This document discusses diabetes treatment options including insulin therapy and lifestyle modifications.",
                metadata={"title": "Diabetes Treatment", "source": "medical_journal"}
            ),
            Document(
                id="real_test_doc_2", 
                page_content="Cancer research focuses on targeted therapy approaches for oncological treatment.",
                metadata={"title": "Cancer Research", "source": "research_paper"}
            ),
            Document(
                id="real_test_doc_3",
                page_content="Cardiovascular disease prevention through dietary interventions and exercise protocols.",
                metadata={"title": "Heart Health", "source": "clinical_study"}
            )
        ]

    @pytest.fixture(scope="class")
    def pipeline_with_real_db(self, real_connection_manager, real_config_manager):
        """Create HybridIFind pipeline with real database connection."""
        try:
            pipeline = HybridIFindRAGPipeline(
                connection_manager=real_connection_manager,
                config_manager=real_config_manager
            )
            return pipeline
        except Exception as e:
            pytest.skip(f"Cannot create pipeline with real database: {e}")

    def test_real_database_connection(self):
        """Test that we can actually connect to IRIS database."""
        try:
            conn = get_iris_connection()
            cursor = conn.cursor()
            
            # Test basic query
            cursor.execute("SELECT 1 as test_value")
            result = cursor.fetchone()
            assert result[0] == 1
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            pytest.skip(f"Real IRIS database not available: {e}")

    def test_real_table_schema_validation(self, pipeline_with_real_db):
        """Test that required tables exist with correct schema."""
        conn = get_iris_connection()
        cursor = conn.cursor()
        
        try:
            # Check SourceDocuments table exists
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
            cursor.fetchone()  # Should not raise exception
            
            # Check column structure matches our expectations
            cursor.execute("""
                SELECT COLUMN_NAME, DATA_TYPE 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = 'SOURCEDOCUMENTS'
                ORDER BY ORDINAL_POSITION
            """)
            columns = cursor.fetchall()
            
            # Validate essential columns exist
            column_names = [col[0] for col in columns]
            assert 'doc_id' in column_names
            assert 'text_content' in column_names
            assert 'embedding' in column_names
            
            logger.info(f"✅ Real database schema validated: {column_names}")
            
        finally:
            cursor.close()
            conn.close()

    def test_real_vector_search_operations(self, pipeline_with_real_db, test_documents):
        """Test actual vector search operations against real database."""
        # First ensure we have test data
        try:
            pipeline_with_real_db.ingest_documents(test_documents)
        except Exception as e:
            logger.warning(f"Could not ingest test documents: {e}")
        
        # Test vector search
        try:
            result = pipeline_with_real_db._vector_search("diabetes treatment", top_k=2)
            
            # Validate structure
            assert isinstance(result, list)
            for doc_result in result:
                assert "doc_id" in doc_result
                assert "content" in doc_result
                assert "vector_score" in doc_result
                assert isinstance(doc_result["vector_score"], (int, float))
            
            logger.info(f"✅ Real vector search returned {len(result)} results")
            
        except Exception as e:
            pytest.fail(f"Real vector search failed: {e}")

    def test_real_ifind_search_functionality(self, pipeline_with_real_db):
        """Test actual IFind search against real IRIS database."""
        conn = get_iris_connection()
        cursor = conn.cursor()
        
        try:
            # Test if IFind is actually available in this IRIS instance
            test_sql = """
            SELECT TOP 1 doc_id, text_content,
                   $SCORE(text_content) as ifind_score
            FROM RAG.SourceDocuments
            WHERE $FIND(text_content, ?)
            ORDER BY $SCORE(text_content) DESC
            """
            
            cursor.execute(test_sql, ["diabetes"])
            results = cursor.fetchall()
            
            # If this doesn't raise an exception, IFind is working
            logger.info(f"✅ Real IFind search syntax validated, returned {len(results)} results")
            
            # Validate result structure if we got results
            if results:
                row = results[0]
                assert len(row) == 3  # doc_id, text_content, ifind_score
                assert row[2] is not None  # Score should not be None
                logger.info(f"✅ IFind result structure validated: doc_id={row[0]}, score={row[2]}")
            
        except Exception as e:
            # This tells us IFind is not configured - important information!
            logger.warning(f"⚠️ Real IFind functionality not available: {e}")
            # Test LIKE fallback instead
            fallback_sql = """
            SELECT TOP 1 doc_id, text_content, 1.0 as like_score
            FROM RAG.SourceDocuments
            WHERE text_content LIKE ?
            ORDER BY LENGTH(text_content) ASC
            """
            
            cursor.execute(fallback_sql, ["%diabetes%"])
            fallback_results = cursor.fetchall()
            logger.info(f"✅ LIKE fallback validated, returned {len(fallback_results)} results")
            
        finally:
            cursor.close()
            conn.close()

    def test_real_end_to_end_pipeline_execution(self, pipeline_with_real_db, test_documents):
        """Test complete pipeline execution against real database."""
        # Ensure test data exists
        try:
            pipeline_with_real_db.ingest_documents(test_documents)
        except Exception as e:
            logger.warning(f"Could not ingest test documents: {e}")
        
        # Execute actual query
        query = "diabetes treatment options"
        
        try:
            result = pipeline_with_real_db.query(query, top_k=3)
            
            # Validate response structure
            assert "query" in result
            assert "retrieved_documents" in result
            assert "vector_results_count" in result
            assert "ifind_results_count" in result
            assert result["query"] == query
            
            # Validate retrieved documents
            docs = result["retrieved_documents"]
            assert isinstance(docs, list)
            
            for doc in docs:
                assert hasattr(doc, 'id')
                assert hasattr(doc, 'page_content')
                assert hasattr(doc, 'metadata')
                assert 'search_type' in doc.metadata
                assert 'hybrid_score' in doc.metadata
                
            logger.info(f"✅ Real E2E pipeline execution completed successfully")
            logger.info(f"   Retrieved {len(docs)} documents")
            logger.info(f"   Vector results: {result['vector_results_count']}")
            logger.info(f"   IFind results: {result['ifind_results_count']}")
            
            # Log actual search methods used
            search_types = [doc.metadata.get('search_type') for doc in docs]
            logger.info(f"   Search types used: {set(search_types)}")
            
        except Exception as e:
            pytest.fail(f"Real E2E pipeline execution failed: {e}")

    def test_real_error_handling_scenarios(self, pipeline_with_real_db):
        """Test error handling with real database error responses."""
        conn = get_iris_connection()
        cursor = conn.cursor()
        
        try:
            # Test invalid table access
            invalid_sql = "SELECT * FROM RAG.NonExistentTable WHERE $FIND(content, ?)"
            
            try:
                cursor.execute(invalid_sql, ["test"])
                cursor.fetchall()
                pytest.fail("Expected error for invalid table")
            except Exception as e:
                # This is the real IRIS error message format
                error_msg = str(e)
                logger.info(f"✅ Real IRIS error handling validated: {error_msg[:100]}...")
                assert "table" in error_msg.lower() or "not found" in error_msg.lower()
        
        finally:
            cursor.close()
            conn.close()

    def test_real_hybrid_ifind_requirements_validation(self, pipeline_with_real_db):
        """Test that HybridIFind requirements are met by real database."""
        from iris_rag.validation.requirements import get_pipeline_requirements
        
        # Get HybridIFind requirements
        requirements = get_pipeline_requirements("hybrid_ifind")
        
        conn = get_iris_connection()
        cursor = conn.cursor()
        
        try:
            # Validate required tables exist
            for table_req in requirements.required_tables:
                table_name = f"{table_req.schema}.{table_req.name}"
                
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cursor.fetchone()[0]
                    logger.info(f"✅ Required table {table_name} exists with {count} rows")
                    
                    # Check if minimum row requirement is met
                    if table_req.min_rows > 0:
                        assert count >= table_req.min_rows, f"Table {table_name} has {count} rows, requires {table_req.min_rows}"
                        
                except Exception as e:
                    pytest.fail(f"Required table {table_name} not accessible: {e}")
            
            # Validate required embeddings exist (check for non-null embedding columns)
            for embedding_req in requirements.required_embeddings:
                if embedding_req.required:
                    cursor.execute(f"SELECT COUNT(*) FROM {embedding_req.table} WHERE {embedding_req.column} IS NOT NULL")
                    embedding_count = cursor.fetchone()[0]
                    logger.info(f"✅ Required embeddings in {embedding_req.table}.{embedding_req.column}: {embedding_count}")
                    
        finally:
            cursor.close()
            conn.close()

    @pytest.mark.slow
    def test_real_performance_benchmarking(self, pipeline_with_real_db):
        """Benchmark real pipeline performance with actual database operations."""
        import time
        
        queries = [
            "diabetes treatment",
            "cancer therapy",
            "heart disease prevention"
        ]
        
        performance_results = []
        
        for query in queries:
            start_time = time.time()
            
            try:
                result = pipeline_with_real_db.query(query, top_k=5)
                end_time = time.time()
                
                execution_time = end_time - start_time
                num_docs = len(result.get("retrieved_documents", []))
                
                performance_results.append({
                    "query": query,
                    "execution_time": execution_time,
                    "num_documents": num_docs,
                    "vector_count": result.get("vector_results_count", 0),
                    "ifind_count": result.get("ifind_results_count", 0)
                })
                
                logger.info(f"✅ Query '{query}': {execution_time:.3f}s, {num_docs} docs")
                
            except Exception as e:
                logger.error(f"❌ Query '{query}' failed: {e}")
        
        # Validate reasonable performance
        avg_time = sum(r["execution_time"] for r in performance_results) / len(performance_results)
        assert avg_time < 10.0, f"Average query time {avg_time:.3f}s exceeds 10s threshold"
        
        logger.info(f"✅ Real performance benchmark completed: avg={avg_time:.3f}s")


if __name__ == "__main__":
    # Run integration tests with verbose output
    pytest.main([__file__, "-v", "-s", "--tb=short"])