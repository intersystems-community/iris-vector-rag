#!/usr/bin/env python3
"""
Emergency debug script for ColBERTRAG and NodeRAG retrieval failures.
This script traces the exact failure points in both pipelines.
"""

import logging
import sys
import os
import traceback
from typing import List, Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('debug_colbert_noderag.log')
    ]
)

logger = logging.getLogger(__name__)

def debug_colbert_pipeline():
    """Debug ColBERTRAG pipeline step by step."""
    logger.info("=" * 80)
    logger.info("DEBUGGING COLBERTRAG PIPELINE")
    logger.info("=" * 80)
    
    try:
        # Import required modules
        from iris_rag.core.connection import ConnectionManager
        from iris_rag.config.manager import ConfigurationManager
        from iris_rag.pipelines.colbert import ColBERTRAGPipeline
        from common.utils import get_colbert_query_encoder_func, get_llm_func
        
        # Initialize components
        config_manager = ConfigurationManager()
        connection_manager = ConnectionManager(config_manager)
        
        # Test query
        test_query = "What are the effects of metformin on type 2 diabetes?"
        logger.info(f"Testing query: {test_query}")
        
        # Initialize pipeline
        pipeline = ColBERTRAGPipeline(
            connection_manager=connection_manager,
            config_manager=config_manager
        )
        
        # Step 1: Test query encoder
        logger.info("Step 1: Testing ColBERT query encoder...")
        try:
            query_token_embeddings = pipeline.colbert_query_encoder(test_query)
            logger.info(f"Query encoder SUCCESS: Generated {len(query_token_embeddings)} token embeddings")
            logger.info(f"First token embedding shape: {len(query_token_embeddings[0]) if query_token_embeddings else 'None'}")
        except Exception as e:
            logger.error(f"Query encoder FAILED: {e}")
            logger.error(traceback.format_exc())
            return
        
        # Step 2: Test database connection and table existence
        logger.info("Step 2: Testing database connection and tables...")
        connection = connection_manager.get_connection()
        cursor = connection.cursor()
        
        try:
            # Check SourceDocuments table
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
            source_docs_count = cursor.fetchone()[0]
            logger.info(f"SourceDocuments count: {source_docs_count}")
            
            # Check documents with embeddings
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NOT NULL")
            docs_with_embeddings = cursor.fetchone()[0]
            logger.info(f"Documents with embeddings: {docs_with_embeddings}")
            
            # Check DocumentTokenEmbeddings table
            try:
                cursor.execute("SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings")
                token_embeddings_count = cursor.fetchone()[0]
                logger.info(f"DocumentTokenEmbeddings count: {token_embeddings_count}")
            except Exception as e:
                logger.error(f"DocumentTokenEmbeddings table issue: {e}")
                token_embeddings_count = 0
                
        except Exception as e:
            logger.error(f"Database check FAILED: {e}")
            return
        finally:
            cursor.close()
        
        # Step 3: Test HNSW candidate retrieval
        logger.info("Step 3: Testing HNSW candidate retrieval...")
        try:
            candidate_doc_ids = pipeline._retrieve_candidate_documents_hnsw(test_query, k=10)
            logger.info(f"HNSW retrieval: Found {len(candidate_doc_ids)} candidates")
            if candidate_doc_ids:
                logger.info(f"First 5 candidate IDs: {candidate_doc_ids[:5]}")
            else:
                logger.error("CRITICAL: No candidate documents found via HNSW!")
                
                # Debug HNSW search in detail
                debug_hnsw_search(pipeline, test_query)
                
        except Exception as e:
            logger.error(f"HNSW retrieval FAILED: {e}")
            logger.error(traceback.format_exc())
            return
        
        # Step 4: Test token embeddings loading (if candidates found)
        if candidate_doc_ids and token_embeddings_count > 0:
            logger.info("Step 4: Testing token embeddings loading...")
            try:
                doc_embeddings_map = pipeline._load_token_embeddings_for_candidates(candidate_doc_ids[:5])
                logger.info(f"Token embeddings loaded for {len(doc_embeddings_map)} documents")
                if not doc_embeddings_map:
                    logger.error("CRITICAL: No token embeddings loaded for candidates!")
                    debug_token_embeddings_loading(pipeline, candidate_doc_ids[:5])
            except Exception as e:
                logger.error(f"Token embeddings loading FAILED: {e}")
                logger.error(traceback.format_exc())
        
        # Step 5: Test full pipeline execution
        logger.info("Step 5: Testing full pipeline execution...")
        try:
            result = pipeline.run(test_query, top_k=3)
            logger.info(f"Pipeline execution result:")
            logger.info(f"  - Query: {result.get('query', 'N/A')}")
            logger.info(f"  - Answer length: {len(result.get('answer', ''))}")
            logger.info(f"  - Retrieved documents: {len(result.get('retrieved_documents', []))}")
            
            if not result.get('retrieved_documents'):
                logger.error("CRITICAL: Pipeline returned no documents!")
            
        except Exception as e:
            logger.error(f"Full pipeline execution FAILED: {e}")
            logger.error(traceback.format_exc())
            
    except Exception as e:
        logger.error(f"ColBERT debug setup FAILED: {e}")
        logger.error(traceback.format_exc())

def debug_hnsw_search(pipeline, query_text):
    """Debug HNSW search in detail."""
    logger.info("--- DETAILED HNSW SEARCH DEBUG ---")
    
    connection = pipeline.connection_manager.get_connection()
    cursor = connection.cursor()
    
    try:
        # Generate query embedding
        query_token_embeddings = pipeline.colbert_query_encoder(query_text)
        import numpy as np
        avg_embedding = np.mean(query_token_embeddings, axis=0)
        query_vector_str = f"[{','.join([f'{float(x):.10f}' for x in avg_embedding])}]"
        
        logger.info(f"Query vector length: {len(avg_embedding)}")
        logger.info(f"Query vector sample: {query_vector_str[:100]}...")
        
        # Test basic vector search without HNSW
        test_sql = """
            SELECT TOP 5 doc_id, 
                   VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) AS score,
                   SUBSTRING(text_content, 1, 100) AS preview
            FROM RAG.SourceDocuments 
            WHERE embedding IS NOT NULL
            ORDER BY score DESC
        """
        
        cursor.execute(test_sql, [query_vector_str])
        results = cursor.fetchall()
        
        logger.info(f"Basic vector search returned {len(results)} results:")
        for i, row in enumerate(results):
            doc_id, score, preview = row
            logger.info(f"  {i+1}. Doc {doc_id}: score={float(score):.4f}, preview='{preview}...'")
            
        if not results:
            logger.error("CRITICAL: Basic vector search returned no results!")
            
            # Check if embeddings are valid
            cursor.execute("SELECT TOP 3 doc_id, embedding FROM RAG.SourceDocuments WHERE embedding IS NOT NULL")
            embedding_samples = cursor.fetchall()
            
            for doc_id, embedding in embedding_samples:
                logger.info(f"Sample embedding for doc {doc_id}: {str(embedding)[:100]}...")
                
    except Exception as e:
        logger.error(f"HNSW debug FAILED: {e}")
        logger.error(traceback.format_exc())
    finally:
        cursor.close()

def debug_token_embeddings_loading(pipeline, candidate_doc_ids):
    """Debug token embeddings loading in detail."""
    logger.info("--- DETAILED TOKEN EMBEDDINGS DEBUG ---")
    
    connection = pipeline.connection_manager.get_connection()
    cursor = connection.cursor()
    
    try:
        # Check if DocumentTokenEmbeddings table has data for these candidates
        placeholders = ','.join(['?' for _ in candidate_doc_ids])
        check_sql = f"""
            SELECT doc_id, COUNT(*) as token_count
            FROM RAG.DocumentTokenEmbeddings 
            WHERE doc_id IN ({placeholders})
            GROUP BY doc_id
        """
        
        cursor.execute(check_sql, candidate_doc_ids)
        results = cursor.fetchall()
        
        logger.info(f"Token embeddings check for {len(candidate_doc_ids)} candidates:")
        for doc_id, token_count in results:
            logger.info(f"  Doc {doc_id}: {token_count} tokens")
            
        if not results:
            logger.error("CRITICAL: No token embeddings found for any candidate documents!")
            
            # Check if DocumentTokenEmbeddings table has any data at all
            cursor.execute("SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings")
            total_tokens = cursor.fetchone()[0]
            logger.info(f"Total token embeddings in database: {total_tokens}")
            
            if total_tokens > 0:
                # Show sample token embeddings
                cursor.execute("SELECT TOP 3 doc_id, token_index, token_embedding FROM RAG.DocumentTokenEmbeddings")
                samples = cursor.fetchall()
                for doc_id, token_index, embedding in samples:
                    logger.info(f"Sample token: doc_id={doc_id}, token_index={token_index}, embedding={str(embedding)[:50]}...")
                    
    except Exception as e:
        logger.error(f"Token embeddings debug FAILED: {e}")
        logger.error(traceback.format_exc())
    finally:
        cursor.close()

def debug_noderag_pipeline():
    """Debug NodeRAG pipeline step by step."""
    logger.info("=" * 80)
    logger.info("DEBUGGING NODERAG PIPELINE")
    logger.info("=" * 80)
    
    try:
        # Import required modules
        from iris_rag.core.connection import ConnectionManager
        from iris_rag.config.manager import ConfigurationManager
        from iris_rag.pipelines.noderag import NodeRAGPipeline
        from iris_rag.embeddings.manager import EmbeddingManager
        
        # Initialize components
        config_manager = ConfigurationManager()
        connection_manager = ConnectionManager(config_manager)
        embedding_manager = EmbeddingManager(config_manager=config_manager)
        
        # Test query
        test_query = "What are the effects of metformin on type 2 diabetes?"
        logger.info(f"Testing query: {test_query}")
        
        # Initialize pipeline
        pipeline = NodeRAGPipeline(
            connection_manager=connection_manager,
            config_manager=config_manager,
            embedding_manager=embedding_manager
        )
        
        # Step 1: Test embedding generation
        logger.info("Step 1: Testing embedding generation...")
        try:
            query_embedding = embedding_manager.embed_text(test_query)
            logger.info(f"Embedding generation SUCCESS: {len(query_embedding)} dimensions")
        except Exception as e:
            logger.error(f"Embedding generation FAILED: {e}")
            logger.error(traceback.format_exc())
            return
        
        # Step 2: Test database tables
        logger.info("Step 2: Testing database tables...")
        connection = connection_manager.get_connection()
        cursor = connection.cursor()
        
        try:
            # Check KnowledgeGraphNodes
            try:
                cursor.execute("SELECT COUNT(*) FROM RAG.KnowledgeGraphNodes")
                kg_nodes_count = cursor.fetchone()[0]
                logger.info(f"KnowledgeGraphNodes count: {kg_nodes_count}")
            except Exception as e:
                logger.warning(f"KnowledgeGraphNodes table issue: {e}")
                kg_nodes_count = 0
            
            # Check KnowledgeGraphEdges
            try:
                cursor.execute("SELECT COUNT(*) FROM RAG.KnowledgeGraphEdges")
                kg_edges_count = cursor.fetchone()[0]
                logger.info(f"KnowledgeGraphEdges count: {kg_edges_count}")
            except Exception as e:
                logger.warning(f"KnowledgeGraphEdges table issue: {e}")
                kg_edges_count = 0
            
            # Check SourceDocuments as fallback
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NOT NULL")
            source_docs_with_embeddings = cursor.fetchone()[0]
            logger.info(f"SourceDocuments with embeddings: {source_docs_with_embeddings}")
            
        except Exception as e:
            logger.error(f"Database check FAILED: {e}")
            return
        finally:
            cursor.close()
        
        # Step 3: Test initial node identification
        logger.info("Step 3: Testing initial node identification...")
        try:
            seed_node_ids = pipeline._identify_initial_search_nodes(test_query, top_n_seed=5)
            logger.info(f"Initial node identification: Found {len(seed_node_ids)} seed nodes")
            if seed_node_ids:
                logger.info(f"Seed node IDs: {seed_node_ids}")
            else:
                logger.error("CRITICAL: No seed nodes identified!")
                debug_noderag_vector_search(pipeline, test_query)
                
        except Exception as e:
            logger.error(f"Initial node identification FAILED: {e}")
            logger.error(traceback.format_exc())
            return
        
        # Step 4: Test graph traversal (if seed nodes found)
        if seed_node_ids:
            logger.info("Step 4: Testing graph traversal...")
            try:
                relevant_node_ids = pipeline._traverse_graph(seed_node_ids, test_query)
                logger.info(f"Graph traversal: Found {len(relevant_node_ids)} relevant nodes")
                if relevant_node_ids:
                    logger.info(f"Relevant node IDs: {list(relevant_node_ids)[:10]}")  # Show first 10
            except Exception as e:
                logger.error(f"Graph traversal FAILED: {e}")
                logger.error(traceback.format_exc())
                relevant_node_ids = set(seed_node_ids)  # Fallback to seed nodes
        else:
            relevant_node_ids = set()
        
        # Step 5: Test content retrieval (if nodes found)
        if relevant_node_ids:
            logger.info("Step 5: Testing content retrieval...")
            try:
                documents = pipeline._retrieve_content_for_nodes(relevant_node_ids)
                logger.info(f"Content retrieval: Retrieved {len(documents)} documents")
                
                for i, doc in enumerate(documents[:3]):  # Show first 3
                    logger.info(f"  Doc {i+1}: ID={doc.id}, content_length={len(doc.page_content)}")
                    logger.info(f"    Content preview: {doc.page_content[:100]}...")
                    
                if not documents:
                    logger.error("CRITICAL: No content retrieved for nodes!")
                    debug_noderag_content_retrieval(pipeline, relevant_node_ids)
                    
            except Exception as e:
                logger.error(f"Content retrieval FAILED: {e}")
                logger.error(traceback.format_exc())
        
        # Step 6: Test full pipeline execution
        logger.info("Step 6: Testing full pipeline execution...")
        try:
            result = pipeline.run(test_query, top_k=3)
            logger.info(f"Pipeline execution result:")
            logger.info(f"  - Query: {result.get('query', 'N/A')}")
            logger.info(f"  - Answer length: {len(result.get('answer', ''))}")
            logger.info(f"  - Retrieved documents: {len(result.get('retrieved_documents', []))}")
            
            if not result.get('retrieved_documents'):
                logger.error("CRITICAL: Pipeline returned no documents!")
            
        except Exception as e:
            logger.error(f"Full pipeline execution FAILED: {e}")
            logger.error(traceback.format_exc())
            
    except Exception as e:
        logger.error(f"NodeRAG debug setup FAILED: {e}")
        logger.error(traceback.format_exc())

def debug_noderag_vector_search(pipeline, query_text):
    """Debug NodeRAG vector search in detail."""
    logger.info("--- DETAILED NODERAG VECTOR SEARCH DEBUG ---")
    
    connection = pipeline.connection_manager.get_connection()
    cursor = connection.cursor()
    
    try:
        # Generate query embedding
        query_embedding = pipeline.embedding_manager.embed_text(query_text)
        iris_vector_str = ','.join(map(str, query_embedding))
        
        logger.info(f"Query embedding length: {len(query_embedding)}")
        logger.info(f"Query embedding sample: {iris_vector_str[:100]}...")
        
        # Test search on SourceDocuments (fallback)
        test_sql = """
            SELECT TOP 5 doc_id,
                   VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) AS score,
                   SUBSTRING(text_content, 1, 100) AS preview
            FROM RAG.SourceDocuments 
            WHERE embedding IS NOT NULL
            ORDER BY score DESC
        """
        
        cursor.execute(test_sql, [iris_vector_str])
        results = cursor.fetchall()
        
        logger.info(f"SourceDocuments vector search returned {len(results)} results:")
        for i, row in enumerate(results):
            doc_id, score, preview = row
            logger.info(f"  {i+1}. Doc {doc_id}: score={float(score):.4f}, preview='{preview}...'")
            
        if not results:
            logger.error("CRITICAL: SourceDocuments vector search returned no results!")
            
    except Exception as e:
        logger.error(f"NodeRAG vector search debug FAILED: {e}")
        logger.error(traceback.format_exc())
    finally:
        cursor.close()

def debug_noderag_content_retrieval(pipeline, node_ids):
    """Debug NodeRAG content retrieval in detail."""
    logger.info("--- DETAILED NODERAG CONTENT RETRIEVAL DEBUG ---")
    
    connection = pipeline.connection_manager.get_connection()
    cursor = connection.cursor()
    
    try:
        # Check content for these node IDs
        placeholders = ','.join(['?' for _ in node_ids])
        node_id_list = list(node_ids)
        
        # Test SourceDocuments query
        test_sql = f"""
            SELECT doc_id, 
                   CASE WHEN text_content IS NULL THEN 'NULL' ELSE 'NOT NULL' END as content_status,
                   CASE WHEN text_content IS NULL THEN 0 ELSE LENGTH(text_content) END as content_length
            FROM RAG.SourceDocuments 
            WHERE doc_id IN ({placeholders})
        """
        
        cursor.execute(test_sql, node_id_list)
        results = cursor.fetchall()
        
        logger.info(f"Content check for {len(node_id_list)} nodes:")
        for doc_id, content_status, content_length in results:
            logger.info(f"  Doc {doc_id}: {content_status}, length={content_length}")
            
        if not results:
            logger.error("CRITICAL: No content found for any node IDs!")
            
    except Exception as e:
        logger.error(f"NodeRAG content retrieval debug FAILED: {e}")
        logger.error(traceback.format_exc())
    finally:
        cursor.close()

def main():
    """Main debug function."""
    logger.info("Starting emergency debug for ColBERTRAG and NodeRAG failures...")
    
    # Debug ColBERTRAG
    debug_colbert_pipeline()
    
    logger.info("\n" + "=" * 80 + "\n")
    
    # Debug NodeRAG
    debug_noderag_pipeline()
    
    logger.info("Debug completed. Check debug_colbert_noderag.log for full details.")

if __name__ == "__main__":
    main()