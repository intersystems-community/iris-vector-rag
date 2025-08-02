#!/usr/bin/env python3
"""
ColBERT Performance Profiler

This script profiles the ColBERT pipeline to identify the real bottlenecks
and count exactly how many vector operations are being performed.
"""

import os
import sys
import time
import logging
from typing import List

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from common.iris_connection_manager import get_iris_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ColBERTProfiler:
    """Profile ColBERT operations to identify bottlenecks."""
    
    def __init__(self):
        self.connection = get_iris_connection()
        self.operation_counts = {
            'db_queries': 0,
            'vector_operations': 0,
            'string_parsing': 0,
            'maxsim_calculations': 0
        }
        self.timing_breakdown = {}
    
    def profile_current_implementation(self):
        """Profile the current ColBERT implementation step by step."""
        logger.info("🔍 Profiling Current ColBERT Implementation")
        
        # Simulate query token embeddings (3 tokens, 384 dimensions each)
        query_tokens = [
            [0.1] * 384,  # Token 1
            [0.2] * 384,  # Token 2  
            [0.3] * 384,  # Token 3
        ]
        
        total_start = time.time()
        cursor = self.connection.cursor()
        
        try:
            # Step 1: Get all document IDs
            step_start = time.time()
            cursor.execute("SELECT DISTINCT doc_id FROM RAG.DocumentTokenEmbeddings")
            doc_ids = [row[0] for row in cursor.fetchall()]
            self.operation_counts['db_queries'] += 1
            self.timing_breakdown['get_doc_ids'] = time.time() - step_start
            
            logger.info(f"📊 Found {len(doc_ids)} documents to evaluate")
            
            # Step 2: Process each document (current implementation)
            step_start = time.time()
            doc_scores = []
            
            # Sample first 10 documents to profile
            sample_docs = doc_ids[:10]
            
            for i, doc_id in enumerate(sample_docs):
                doc_start = time.time()
                
                # Get token embeddings for this document
                cursor.execute("""
                SELECT token_embedding 
                FROM RAG.DocumentTokenEmbeddings 
                WHERE doc_id = ? 
                ORDER BY token_index
                """, (doc_id,))
                self.operation_counts['db_queries'] += 1
                
                token_rows = cursor.fetchall()
                
                # Parse embeddings (this is where string parsing happens)
                doc_token_embeddings = []
                for token_row in token_rows:
                    embedding_str = token_row[0]
                    self.operation_counts['string_parsing'] += 1
                    
                    # Parse vector string
                    if embedding_str.startswith('[') and embedding_str.endswith(']'):
                        embedding_values = [float(x.strip()) for x in embedding_str[1:-1].split(',')]
                    else:
                        embedding_values = [float(x.strip()) for x in embedding_str.split(',')]
                    doc_token_embeddings.append(embedding_values)
                
                # Calculate MaxSim (this is where vector operations happen)
                if doc_token_embeddings:
                    maxsim_score = self._calculate_maxsim_with_profiling(query_tokens, doc_token_embeddings)
                    doc_scores.append((doc_id, maxsim_score))
                    self.operation_counts['maxsim_calculations'] += 1
                
                doc_time = time.time() - doc_start
                logger.info(f"   Doc {i+1}/{len(sample_docs)}: {doc_id} - {len(token_rows)} tokens, {doc_time:.3f}s")
            
            self.timing_breakdown['process_documents'] = time.time() - step_start
            
            # Step 3: Sort and get top results
            step_start = time.time()
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            top_docs = doc_scores[:5]
            self.timing_breakdown['sort_results'] = time.time() - step_start
            
            total_time = time.time() - total_start
            self.timing_breakdown['total'] = total_time
            
            # Print detailed analysis
            self._print_performance_analysis(len(sample_docs), len(doc_ids))
            
        finally:
            cursor.close()
    
    def _calculate_maxsim_with_profiling(self, query_tokens: List[List[float]], doc_tokens: List[List[float]]) -> float:
        """Calculate MaxSim with operation counting."""
        import numpy as np
        
        if not query_tokens or not doc_tokens:
            return 0.0
        
        # Convert to numpy arrays
        query_matrix = np.array(query_tokens)  # Shape: (num_query_tokens, 384)
        doc_matrix = np.array(doc_tokens)      # Shape: (num_doc_tokens, 384)
        
        # This is the expensive operation: matrix multiplication
        # query_matrix: (3, 384) x doc_matrix.T: (384, num_doc_tokens) = (3, num_doc_tokens)
        similarity_matrix = np.dot(query_matrix, doc_matrix.T)
        
        # Count vector operations: 3 query tokens × num_doc_tokens × 384 dimensions
        vector_ops = len(query_tokens) * len(doc_tokens) * 384
        self.operation_counts['vector_operations'] += vector_ops
        
        # MaxSim: for each query token, find max similarity with any doc token
        max_similarities = np.max(similarity_matrix, axis=1)
        maxsim_score = np.mean(max_similarities)
        
        return float(maxsim_score)
    
    def _print_performance_analysis(self, docs_processed: int, total_docs: int):
        """Print detailed performance analysis."""
        logger.info("\n" + "="*60)
        logger.info("📊 COLBERT PERFORMANCE ANALYSIS")
        logger.info("="*60)
        
        # Timing breakdown
        logger.info("⏱️ TIMING BREAKDOWN:")
        for operation, duration in self.timing_breakdown.items():
            percentage = (duration / self.timing_breakdown['total']) * 100
            logger.info(f"   {operation:20s}: {duration:6.3f}s ({percentage:5.1f}%)")
        
        # Operation counts
        logger.info("\n🔢 OPERATION COUNTS:")
        for operation, count in self.operation_counts.items():
            logger.info(f"   {operation:20s}: {count:,}")
        
        # Extrapolated analysis
        logger.info(f"\n📈 EXTRAPOLATED TO ALL {total_docs:,} DOCUMENTS:")
        
        # Database queries
        queries_per_doc = self.operation_counts['db_queries'] / docs_processed
        total_queries = queries_per_doc * total_docs
        logger.info(f"   Database queries: {total_queries:,.0f}")
        
        # Vector operations
        vector_ops_per_doc = self.operation_counts['vector_operations'] / docs_processed
        total_vector_ops = vector_ops_per_doc * total_docs
        logger.info(f"   Vector operations: {total_vector_ops:,.0f}")
        
        # String parsing
        string_ops_per_doc = self.operation_counts['string_parsing'] / docs_processed
        total_string_ops = string_ops_per_doc * total_docs
        logger.info(f"   String parsing ops: {total_string_ops:,.0f}")
        
        # Time extrapolation
        time_per_doc = self.timing_breakdown['process_documents'] / docs_processed
        estimated_total_time = time_per_doc * total_docs
        logger.info(f"   Estimated total time: {estimated_total_time:.1f}s")
        
        # Bottleneck identification
        logger.info("\n🎯 BOTTLENECK ANALYSIS:")
        if self.timing_breakdown['process_documents'] > self.timing_breakdown['total'] * 0.8:
            logger.info("   PRIMARY BOTTLENECK: Document processing loop")
            
            # Break down document processing
            avg_doc_time = self.timing_breakdown['process_documents'] / docs_processed
            logger.info(f"   Average time per document: {avg_doc_time:.3f}s")
            
            if vector_ops_per_doc > 1000000:  # More than 1M vector operations per doc
                logger.info("   SECONDARY BOTTLENECK: Vector operations (MaxSim calculations)")
            else:
                logger.info("   SECONDARY BOTTLENECK: Database queries and string parsing")
        
        logger.info("="*60)

def main():
    """Run the ColBERT performance profiler."""
    profiler = ColBERTProfiler()
    profiler.profile_current_implementation()

if __name__ == "__main__":
    main()