#!/usr/bin/env python3
"""
Comprehensive Chunking Strategy Comparison Matrix for All 7 RAG Techniques

This script creates a comprehensive performance matrix showing how each of the 7 RAG techniques
performs with each of the 4 chunking strategies:
- Recursive chunking (LangChain-inspired hierarchical splitting)
- Semantic chunking (boundary detection with topic coherence)
- Adaptive chunking (automatic strategy selection)
- Hybrid chunking (multi-strategy approach with fallback)
- Plus non-chunked baseline for comparison

The goal is to provide enterprise deployment recommendations based on comprehensive analysis.
"""

import sys
import os
import json
import time
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import statistics

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Assuming scripts is in project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.iris_connector import get_iris_connection # Updated import
from common.embedding_utils import get_embedding_model # Updated import
from chunking.enhanced_chunking_service import EnhancedDocumentChunkingService # Reverted: chunking is at project root

# Import all RAG techniques
from iris_rag.pipelines.basic import BasicRAGPipeline # Updated import
from iris_rag.pipelines.hyde import HyDERAGPipeline # Updated import
from iris_rag.pipelines.crag import CRAGPipeline # Updated import
from iris_rag.pipelines.colbert import ColBERTRAGPipeline # Updated import
from iris_rag.pipelines.noderag import NodeRAGPipeline # Updated import
from iris_rag.pipelines.graphrag import GraphRAGPipeline # Updated import
from iris_rag.pipelines.hybrid_ifind import HybridIFindRAGPipeline # Updated import

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ChunkingPerformanceResult:
    """Results for a single RAG technique with a specific chunking strategy."""
    technique_name: str
    chunking_strategy: str
    success: bool
    response_time_ms: float
    retrieved_documents_count: int
    answer_length: int
    error_message: Optional[str] = None
    chunk_count: int = 0
    avg_chunk_size: float = 0.0
    chunking_time_ms: float = 0.0

@dataclass
class ComprehensiveMatrixResults:
    """Complete results matrix for all techniques and strategies."""
    results: List[ChunkingPerformanceResult]
    test_queries: List[str]
    total_documents: int
    execution_time_seconds: float
    timestamp: str

class ChunkingStrategyMatrix:
    """Comprehensive chunking strategy comparison matrix."""
    
    def __init__(self, fast_mode: bool = False):
        self.fast_mode = fast_mode
        self.iris_connector = get_iris_connection()
        self.embedding_func = get_embedding_model(mock=True)
        # Create a wrapper function that matches the expected interface
        def embedding_wrapper(texts):
            if isinstance(texts, str):
                texts = [texts]
            return self.embedding_func.encode(texts).tolist()
        
        self.chunking_service = EnhancedDocumentChunkingService(
            embedding_func=embedding_wrapper
        )
        
        # Test queries for evaluation
        self.test_queries = [
            "What are the effects of COVID-19 on cardiovascular health?",
            "How does machine learning improve medical diagnosis?",
            "What are the latest treatments for cancer immunotherapy?",
            "How do genetic mutations affect protein function?",
            "What is the role of inflammation in autoimmune diseases?"
        ]
        
        if fast_mode:
            self.test_queries = self.test_queries[:2]  # Use fewer queries in fast mode
        
        # RAG techniques to test
        self.rag_techniques = {
            'BasicRAG': BasicRAGPipeline,
            'HyDE': HyDERAGPipeline,
            'CRAG': CRAGPipeline,
            'OptimizedColBERT': ColBERTRAGPipeline,
            'NodeRAG': NodeRAGPipeline,
            'GraphRAG': GraphRAGPipeline,
            'HybridiFindRAG': HybridIFindRAGPipeline
        }
        
        # Chunking strategies to test
        self.chunking_strategies = [
            'recursive',
            'semantic', 
            'adaptive',
            'hybrid'
        ]
        
        self.results: List[ChunkingPerformanceResult] = []
        
    def setup_chunking_infrastructure(self) -> bool:
        """Deploy chunking schema and prepare infrastructure."""
        try:
            logger.info("Setting up chunking infrastructure...")
            
            # Read and execute chunking schema
            schema_path = os.path.join(os.path.dirname(__file__), '..', 'chunking', 'chunking_schema.sql')
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            
            # Split into individual statements and execute
            statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
            
            with self.iris_connector.cursor() as cursor:
                for stmt in statements:
                    if stmt and not stmt.startswith('--'):
                        try:
                            cursor.execute(stmt)
                            logger.debug(f"Executed: {stmt[:100]}...")
                        except Exception as e:
                            if "already exists" in str(e).lower():
                                logger.debug(f"Schema element already exists: {stmt[:50]}...")
                            else:
                                logger.warning(f"Schema execution warning: {e}")
                
                self.iris_connector.commit()
            
            logger.info("✅ Chunking infrastructure setup complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup chunking infrastructure: {e}")
            return False
    
    def get_sample_documents(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get sample documents for chunking and testing."""
        try:
            query = f"""
            SELECT TOP {limit} doc_id, title, text_content, abstract
            FROM RAG.SourceDocuments_V2
            WHERE text_content IS NOT NULL
            AND LENGTH(text_content) > 1000
            ORDER BY doc_id
            """
            
            with self.iris_connector.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
                
                documents = []
                for row in rows:
                    documents.append({
                        'doc_id': row[0],
                        'title': row[1] or '',
                        'content': row[2] or '',
                        'abstract': row[3] or ''
                    })
                
                logger.info(f"Retrieved {len(documents)} sample documents")
                return documents
                
        except Exception as e:
            logger.error(f"Failed to get sample documents: {e}")
            return []
    
    def process_documents_with_chunking(self, documents: List[Dict[str, Any]], 
                                      strategy: str) -> Dict[str, Any]:
        """Process documents with a specific chunking strategy."""
        try:
            logger.info(f"Processing {len(documents)} documents with {strategy} chunking...")
            start_time = time.time()
            
            total_chunks = 0
            total_chunk_size = 0
            processed_docs = 0
            
            # Clear existing chunks for this strategy
            with self.iris_connector.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM RAG.DocumentChunks WHERE chunk_type = ?",
                    (strategy,)
                )
                self.iris_connector.commit()
            
            # Process each document
            for doc in documents:
                try:
                    # Combine title, abstract, and content for chunking
                    full_text = f"{doc['title']}\n\n{doc['abstract']}\n\n{doc['content']}"
                    
                    # Generate chunks
                    chunk_records = self.chunking_service.chunk_document(
                        doc_id=doc['doc_id'],
                        text=full_text,
                        strategy_name=strategy
                    )
                    
                    # Insert chunks into database
                    if chunk_records:
                        with self.iris_connector.cursor() as cursor:
                            for chunk in chunk_records:
                                cursor.execute("""
                                    INSERT INTO RAG.DocumentChunks 
                                    (chunk_id, doc_id, chunk_index, chunk_type, chunk_text, 
                                     start_position, end_position, embedding_str, chunk_metadata, parent_chunk_id)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    chunk['chunk_id'],
                                    chunk['doc_id'],
                                    chunk['chunk_index'],
                                    chunk['chunk_type'],
                                    chunk['chunk_text'],
                                    chunk['start_position'],
                                    chunk['end_position'],
                                    chunk['embedding_str'],
                                    chunk['chunk_metadata'],
                                    chunk['parent_chunk_id']
                                ))
                        
                        self.iris_connector.commit()
                        
                        # Update statistics
                        total_chunks += len(chunk_records)
                        total_chunk_size += sum(len(c['chunk_text']) for c in chunk_records)
                        processed_docs += 1
                        
                        if processed_docs % 10 == 0:
                            logger.info(f"Processed {processed_docs}/{len(documents)} documents...")
                
                except Exception as e:
                    logger.warning(f"Failed to process document {doc['doc_id']}: {e}")
                    continue
            
            processing_time = time.time() - start_time
            avg_chunk_size = total_chunk_size / total_chunks if total_chunks > 0 else 0
            
            logger.info(f"✅ {strategy} chunking complete: {total_chunks} chunks, "
                       f"{avg_chunk_size:.1f} avg size, {processing_time:.2f}s")
            
            return {
                'strategy': strategy,
                'total_chunks': total_chunks,
                'avg_chunk_size': avg_chunk_size,
                'processing_time_ms': processing_time * 1000,
                'processed_documents': processed_docs
            }
            
        except Exception as e:
            logger.error(f"Failed to process documents with {strategy} chunking: {e}")
            return {
                'strategy': strategy,
                'total_chunks': 0,
                'avg_chunk_size': 0,
                'processing_time_ms': 0,
                'processed_documents': 0
            }
    
    def test_rag_technique_with_chunking(self, technique_name: str, 
                                       chunking_strategy: str,
                                       query: str) -> ChunkingPerformanceResult:
        """Test a specific RAG technique with a chunking strategy."""
        try:
            logger.info(f"Testing {technique_name} with {chunking_strategy} chunking...")
            
            # Initialize RAG pipeline
            pipeline_class = self.rag_techniques[technique_name]
            # Create embedding function wrapper for RAG pipelines
            def embedding_func_wrapper(texts):
                if isinstance(texts, str):
                    texts = [texts]
                return self.embedding_func.encode(texts).tolist()
            
            pipeline = pipeline_class(
                iris_connector=self.iris_connector,
                embedding_func=embedding_func_wrapper
            )
            
            # Execute query with chunked data
            start_time = time.time()
            result = pipeline.query(query)
            response_time = (time.time() - start_time) * 1000
            
            # Get chunk statistics
            chunk_stats = self.get_chunk_statistics(chunking_strategy)
            
            return ChunkingPerformanceResult(
                technique_name=technique_name,
                chunking_strategy=chunking_strategy,
                success=True,
                response_time_ms=response_time,
                retrieved_documents_count=len(result.get('retrieved_documents', [])),
                answer_length=len(result.get('answer', '')),
                chunk_count=chunk_stats['count'],
                avg_chunk_size=chunk_stats['avg_size'],
                chunking_time_ms=chunk_stats['processing_time']
            )
            
        except Exception as e:
            logger.error(f"Failed to test {technique_name} with {chunking_strategy}: {e}")
            return ChunkingPerformanceResult(
                technique_name=technique_name,
                chunking_strategy=chunking_strategy,
                success=False,
                response_time_ms=0,
                retrieved_documents_count=0,
                answer_length=0,
                error_message=str(e)
            )
    
    def test_rag_technique_without_chunking(self, technique_name: str, 
                                          query: str) -> ChunkingPerformanceResult:
        """Test a RAG technique without chunking (baseline)."""
        try:
            logger.info(f"Testing {technique_name} without chunking (baseline)...")
            
            # Initialize RAG pipeline normally
            pipeline_class = self.rag_techniques[technique_name]
            
            # Create embedding function wrapper for RAG pipelines
            def embedding_func_wrapper(texts):
                if isinstance(texts, str):
                    texts = [texts]
                return self.embedding_func.encode(texts).tolist()
            
            pipeline = pipeline_class(
                iris_connector=self.iris_connector,
                embedding_func=embedding_func_wrapper
            )
            
            # Execute query
            start_time = time.time()
            result = pipeline.query(query)
            response_time = (time.time() - start_time) * 1000
            
            return ChunkingPerformanceResult(
                technique_name=technique_name,
                chunking_strategy='none',
                success=True,
                response_time_ms=response_time,
                retrieved_documents_count=len(result.get('retrieved_documents', [])),
                answer_length=len(result.get('answer', ''))
            )
            
        except Exception as e:
            logger.error(f"Failed to test {technique_name} without chunking: {e}")
            return ChunkingPerformanceResult(
                technique_name=technique_name,
                chunking_strategy='none',
                success=False,
                response_time_ms=0,
                retrieved_documents_count=0,
                answer_length=0,
                error_message=str(e)
            )
    
    def get_chunk_statistics(self, strategy: str) -> Dict[str, Any]:
        """Get statistics for a chunking strategy."""
        try:
            query = """
            SELECT 
                COUNT(*) as chunk_count,
                AVG(LENGTH(chunk_text)) as avg_size,
                MIN(LENGTH(chunk_text)) as min_size,
                MAX(LENGTH(chunk_text)) as max_size
            FROM RAG.DocumentChunks 
            WHERE chunk_type = ?
            """
            
            with self.iris_connector.cursor() as cursor:
                cursor.execute(query, (strategy,))
                row = cursor.fetchone()
                
                return {
                    'count': row[0] or 0,
                    'avg_size': row[1] or 0,
                    'min_size': row[2] or 0,
                    'max_size': row[3] or 0,
                    'processing_time': 0  # Will be filled by chunking process
                }
                
        except Exception as e:
            logger.error(f"Failed to get chunk statistics for {strategy}: {e}")
            return {'count': 0, 'avg_size': 0, 'min_size': 0, 'max_size': 0, 'processing_time': 0}
    
    def run_comprehensive_matrix(self) -> ComprehensiveMatrixResults:
        """Run the complete chunking strategy comparison matrix."""
        logger.info("🚀 Starting Comprehensive Chunking Strategy Matrix")
        start_time = time.time()
        
        # Setup infrastructure
        if not self.setup_chunking_infrastructure():
            raise Exception("Failed to setup chunking infrastructure")
        
        # Get sample documents
        doc_limit = 20 if self.fast_mode else 100
        documents = self.get_sample_documents(limit=doc_limit)
        if not documents:
            raise Exception("No documents available for testing")
        
        logger.info(f"Testing with {len(documents)} documents")
        
        # Process documents with each chunking strategy
        chunking_stats = {}
        for strategy in self.chunking_strategies:
            stats = self.process_documents_with_chunking(documents, strategy)
            chunking_stats[strategy] = stats
        
        # Test each RAG technique with each chunking strategy
        for technique_name in self.rag_techniques.keys():
            logger.info(f"\n📊 Testing {technique_name} across all strategies...")
            
            # Test without chunking (baseline)
            for query in self.test_queries:
                result = self.test_rag_technique_without_chunking(technique_name, query)
                self.results.append(result)
            
            # Test with each chunking strategy
            for strategy in self.chunking_strategies:
                for query in self.test_queries:
                    result = self.test_rag_technique_with_chunking(
                        technique_name, strategy, query
                    )
                    # Add chunking statistics
                    if strategy in chunking_stats:
                        result.chunk_count = chunking_stats[strategy]['total_chunks']
                        result.avg_chunk_size = chunking_stats[strategy]['avg_chunk_size']
                        result.chunking_time_ms = chunking_stats[strategy]['processing_time_ms']
                    
                    self.results.append(result)
        
        execution_time = time.time() - start_time
        
        return ComprehensiveMatrixResults(
            results=self.results,
            test_queries=self.test_queries,
            total_documents=len(documents),
            execution_time_seconds=execution_time,
            timestamp=datetime.now().isoformat()
        )
    
    def generate_performance_matrix(self, results: ComprehensiveMatrixResults) -> Dict[str, Any]:
        """Generate comprehensive performance analysis matrix."""
        logger.info("📈 Generating performance matrix...")
        
        # Organize results by technique and strategy
        matrix = {}
        for result in results.results:
            if result.technique_name not in matrix:
                matrix[result.technique_name] = {}
            
            strategy = result.chunking_strategy
            if strategy not in matrix[result.technique_name]:
                matrix[result.technique_name][strategy] = []
            
            matrix[result.technique_name][strategy].append(result)
        
        # Calculate aggregated metrics
        performance_summary = {}
        for technique in matrix:
            performance_summary[technique] = {}
            
            for strategy in matrix[technique]:
                results_list = matrix[technique][strategy]
                successful_results = [r for r in results_list if r.success]
                
                if successful_results:
                    avg_response_time = statistics.mean([r.response_time_ms for r in successful_results])
                    avg_docs_retrieved = statistics.mean([r.retrieved_documents_count for r in successful_results])
                    avg_answer_length = statistics.mean([r.answer_length for r in successful_results])
                    success_rate = len(successful_results) / len(results_list)
                    
                    # Calculate improvement over baseline (none strategy)
                    baseline_time = None
                    if 'none' in matrix[technique]:
                        baseline_results = [r for r in matrix[technique]['none'] if r.success]
                        if baseline_results:
                            baseline_time = statistics.mean([r.response_time_ms for r in baseline_results])
                    
                    improvement_ratio = 1.0
                    overhead_ms = 0.0
                    if baseline_time and strategy != 'none':
                        improvement_ratio = baseline_time / avg_response_time if avg_response_time > 0 else 1.0
                        overhead_ms = avg_response_time - baseline_time
                    
                    performance_summary[technique][strategy] = {
                        'success_rate': success_rate,
                        'avg_response_time_ms': avg_response_time,
                        'avg_documents_retrieved': avg_docs_retrieved,
                        'avg_answer_length': avg_answer_length,
                        'improvement_ratio': improvement_ratio,
                        'overhead_ms': overhead_ms,
                        'chunk_count': successful_results[0].chunk_count if successful_results else 0,
                        'avg_chunk_size': successful_results[0].avg_chunk_size if successful_results else 0,
                        'chunking_time_ms': successful_results[0].chunking_time_ms if successful_results else 0
                    }
                else:
                    performance_summary[technique][strategy] = {
                        'success_rate': 0.0,
                        'avg_response_time_ms': 0.0,
                        'avg_documents_retrieved': 0.0,
                        'avg_answer_length': 0.0,
                        'improvement_ratio': 0.0,
                        'overhead_ms': 0.0,
                        'chunk_count': 0,
                        'avg_chunk_size': 0,
                        'chunking_time_ms': 0
                    }
        
        return {
            'performance_matrix': performance_summary,
            'raw_results': [
                {
                    'technique': r.technique_name,
                    'strategy': r.chunking_strategy,
                    'success': r.success,
                    'response_time_ms': r.response_time_ms,
                    'retrieved_docs': r.retrieved_documents_count,
                    'answer_length': r.answer_length,
                    'error': r.error_message,
                    'chunk_count': r.chunk_count,
                    'avg_chunk_size': r.avg_chunk_size,
                    'chunking_time_ms': r.chunking_time_ms
                }
                for r in results.results
            ],
            'test_metadata': {
                'total_documents': results.total_documents,
                'test_queries': results.test_queries,
                'execution_time_seconds': results.execution_time_seconds,
                'timestamp': results.timestamp
            }
        }
    
    def generate_recommendations(self, performance_matrix: Dict[str, Any]) -> Dict[str, Any]:
        """Generate enterprise deployment recommendations."""
        logger.info("🎯 Generating deployment recommendations...")
        
        matrix = performance_matrix['performance_matrix']
        strategies = ['none', 'recursive', 'semantic', 'adaptive', 'hybrid']
        
        # Find best strategy for each technique
        technique_recommendations = {}
        for technique in matrix:
            best_strategy = 'none'
            best_score = 0.0
            
            for strategy in strategies:
                if strategy in matrix[technique]:
                    metrics = matrix[technique][strategy]
                    if metrics['success_rate'] > 0.8:  # Only consider successful strategies
                        # Score based on improvement ratio and low overhead
                        score = metrics['improvement_ratio'] * metrics['success_rate']
                        if metrics['overhead_ms'] < 0:  # Negative overhead is good
                            score *= 1.2
                        
                        if score > best_score:
                            best_score = score
                            best_strategy = strategy
            
            technique_recommendations[technique] = {
                'recommended_strategy': best_strategy,
                'score': best_score,
                'metrics': matrix[technique].get(best_strategy, {})
            }
        
        # Find best techniques for each strategy
        strategy_recommendations = {}
        for strategy in strategies:
            technique_scores = []
            for technique in matrix:
                if strategy in matrix[technique]:
                    metrics = matrix[technique][strategy]
                    if metrics['success_rate'] > 0.8:
                        score = metrics['improvement_ratio'] * metrics['success_rate']
                        technique_scores.append((technique, score, metrics))
            
            # Sort by score
            technique_scores.sort(key=lambda x: x[1], reverse=True)
            strategy_recommendations[strategy] = {
                'best_techniques': technique_scores[:3],  # Top 3
                'total_compatible': len(technique_scores)
            }
        
        # Overall enterprise recommendations
        enterprise_recommendations = {
            'fastest_combinations': [],
            'most_reliable_combinations': [],
            'best_improvement_combinations': [],
            'production_ready_combinations': []
        }
        
        # Find fastest combinations
        all_combinations = []
        for technique in matrix:
            for strategy in matrix[technique]:
                metrics = matrix[technique][strategy]
                if metrics['success_rate'] > 0.8:
                    all_combinations.append((
                        technique, strategy, metrics['avg_response_time_ms'], 
                        metrics['success_rate'], metrics['improvement_ratio']
                    ))
        
        # Sort by response time
        all_combinations.sort(key=lambda x: x[2])
        enterprise_recommendations['fastest_combinations'] = all_combinations[:5]
        
        # Sort by success rate
        all_combinations.sort(key=lambda x: x[3], reverse=True)
        enterprise_recommendations['most_reliable_combinations'] = all_combinations[:5]
        
        # Sort by improvement ratio
        all_combinations.sort(key=lambda x: x[4], reverse=True)
        enterprise_recommendations['best_improvement_combinations'] = all_combinations[:5]
        
        # Production ready (high success rate + reasonable performance)
        production_ready = [
            combo for combo in all_combinations 
            if combo[3] > 0.9 and combo[2] < 2000  # >90% success, <2s response
        ]
        enterprise_recommendations['production_ready_combinations'] = production_ready[:10]
        
        return {
            'technique_recommendations': technique_recommendations,
            'strategy_recommendations': strategy_recommendations,
            'enterprise_recommendations': enterprise_recommendations
        }

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Chunking Strategy Matrix')
    parser.add_argument('--fast', action='store_true', help='Run in fast mode with fewer documents and queries')
    parser.add_argument('--output', default='chunking_strategy_matrix_results.json', help='Output file path')
    
    args = parser.parse_args()
    
    try:
        # Initialize matrix runner
        matrix = ChunkingStrategyMatrix(fast_mode=args.fast)
        
        # Run comprehensive analysis
        results = matrix.run_comprehensive_matrix()
        
        # Generate performance matrix
        performance_matrix = matrix.generate_performance_matrix(results)
        
        # Generate recommendations
        recommendations = matrix.generate_recommendations(performance_matrix)
        
        # Combine all results
        final_results = {
            **performance_matrix,
            'recommendations': recommendations,
            'execution_summary': {
                'total_tests': len(results.results),
                'successful_tests': len([r for r in results.results if r.success]),
                'total_techniques': len(matrix.rag_techniques),
                'total_strategies': len(matrix.chunking_strategies) + 1,  # +1 for 'none'
                'execution_time_seconds': results.execution_time_seconds,
                'fast_mode': args.fast
            }
        }
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"chunking_strategy_matrix_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "="*80)
        print("🎉 COMPREHENSIVE CHUNKING STRATEGY MATRIX COMPLETE")
        print("="*80)
        
        print(f"\n📊 EXECUTION SUMMARY:")
        print(f"   • Total Tests: {final_results['execution_summary']['total_tests']}")
        print(f"   • Successful Tests: {final_results['execution_summary']['successful_tests']}")
        print(f"   • Success Rate: {final_results['execution_summary']['successful_tests']/final_results['execution_summary']['total_tests']*100:.1f}%")
        print(f"   • Execution Time: {final_results['execution_summary']['execution_time_seconds']:.1f}s")
        print(f"   • Results File: {output_file}")
        
        print(f"\n🏆 TOP RECOMMENDATIONS:")
        
        # Show best technique-strategy combinations
        fastest = recommendations['enterprise_recommendations']['fastest_combinations'][:3]
        print(f"\n   ⚡ FASTEST COMBINATIONS:")
        for i, (tech, strat, time_ms, success, improvement) in enumerate(fastest, 1):
            print(f"      {i}. {tech} + {strat}: {time_ms:.1f}ms (success: {success:.1%})")
        
        most_reliable = recommendations['enterprise_recommendations']['most_reliable_combinations'][:3]
        print(f"\n   🛡️ MOST RELIABLE COMBINATIONS:")
        for i, (tech, strat, time_ms, success, improvement) in enumerate(most_reliable, 1):
            print(f"      {i}. {tech} + {strat}: {success:.1%} success ({time_ms:.1f}ms)")
        
        best_improvement = recommendations['enterprise_recommendations']['best_improvement_combinations'][:3]
        print(f"\n   📈 BEST IMPROVEMENT COMBINATIONS:")
        for i, (tech, strat, time_ms, success, improvement) in enumerate(best_improvement, 1):
            print(f"      {i}. {tech} + {strat}: {improvement:.2f}x improvement ({time_ms:.1f}ms)")
        
        print(f"\n✅ Matrix analysis complete! Check {output_file} for detailed results.")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Matrix execution failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)