#!/usr/bin/env python3
"""
Comprehensive 10K Enterprise RAG System Validation
Tests all 7 RAG techniques at 10,000 document scale with performance monitoring
"""

import sys
import os
import json
import time
import logging
import psutil
import gc
from datetime import datetime
from typing import Dict, Any
import traceback

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Assuming scripts is in project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.iris_connector import get_iris_connection # Updated import
from common.embedding_utils import get_embedding_model # Updated import
from dotenv import load_dotenv

# Import all 7 RAG techniques
from iris_rag.pipelines.basic import BasicRAGPipeline # Updated import
from iris_rag.pipelines.hyde import HyDERAGPipeline # Updated import
from iris_rag.pipelines.crag import CRAGPipeline # Updated import
from iris_rag.pipelines.colbert import ColBERTRAGPipeline # Updated import
from iris_rag.pipelines.noderag import NodeRAGPipeline # Updated import
from iris_rag.pipelines.graphrag import GraphRAGPipeline # Updated import
from iris_rag.pipelines.hybrid_ifind import HybridIFindRAGPipeline # Updated import

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'enterprise_10k_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Enterprise10KValidation:
    """Comprehensive validation of all 7 RAG techniques at 10K scale"""
    
    def __init__(self):
        self.connection = get_iris_connection()
        self.embedding_model = get_embedding_model('sentence-transformers/all-MiniLM-L6-v2')
        
        # Test queries for comprehensive evaluation
        self.test_queries = [
            "What is diabetes and how is it treated?",
            "Explain the mechanism of action of insulin in glucose metabolism",
            "What are the risk factors for cardiovascular disease?",
            "Describe the pathophysiology of hypertension",
            "What are the latest treatments for cancer immunotherapy?"
        ]
        
        # RAG technique configurations
        self.rag_techniques = {
            'BasicRAG': {
                'class': BasicRAGPipeline,
                'description': 'Reliable production baseline with vector similarity search'
            },
            'HyDE': {
                'class': HyDERAGPipeline,
                'description': 'Hypothetical document generation for enhanced retrieval'
            },
            'CRAG': {
                'class': CRAGPipeline,
                'description': 'Corrective retrieval with enhanced coverage'
            },
            'ColBERT': {
                'class': ColBERTRAGPipeline,
                'description': 'Token-level semantic matching with fine-grained relevance'
            },
            'NodeRAG': {
                'class': NodeRAGPipeline,
                'description': 'Maximum coverage specialist with comprehensive retrieval'
            },
            'GraphRAG': {
                'class': GraphRAGPipeline,
                'description': 'Ultra-fast graph-based retrieval with entity relationships'
            },
            'HybridIFindRAG': {
                'class': HybridIFindRAGPipeline,
                'description': 'Multi-modal fusion approach combining multiple strategies'
            }
        }
        
        self.validation_results = {}
    
    def embedding_func(self, texts):
        """Embedding function for RAG techniques"""
        if isinstance(texts, str):
            texts = [texts]
        return self.embedding_model.encode(texts)
    
    def llm_func(self, prompt):
        """LLM function for RAG techniques"""
        return f"Based on the provided medical literature context: {prompt[:100]}..."
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        try:
            memory = psutil.virtual_memory()
            process = psutil.Process()
            
            return {
                'system_memory_total_gb': memory.total / (1024**3),
                'system_memory_used_gb': memory.used / (1024**3),
                'system_memory_percent': memory.percent,
                'process_memory_mb': process.memory_info().rss / (1024**2),
                'process_memory_percent': process.memory_percent(),
                'cpu_percent': psutil.cpu_percent(interval=1),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"❌ Failed to get system metrics: {e}")
            return {}
    
    def get_database_scale_metrics(self) -> Dict[str, Any]:
        """Get database metrics at current scale"""
        try:
            cursor = self.connection.cursor()
            
            # Core document counts
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
            doc_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM RAG.DocumentChunks")
            chunk_count = cursor.fetchone()[0]
            
            # Knowledge Graph scale
            try:
                cursor.execute("SELECT COUNT(*) FROM RAG.KnowledgeGraphEntities")
                entity_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM RAG.KnowledgeGraphRelationships")
                rel_count = cursor.fetchone()[0]
            except:
                entity_count = 0
                rel_count = 0
            
            # ColBERT token embeddings scale
            try:
                cursor.execute("SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings")
                token_count = cursor.fetchone()[0]
            except:
                token_count = 0
            
            cursor.close()
            
            return {
                'document_count': doc_count,
                'chunk_count': chunk_count,
                'entity_count': entity_count,
                'relationship_count': rel_count,
                'token_embedding_count': token_count,
                'chunks_per_document': chunk_count / doc_count if doc_count > 0 else 0,
                'entities_per_document': entity_count / doc_count if doc_count > 0 else 0,
                'scale_category': self.categorize_scale(doc_count),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get database scale metrics: {e}")
            return {}
    
    def categorize_scale(self, doc_count: int) -> str:
        """Categorize the current scale"""
        if doc_count >= 50000:
            return "Enterprise Scale (50K+)"
        elif doc_count >= 25000:
            return "Large Scale (25K+)"
        elif doc_count >= 10000:
            return "Medium Scale (10K+)"
        elif doc_count >= 5000:
            return "Small Scale (5K+)"
        elif doc_count >= 1000:
            return "Development Scale (1K+)"
        else:
            return "Prototype Scale (<1K)"
    
    def test_single_technique(self, technique_name: str, technique_config: Dict[str, Any]) -> Dict[str, Any]:
        """Test a single RAG technique comprehensively"""
        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 TESTING {technique_name.upper()}")
        logger.info(f"📝 {technique_config['description']}")
        logger.info(f"{'='*60}")
        
        technique_results = {
            'technique_name': technique_name,
            'description': technique_config['description'],
            'test_results': [],
            'performance_metrics': {},
            'error_details': None,
            'success': False
        }
        
        try:
            # Initialize technique
            logger.info(f"🔧 Initializing {technique_name}...")
            start_init = time.time()
            
            technique_class = technique_config['class']
            pipeline = technique_class(
                self.connection, 
                self.embedding_func, 
                self.llm_func
            )
            
            init_time = time.time() - start_init
            logger.info(f"✅ {technique_name} initialized in {init_time:.2f}s")
            
            # System metrics before testing
            system_before = self.get_system_metrics()
            
            # Test with all queries
            query_results = []
            total_response_time = 0
            successful_queries = 0
            
            for i, query in enumerate(self.test_queries, 1):
                logger.info(f"🔍 Query {i}/{len(self.test_queries)}: {query[:50]}...")
                
                try:
                    query_start = time.time()
                    
                    # Execute query
                    result = pipeline.run(query, top_k=5)
                    
                    query_time = time.time() - query_start
                    total_response_time += query_time
                    successful_queries += 1
                    
                    # Analyze result quality
                    answer_length = len(result.get('answer', ''))
                    retrieved_docs = len(result.get('retrieved_documents', []))
                    
                    query_result = {
                        'query_index': i,
                        'query': query,
                        'response_time_seconds': query_time,
                        'answer_length': answer_length,
                        'documents_retrieved': retrieved_docs,
                        'success': True
                    }
                    
                    # Technique-specific metrics
                    if 'entities' in result:
                        query_result['entities_found'] = len(result['entities'])
                    if 'relationships' in result:
                        query_result['relationships_found'] = len(result['relationships'])
                    if 'similarity_scores' in result:
                        scores = result['similarity_scores']
                        if scores:
                            query_result['avg_similarity'] = sum(scores) / len(scores)
                            query_result['max_similarity'] = max(scores)
                    
                    query_results.append(query_result)
                    
                    logger.info(f"   ✅ Response: {query_time:.2f}s, {retrieved_docs} docs, {answer_length} chars")
                    
                    # Memory cleanup between queries
                    if i % 3 == 0:
                        gc.collect()
                    
                except Exception as e:
                    logger.error(f"   ❌ Query failed: {e}")
                    query_results.append({
                        'query_index': i,
                        'query': query,
                        'error': str(e),
                        'success': False
                    })
            
            # System metrics after testing
            system_after = self.get_system_metrics()
            
            # Calculate performance metrics
            avg_response_time = total_response_time / successful_queries if successful_queries > 0 else 0
            success_rate = successful_queries / len(self.test_queries) * 100
            
            memory_delta = system_after.get('process_memory_mb', 0) - system_before.get('process_memory_mb', 0)
            
            technique_results.update({
                'test_results': query_results,
                'performance_metrics': {
                    'initialization_time_seconds': init_time,
                    'total_queries': len(self.test_queries),
                    'successful_queries': successful_queries,
                    'success_rate_percent': success_rate,
                    'total_response_time_seconds': total_response_time,
                    'average_response_time_seconds': avg_response_time,
                    'queries_per_second': successful_queries / total_response_time if total_response_time > 0 else 0,
                    'memory_delta_mb': memory_delta,
                    'system_before': system_before,
                    'system_after': system_after
                },
                'success': success_rate >= 80  # Consider successful if 80%+ queries work
            })
            
            if technique_results['success']:
                logger.info(f"✅ {technique_name} validation PASSED")
                logger.info(f"   📊 Success rate: {success_rate:.1f}%")
                logger.info(f"   ⚡ Avg response: {avg_response_time:.2f}s")
                logger.info(f"   🧠 Memory delta: {memory_delta:.1f}MB")
            else:
                logger.warning(f"⚠️ {technique_name} validation PARTIAL")
                logger.warning(f"   📊 Success rate: {success_rate:.1f}% (below 80% threshold)")
            
        except Exception as e:
            logger.error(f"❌ {technique_name} validation FAILED: {e}")
            technique_results.update({
                'error_details': str(e),
                'success': False
            })
            traceback.print_exc()
        
        return technique_results

def main():
    """Main execution function"""
    logger.info("🚀 ENTERPRISE 10K RAG SYSTEM VALIDATION")
    logger.info("="*80)
    
    try:
        validator = Enterprise10KValidation()
        
        # Get current system scale
        logger.info("📊 Assessing current system scale...")
        system_scale = validator.get_database_scale_metrics()
        
        current_docs = system_scale.get('document_count', 0)
        scale_category = system_scale.get('scale_category', 'Unknown')
        
        logger.info(f"📈 Current scale: {current_docs:,} documents ({scale_category})")
        logger.info(f"📋 Chunks: {system_scale.get('chunk_count', 0):,}")
        logger.info(f"🔗 Entities: {system_scale.get('entity_count', 0):,}")
        
        # Test all techniques
        logger.info(f"\n🧪 Testing {len(validator.rag_techniques)} RAG techniques...")
        
        start_time = time.time()
        successful_techniques = 0
        all_results = {}
        
        for technique_name, technique_config in validator.rag_techniques.items():
            technique_result = validator.test_single_technique(technique_name, technique_config)
            all_results[technique_name] = technique_result
            
            if technique_result['success']:
                successful_techniques += 1
            
            # Brief pause between techniques
            time.sleep(2)
            gc.collect()
        
        total_validation_time = time.time() - start_time
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"enterprise_10k_validation_results_{timestamp}.json"
        
        final_results = {
            'system_scale': system_scale,
            'technique_results': all_results,
            'validation_summary': {
                'total_validation_time_seconds': total_validation_time,
                'total_validation_time_minutes': total_validation_time / 60,
                'techniques_tested': len(validator.rag_techniques),
                'techniques_successful': successful_techniques,
                'success_rate_percent': successful_techniques / len(validator.rag_techniques) * 100,
                'system_scale_category': scale_category,
                'completion_time': datetime.now().isoformat()
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        logger.info(f"\n💾 Results saved to {results_file}")
        
        # Final summary
        logger.info("\n" + "="*80)
        logger.info("🎉 ENTERPRISE 10K VALIDATION COMPLETE")
        logger.info("="*80)
        
        summary = final_results['validation_summary']
        logger.info(f"📊 Techniques tested: {summary['techniques_tested']}")
        logger.info(f"✅ Techniques successful: {summary['techniques_successful']}")
        logger.info(f"📈 Success rate: {summary['success_rate_percent']:.1f}%")
        logger.info(f"⏱️ Total time: {summary['total_validation_time_minutes']:.1f} minutes")
        
        return 0 if summary['success_rate_percent'] >= 80 else 1
        
    except Exception as e:
        logger.error(f"❌ Critical error in 10K validation: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)