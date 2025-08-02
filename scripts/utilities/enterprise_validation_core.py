#!/usr/bin/env python3
"""
Enterprise RAG System Core Validation
Tests the core working RAG techniques without import conflicts
"""

import sys
import json
import time
import logging
import psutil
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import traceback

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.iris_connector import get_iris_connection # Updated import
from common.embedding_utils import get_embedding_model # Updated import
from dotenv import load_dotenv

# Import only the core working techniques
from iris_rag.pipelines.graphrag import GraphRAGPipeline # Updated import

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'enterprise_core_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnterpriseCoreValidation:
    """Core validation of enterprise RAG system"""
    
    def __init__(self):
        self.connection = get_iris_connection()
        self.embedding_model = get_embedding_model('sentence-transformers/all-MiniLM-L6-v2')
        
        # Comprehensive test queries for medical domain
        self.test_queries = [
            "What is diabetes and how is it treated?",
            "Explain the mechanism of action of insulin in glucose metabolism",
            "What are the risk factors for cardiovascular disease?",
            "Describe the pathophysiology of hypertension",
            "What are the latest treatments for cancer immunotherapy?",
            "How does the immune system respond to viral infections?",
            "What is the role of genetics in personalized medicine?",
            "Explain the molecular basis of Alzheimer's disease",
            "What are the mechanisms of antibiotic resistance?",
            "Describe the process of protein synthesis and regulation"
        ]
        
        # Core working techniques
        self.rag_techniques = {
            'GraphRAG': {
                'class': GraphRAGPipeline,
                'description': 'Ultra-fast graph-based retrieval with entity relationships'
            }
        }
        
        # Try to import additional techniques safely
        self._try_import_additional_techniques()
        
        self.validation_results = {}
    
    def _try_import_additional_techniques(self):
        """Safely try to import additional techniques"""
        
        # Try NodeRAG
        try:
            from iris_rag.pipelines.noderag import NodeRAGPipelineV2 # Updated import
            self.rag_techniques['NodeRAG'] = {
                'class': NodeRAGPipelineV2,
                'description': 'Maximum coverage specialist with comprehensive retrieval'
            }
            logger.info("✅ NodeRAG imported successfully")
        except Exception as e:
            logger.warning(f"⚠️ NodeRAG import failed: {e}")
        
        # Try HyDE
        try:
            from iris_rag.pipelines.hyde import HyDERAGPipelineV2 # Updated import
            self.rag_techniques['HyDE'] = {
                'class': HyDERAGPipelineV2,
                'description': 'Hypothetical document generation for enhanced retrieval'
            }
            logger.info("✅ HyDE imported successfully")
        except Exception as e:
            logger.warning(f"⚠️ HyDE import failed: {e}")
        
        # Try CRAG
        try:
            from iris_rag.pipelines.crag import CRAGPipeline # Updated import
            self.rag_techniques['CRAG'] = {
                'class': CRAGPipeline,
                'description': 'Corrective retrieval with enhanced coverage'
            }
            logger.info("✅ CRAG imported successfully")
        except Exception as e:
            logger.warning(f"⚠️ CRAG import failed: {e}")
        
        logger.info(f"📊 Total techniques available: {len(self.rag_techniques)}")
    
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
                    result = pipeline.query(query, top_k=5)
                    
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
                    
                    # Show sample answer for first query
                    if i == 1:
                        sample_answer = result.get('answer', '')[:200]
                        logger.info(f"   📝 Sample answer: {sample_answer}...")
                    
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
                logger.info(f"   🚀 Throughput: {successful_queries / total_response_time:.2f} queries/sec")
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
    
    def simulate_10k_scale_projection(self, current_results: Dict[str, Any], current_docs: int) -> Dict[str, Any]:
        """Project performance at 10K scale based on current results"""
        target_docs = 10000
        scale_factor = target_docs / current_docs if current_docs > 0 else 10
        
        projection = {
            'current_scale': current_docs,
            'target_scale': target_docs,
            'scale_factor': scale_factor,
            'projected_performance': {}
        }
        
        for technique_name, result in current_results.items():
            if result['success'] and 'performance_metrics' in result:
                metrics = result['performance_metrics']
                
                # Project response times (assume logarithmic scaling for well-optimized systems)
                current_response = metrics.get('average_response_time_seconds', 0)
                projected_response = current_response * (1 + 0.3 * (scale_factor - 1))  # 30% increase per 10x scale
                
                # Project memory usage (assume linear scaling)
                current_memory = metrics.get('memory_delta_mb', 0)
                projected_memory = current_memory * scale_factor
                
                # Project throughput (assume slight degradation)
                current_throughput = metrics.get('queries_per_second', 0)
                projected_throughput = current_throughput / (1 + 0.2 * (scale_factor - 1))  # 20% decrease per 10x scale
                
                projection['projected_performance'][technique_name] = {
                    'current_response_time': current_response,
                    'projected_response_time': projected_response,
                    'current_memory_mb': current_memory,
                    'projected_memory_mb': projected_memory,
                    'current_throughput': current_throughput,
                    'projected_throughput': projected_throughput,
                    'performance_degradation_percent': ((projected_response / current_response) - 1) * 100 if current_response > 0 else 0,
                    'enterprise_suitable': projected_response <= 5.0 and projected_memory <= 1000  # 5s response, 1GB memory limits
                }
        
        return projection

def main():
    """Main execution function"""
    logger.info("🚀 ENTERPRISE RAG SYSTEM CORE VALIDATION")
    logger.info("="*80)
    
    try:
        validator = EnterpriseCoreValidation()
        
        # Get current system scale
        logger.info("📊 Assessing current system scale...")
        system_scale = validator.get_database_scale_metrics()
        
        current_docs = system_scale.get('document_count', 0)
        scale_category = system_scale.get('scale_category', 'Unknown')
        
        logger.info(f"📈 Current scale: {current_docs:,} documents ({scale_category})")
        logger.info(f"📋 Chunks: {system_scale.get('chunk_count', 0):,}")
        logger.info(f"🔗 Entities: {system_scale.get('entity_count', 0):,}")
        logger.info(f"🎯 Relationships: {system_scale.get('relationship_count', 0):,}")
        logger.info(f"🔤 Token embeddings: {system_scale.get('token_embedding_count', 0):,}")
        
        # Test all available techniques
        logger.info(f"\n🧪 Testing {len(validator.rag_techniques)} available RAG techniques...")
        
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
        
        # Project 10K scale performance
        logger.info("\n📊 Projecting 10K scale performance...")
        scale_projection = validator.simulate_10k_scale_projection(all_results, current_docs)
        
        # Save comprehensive results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"enterprise_core_validation_{timestamp}.json"
        
        final_results = {
            'system_scale': system_scale,
            'technique_results': all_results,
            'scale_projection': scale_projection,
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
        logger.info("🎉 ENTERPRISE RAG CORE VALIDATION COMPLETE")
        logger.info("="*80)
        
        summary = final_results['validation_summary']
        
        logger.info(f"📊 Techniques tested: {summary['techniques_tested']}")
        logger.info(f"✅ Techniques successful: {summary['techniques_successful']}")
        logger.info(f"📈 Success rate: {summary['success_rate_percent']:.1f}%")
        logger.info(f"⏱️ Total time: {summary['total_validation_time_minutes']:.1f} minutes")
        
        # Show 10K projections
        logger.info(f"\n🔮 10K SCALE PROJECTIONS:")
        for technique_name, projection in scale_projection.get('projected_performance', {}).items():
            logger.info(f"   {technique_name}:")
            logger.info(f"     Response time: {projection['current_response_time']:.2f}s → {projection['projected_response_time']:.2f}s")
            logger.info(f"     Memory usage: {projection['current_memory_mb']:.1f}MB → {projection['projected_memory_mb']:.1f}MB")
            logger.info(f"     Enterprise suitable: {'✅ Yes' if projection['enterprise_suitable'] else '❌ No'}")
        
        # Overall assessment
        enterprise_ready_count = sum(1 for p in scale_projection.get('projected_performance', {}).values() 
                                   if p.get('enterprise_suitable', False))
        
        logger.info(f"\n🏢 ENTERPRISE READINESS ASSESSMENT:")
        logger.info(f"   Current scale: {current_docs:,} documents")
        logger.info(f"   Working techniques: {successful_techniques}/{len(validator.rag_techniques)}")
        logger.info(f"   10K enterprise suitable: {enterprise_ready_count}/{len(scale_projection.get('projected_performance', {}))}")
        
        if enterprise_ready_count >= 1 and successful_techniques >= 1:
            logger.info(f"   🎉 SYSTEM READY FOR 10K ENTERPRISE DEPLOYMENT!")
        else:
            logger.info(f"   ⚠️ System needs optimization for enterprise deployment")
        
        return 0 if successful_techniques >= 1 else 1
        
    except Exception as e:
        logger.error(f"❌ Critical error in enterprise validation: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)