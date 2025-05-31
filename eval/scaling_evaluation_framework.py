#!/usr/bin/env python3
"""
Comprehensive Scaling and Evaluation Framework for 7 RAG Techniques
Tests all techniques across increasing dataset sizes (1K to 50K documents) with RAGAS metrics
"""

import sys
import os
import json
import time
import logging
import psutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# RAGAS imports
try:
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
        answer_similarity,
        answer_correctness,
        context_relevancy
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("⚠️ RAGAS not installed. Install with: pip install ragas datasets")

# RAG imports - all 7 techniques
from basic_rag.pipeline_v2_fixed import BasicRAGPipelineV2Fixed as BasicRAGPipeline
from hyde.pipeline import HyDEPipeline
from crag.pipeline_jdbc_fixed import JDBCFixedCRAGPipeline as CRAGPipeline
from colbert.pipeline import OptimizedColbertRAGPipeline as ColBERTPipeline
from noderag.pipeline import NodeRAGPipeline
from graphrag.pipeline_jdbc_fixed import JDBCFixedGraphRAGPipeline as GraphRAGPipeline
from hybrid_ifind_rag.pipeline import HybridiFindRAGPipeline as HybridIFindRAGPipeline

# Common utilities
from common.iris_connector_jdbc import get_iris_connection
from common.utils import get_embedding_func, get_llm_func, DEFAULT_EMBEDDING_MODEL
from dotenv import load_dotenv

# Langchain for RAGAS LLM/Embeddings
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScalingEvaluationFramework:
    """Comprehensive scaling and evaluation framework for all 7 RAG techniques"""
    
    def __init__(self):
        load_dotenv()
        
        self.connection = get_iris_connection()
        self.embedding_func = get_embedding_func()
        
        # Setup LLM for RAGAS evaluation
        try:
            if os.getenv("OPENAI_API_KEY"):
                self.llm_func = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
                self.embedding_func_ragas = HuggingFaceEmbeddings(model_name=DEFAULT_EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})
                self.real_llm = True
                logger.info("✅ Using OpenAI GPT-3.5-turbo for RAGAS evaluation")
            else:
                self.llm_func = get_llm_func(provider="stub")
                self.embedding_func_ragas = None
                self.real_llm = False
                logger.warning("⚠️ Using stub LLM (set OPENAI_API_KEY for real RAGAS evaluation)")
        except Exception as e:
            self.llm_func = get_llm_func(provider="stub")
            self.embedding_func_ragas = None
            self.real_llm = False
            logger.warning(f"⚠️ LLM setup failed, using stub: {e}")
        
        # Dataset scaling strategy
        self.dataset_sizes = [1000, 2500, 5000, 10000, 25000, 50000]
        
        # Standardized test queries for consistent evaluation
        self.test_queries = [
            {
                "query": "What is the role of olfactory perception in honeybee behavior?",
                "ground_truth": "Olfactory perception plays a crucial role in honeybee behavior, enabling them to identify flowers, communicate through pheromones, and navigate their environment.",
                "keywords": ["olfactory", "honeybee", "perception", "behavior"],
                "category": "neuroscience"
            },
            {
                "query": "How do honeybees process neural signals related to smell?",
                "ground_truth": "Honeybees process olfactory neural signals through their antennal lobes and mushroom bodies, which integrate sensory information for behavioral responses.",
                "keywords": ["honeybee", "neural", "olfactory", "smell", "signal"],
                "category": "neuroscience"
            },
            {
                "query": "How do microRNAs regulate gene expression?",
                "ground_truth": "MicroRNAs regulate gene expression by binding to complementary sequences on target mRNAs, leading to translational repression or mRNA degradation.",
                "keywords": ["microRNA", "gene", "regulation", "expression", "mRNA"],
                "category": "molecular_biology"
            },
            {
                "query": "What is the relationship between microRNAs and disease?",
                "ground_truth": "MicroRNAs are involved in various diseases including cancer, cardiovascular disease, and neurological disorders through dysregulation of gene expression.",
                "keywords": ["microRNA", "disease", "cancer", "regulation"],
                "category": "medical"
            },
            {
                "query": "How do sensory neurons transmit information?",
                "ground_truth": "Sensory neurons transmit information through electrical signals called action potentials, which travel along axons to relay sensory input to the central nervous system.",
                "keywords": ["sensory", "neuron", "transmit", "signal", "action potential"],
                "category": "neuroscience"
            },
            {
                "query": "What are the mechanisms of neural plasticity?",
                "ground_truth": "Neural plasticity involves synaptic changes, neurogenesis, and structural modifications that allow the nervous system to adapt to experience and injury.",
                "keywords": ["neural", "plasticity", "synapse", "adaptation", "neurogenesis"],
                "category": "neuroscience"
            },
            {
                "query": "How do biological systems process sensory information?",
                "ground_truth": "Biological systems process sensory information through specialized receptors, neural pathways, and brain regions that integrate and interpret sensory inputs.",
                "keywords": ["biological", "sensory", "process", "receptor", "neural"],
                "category": "biology"
            },
            {
                "query": "How do insects use chemical signals for communication?",
                "ground_truth": "Insects use chemical signals called pheromones for various forms of communication including mating, alarm signaling, and trail marking.",
                "keywords": ["insect", "chemical", "signal", "pheromone", "communication"],
                "category": "biology"
            },
            {
                "query": "What are the latest findings in cancer research?",
                "ground_truth": "Recent cancer research has revealed new insights into tumor biology, immunotherapy approaches, and personalized treatment strategies.",
                "keywords": ["cancer", "research", "tumor", "therapy", "treatment"],
                "category": "medical"
            },
            {
                "query": "How do protein interactions affect cellular function?",
                "ground_truth": "Protein interactions are fundamental to cellular function, controlling processes like signal transduction, metabolism, and gene regulation.",
                "keywords": ["protein", "interaction", "cellular", "function", "metabolism"],
                "category": "molecular_biology"
            }
        ]
        
        # Initialize all 7 RAG techniques
        self.technique_names = [
            'BasicRAG', 'HyDE', 'CRAG', 'ColBERT', 
            'NodeRAG', 'GraphRAG', 'HybridIFindRAG'
        ]
        
    def get_database_stats(self) -> Dict[str, Any]:
        """Get current database statistics"""
        try:
            cursor = self.connection.cursor()
            
            # Count documents
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
            doc_count = cursor.fetchone()[0]
            
            # Count chunks
            cursor.execute("SELECT COUNT(*) FROM RAG.DocumentChunks")
            chunk_count = cursor.fetchone()[0]
            
            # Count token embeddings (ColBERT)
            try:
                cursor.execute("SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings")
                token_count = cursor.fetchone()[0]
            except:
                token_count = 0
            
            # Get database size (approximate)
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
            content_size = cursor.fetchone()[0] or 0
            
            cursor.close()
            
            return {
                'document_count': doc_count,
                'chunk_count': chunk_count,
                'token_embedding_count': token_count,
                'content_size_bytes': content_size,
                'content_size_mb': content_size / (1024 * 1024) if content_size else 0
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get database stats: {e}")
            return {
                'document_count': 0,
                'chunk_count': 0,
                'token_embedding_count': 0,
                'content_size_bytes': 0,
                'content_size_mb': 0
            }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system performance metrics"""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            return {
                'memory_total_gb': memory.total / (1024**3),
                'memory_used_gb': memory.used / (1024**3),
                'memory_percent': memory.percent,
                'cpu_percent': cpu_percent,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"❌ Failed to get system metrics: {e}")
            return {}
    
    def initialize_pipeline(self, technique_name: str) -> Optional[Any]:
        """Initialize a specific RAG pipeline"""
        try:
            if technique_name == 'BasicRAG':
                return BasicRAGPipeline(
                    self.connection, self.embedding_func, self.llm_func, schema="RAG"
                )
            elif technique_name == 'HyDE':
                return HyDEPipeline(
                    self.connection, self.embedding_func, self.llm_func
                )
            elif technique_name == 'CRAG':
                return CRAGPipeline(
                    self.connection, self.embedding_func, self.llm_func
                )
            elif technique_name == 'ColBERT':
                return ColBERTPipeline(
                    iris_connector=self.connection,
                    colbert_query_encoder_func=self.embedding_func,
                    colbert_doc_encoder_func=self.embedding_func,
                    llm_func=self.llm_func
                )
            elif technique_name == 'NodeRAG':
                return NodeRAGPipeline(
                    self.connection, self.embedding_func, self.llm_func
                )
            elif technique_name == 'GraphRAG':
                return GraphRAGPipeline(
                    self.connection, self.embedding_func, self.llm_func
                )
            elif technique_name == 'HybridIFindRAG':
                return HybridIFindRAGPipeline(
                    self.connection, self.embedding_func, self.llm_func
                )
            else:
                logger.error(f"❌ Unknown technique: {technique_name}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize {technique_name}: {e}")
            return None
    
    def run_single_query_with_metrics(self, pipeline: Any, technique_name: str, 
                                    query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single query and collect comprehensive metrics"""
        query = query_data["query"]
        
        # System metrics before
        system_before = self.get_system_metrics()
        
        start_time = time.time()
        try:
            # Use different parameters based on pipeline
            if technique_name == 'CRAG':
                result = pipeline.run(query, top_k=10)
            elif technique_name == 'ColBERT':
                # Limit ColBERT to prevent content overflow
                result = pipeline.run(query, top_k=5)
            else:
                result = pipeline.run(query, top_k=10, similarity_threshold=0.1)
            
            response_time = time.time() - start_time
            
            # System metrics after
            system_after = self.get_system_metrics()
            
            # Extract metrics
            documents = result.get('retrieved_documents', [])
            answer = result.get('answer', '')
            
            # Extract context texts for RAGAS
            contexts = []
            for doc in documents:
                if isinstance(doc, dict):
                    text = doc.get('text', '') or doc.get('content', '') or doc.get('chunk_text', '')
                elif hasattr(doc, 'text'):
                    text = doc.text
                elif hasattr(doc, 'content'):
                    text = doc.content
                else:
                    text = str(doc)
                if text:
                    contexts.append(text)
            
            # Calculate similarity scores
            similarity_scores = []
            for doc in documents:
                if isinstance(doc, dict) and 'score' in doc:
                    similarity_scores.append(doc['score'])
                elif hasattr(doc, 'score'):
                    similarity_scores.append(doc.score)
            
            avg_similarity = np.mean(similarity_scores) if similarity_scores else 0.0
            
            # Calculate memory usage change
            memory_delta = 0
            if system_before and system_after:
                memory_delta = system_after.get('memory_used_gb', 0) - system_before.get('memory_used_gb', 0)
            
            return {
                'success': True,
                'response_time': response_time,
                'documents_retrieved': len(documents),
                'avg_similarity_score': avg_similarity,
                'answer_length': len(answer),
                'answer': answer,
                'contexts': contexts,
                'query': query,
                'ground_truth': query_data.get('ground_truth', ''),
                'keywords': query_data.get('keywords', []),
                'category': query_data.get('category', ''),
                'memory_delta_gb': memory_delta,
                'system_before': system_before,
                'system_after': system_after
            }
            
        except Exception as e:
            logger.error(f"❌ {technique_name} failed for query '{query[:50]}...': {e}")
            return {
                'success': False,
                'response_time': time.time() - start_time,
                'documents_retrieved': 0,
                'avg_similarity_score': 0.0,
                'answer_length': 0,
                'answer': '',
                'contexts': [],
                'query': query,
                'ground_truth': query_data.get('ground_truth', ''),
                'keywords': query_data.get('keywords', []),
                'category': query_data.get('category', ''),
                'error': str(e),
                'memory_delta_gb': 0,
                'system_before': system_before,
                'system_after': {}
            }
    
    def evaluate_with_ragas_comprehensive(self, results: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
        """Comprehensive RAGAS evaluation with all available metrics"""
        if not RAGAS_AVAILABLE or not self.real_llm:
            logger.warning("⚠️ RAGAS evaluation requires real LLM and RAGAS installation")
            return None
        
        # Filter successful results with answers
        valid_results = [r for r in results if r['success'] and r['answer'] and r['contexts']]
        
        if not valid_results:
            logger.warning("⚠️ No valid results for RAGAS evaluation")
            return None
        
        try:
            # Prepare data for RAGAS
            data = {
                'question': [r['query'] for r in valid_results],
                'answer': [r['answer'] for r in valid_results],
                'contexts': [r['contexts'] for r in valid_results],
                'ground_truth': [r['ground_truth'] for r in valid_results]
            }
            
            dataset = Dataset.from_dict(data)
            
            # Use all available RAGAS metrics
            metrics = [
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
                answer_similarity,
                answer_correctness,
                context_relevancy
            ]
            
            # Run RAGAS evaluation
            logger.info("🔍 Running comprehensive RAGAS evaluation...")
            ragas_results = evaluate(
                dataset,
                metrics=metrics,
                llm=self.llm_func,
                embeddings=self.embedding_func_ragas
            )
            
            return ragas_results
            
        except Exception as e:
            logger.error(f"❌ RAGAS evaluation failed: {e}")
            traceback.print_exc()
            return None
    
    def run_scaling_evaluation_at_size(self, target_size: int) -> Dict[str, Any]:
        """Run evaluation for all techniques at a specific dataset size"""
        logger.info(f"\n🎯 Running scaling evaluation at {target_size:,} documents...")
        
        # Get current database stats
        db_stats = self.get_database_stats()
        current_size = db_stats['document_count']
        
        logger.info(f"📊 Current database: {current_size:,} documents")
        
        if current_size < target_size:
            logger.warning(f"⚠️ Database has {current_size:,} documents, target is {target_size:,}")
            logger.info("💡 Consider running data ingestion to reach target size")
        
        # Initialize results structure
        evaluation_results = {
            'dataset_size': current_size,
            'target_size': target_size,
            'database_stats': db_stats,
            'system_info': self.get_system_metrics(),
            'techniques': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Test each technique
        for technique_name in self.technique_names:
            logger.info(f"\n📋 Testing {technique_name} at {current_size:,} documents...")
            
            # Initialize pipeline
            pipeline = self.initialize_pipeline(technique_name)
            if not pipeline:
                logger.error(f"❌ Failed to initialize {technique_name}")
                evaluation_results['techniques'][technique_name] = {
                    'success': False,
                    'error': 'Failed to initialize pipeline'
                }
                continue
            
            # Run queries
            technique_results = []
            successful_queries = 0
            total_response_time = 0
            
            for i, query_data in enumerate(self.test_queries):
                logger.info(f"  Query {i+1}/{len(self.test_queries)}: {query_data['query'][:50]}...")
                
                result = self.run_single_query_with_metrics(pipeline, technique_name, query_data)
                technique_results.append(result)
                
                if result['success']:
                    successful_queries += 1
                    total_response_time += result['response_time']
                
                time.sleep(0.5)  # Brief pause between queries
            
            # Calculate aggregate metrics
            successful_results = [r for r in technique_results if r['success']]
            
            if successful_results:
                # Performance metrics
                avg_response_time = np.mean([r['response_time'] for r in successful_results])
                avg_documents = np.mean([r['documents_retrieved'] for r in successful_results])
                avg_similarity = np.mean([r['avg_similarity_score'] for r in successful_results])
                avg_answer_length = np.mean([r['answer_length'] for r in successful_results])
                avg_memory_delta = np.mean([r['memory_delta_gb'] for r in successful_results])
                
                # RAGAS evaluation
                ragas_scores = self.evaluate_with_ragas_comprehensive(successful_results)
                
                evaluation_results['techniques'][technique_name] = {
                    'success': True,
                    'success_rate': successful_queries / len(self.test_queries),
                    'avg_response_time': avg_response_time,
                    'avg_documents_retrieved': avg_documents,
                    'avg_similarity_score': avg_similarity,
                    'avg_answer_length': avg_answer_length,
                    'avg_memory_delta_gb': avg_memory_delta,
                    'ragas_scores': ragas_scores,
                    'individual_results': technique_results
                }
                
                logger.info(f"✅ {technique_name}: {successful_queries}/{len(self.test_queries)} successful")
                logger.info(f"   Avg Response Time: {avg_response_time:.2f}s")
                if ragas_scores:
                    logger.info(f"   RAGAS Scores: {ragas_scores}")
            else:
                logger.error(f"❌ {technique_name}: No successful queries")
                evaluation_results['techniques'][technique_name] = {
                    'success': False,
                    'success_rate': 0,
                    'error': 'No successful queries'
                }
        
        return evaluation_results
    
    def run_complete_scaling_evaluation(self) -> Dict[str, Any]:
        """Run complete scaling evaluation across all dataset sizes"""
        logger.info("🚀 Starting complete scaling evaluation for all 7 RAG techniques...")
        
        scaling_results = {
            'evaluation_plan': {
                'dataset_sizes': self.dataset_sizes,
                'techniques': self.technique_names,
                'test_queries': len(self.test_queries),
                'ragas_metrics': [
                    'answer_relevancy', 'context_precision', 'context_recall',
                    'faithfulness', 'answer_similarity', 'answer_correctness',
                    'context_relevancy'
                ]
            },
            'results_by_size': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Get current database size
        current_db_stats = self.get_database_stats()
        current_size = current_db_stats['document_count']
        
        logger.info(f"📊 Current database size: {current_size:,} documents")
        
        # Find the appropriate size to test based on current database
        test_sizes = [size for size in self.dataset_sizes if size <= current_size]
        if not test_sizes:
            test_sizes = [current_size]  # Test current size if smaller than planned sizes
        
        logger.info(f"🎯 Will test at sizes: {test_sizes}")
        
        # Run evaluation at current size
        for size in test_sizes:
            logger.info(f"\n{'='*60}")
            logger.info(f"🔍 EVALUATING AT {size:,} DOCUMENTS")
            logger.info(f"{'='*60}")
            
            size_results = self.run_scaling_evaluation_at_size(size)
            scaling_results['results_by_size'][str(size)] = size_results
            
            # Save intermediate results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            intermediate_file = f"scaling_evaluation_intermediate_{size}_{timestamp}.json"
            
            with open(intermediate_file, 'w') as f:
                # Convert to serializable format
                serializable_results = self._make_serializable(size_results)
                json.dump(serializable_results, f, indent=2, default=str)
            
            logger.info(f"💾 Intermediate results saved to {intermediate_file}")
        
        # Save final results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_file = f"complete_scaling_evaluation_{timestamp}.json"
        
        with open(final_file, 'w') as f:
            serializable_results = self._make_serializable(scaling_results)
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"💾 Complete scaling evaluation saved to {final_file}")
        
        # Generate comprehensive report
        self.generate_scaling_report(scaling_results, timestamp)
        
        return scaling_results
    
    def _make_serializable(self, data: Any) -> Any:
        """Convert data to JSON-serializable format"""
        if isinstance(data, dict):
            result = {}
            for k, v in data.items():
                if k == 'ragas_scores' and v is not None:
                    result[k] = {key: float(val) for key, val in v.items()}
                else:
                    result[k] = self._make_serializable(v)
            return result
        elif isinstance(data, list):
            return [self._make_serializable(item) for item in data]
        elif isinstance(data, (np.integer, np.floating)):
            return float(data)
        else:
            return data
    
    def generate_scaling_report(self, results: Dict[str, Any], timestamp: str) -> None:
        """Generate comprehensive scaling evaluation report"""
        report_file = f"scaling_evaluation_report_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write("# Comprehensive Scaling Evaluation Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Evaluation overview
            f.write("## Evaluation Overview\n\n")
            plan = results['evaluation_plan']
            f.write(f"- **Techniques Tested:** {len(plan['techniques'])}\n")
            f.write(f"- **Test Queries:** {plan['test_queries']}\n")
            f.write(f"- **RAGAS Metrics:** {', '.join(plan['ragas_metrics'])}\n")
            f.write(f"- **Dataset Sizes:** {', '.join(map(str, plan['dataset_sizes']))}\n\n")
            
            # Results by size
            f.write("## Results by Dataset Size\n\n")
            
            for size_str, size_results in results['results_by_size'].items():
                f.write(f"### {int(size_str):,} Documents\n\n")
                
                # Database stats
                db_stats = size_results['database_stats']
                f.write(f"**Database Statistics:**\n")
                f.write(f"- Documents: {db_stats['document_count']:,}\n")
                f.write(f"- Chunks: {db_stats['chunk_count']:,}\n")
                f.write(f"- Token Embeddings: {db_stats['token_embedding_count']:,}\n")
                f.write(f"- Content Size: {db_stats['content_size_mb']:.1f} MB\n\n")
                
                # Technique performance
                f.write("**Technique Performance:**\n\n")
                f.write("| Technique | Success Rate | Avg Response Time | Avg Documents | RAGAS Score |\n")
                f.write("|-----------|--------------|-------------------|---------------|-------------|\n")
                
                for technique, data in size_results['techniques'].items():
                    if data.get('success', False):
                        success_rate = f"{data['success_rate']*100:.0f}%"
                        response_time = f"{data['avg_response_time']:.2f}s"
                        docs = f"{data['avg_documents_retrieved']:.1f}"
                        
                        # Calculate average RAGAS score
                        ragas_scores = data.get('ragas_scores')
                        if ragas_scores:
                            avg_ragas = np.mean(list(ragas_scores.values()))
                            ragas_str = f"{avg_ragas:.3f}"
                        else:
                            ragas_str = "N/A"
                        
                        f.write(f"| {technique} | {success_rate} | {response_time} | {docs} | {ragas_str} |\n")
                    else:
                        f.write(f"| {technique} | Failed | - | - | - |\n")
                
                f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("### Performance Optimization\n")
            f.write("- Monitor memory usage during scaling\n")
            f.write("- Consider index optimization for larger datasets\n")
            f.write("- Implement query result caching for frequently asked questions\n\n")
            
            f.write("### Quality vs Scale Analysis\n")
            f.write("- Track RAGAS metrics degradation with dataset size\n")
            f.write("- Identify optimal dataset sizes for each technique\n")
            f.write("- Consider technique-specific optimizations\n\n")
        
        logger.info(f"📄 Scaling evaluation report saved to {report_file}")

def main():
    """Main execution function"""
    framework = ScalingEvaluationFramework()
    
    # Run complete scaling evaluation
    results = framework.run_complete_scaling_evaluation()
    
    logger.info("\n🎉 Scaling evaluation complete!")
    logger.info("📊 Check the generated report and JSON files for detailed results")

if __name__ == "__main__":
    main()