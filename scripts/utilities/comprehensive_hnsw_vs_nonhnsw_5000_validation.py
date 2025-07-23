#!/usr/bin/env python3
"""
Comprehensive HNSW vs Non-HNSW Performance Comparison (5000 Documents)

This script runs a comprehensive end-to-end test comparing HNSW vs non-HNSW 
performance across all 7 RAG techniques with 5000 documents and optimal chunking settings.

Objectives:
1. Set up HNSW vector database schema with native VECTOR types
2. Create comprehensive comparison framework for all 7 RAG techniques
3. Run enterprise-scale validation with 5000 real PMC documents
4. Generate comprehensive performance analysis and comparison report
5. Validate chunking integration with both HNSW and non-HNSW approaches
6. Provide definitive, measurable proof of HNSW performance benefits

Features:
- HNSW schema deployment with native VECTOR types and indexes
- Side-by-side comparison of HNSW vs VARCHAR-based vector search
- All 7 RAG techniques tested with both approaches
- Optimal chunking strategy integration (semantic/hybrid)
- Real PMC biomedical data at enterprise scale
- Statistical significance testing with multiple queries
- Comprehensive performance metrics and resource monitoring
- Honest assessment of HNSW benefits vs overhead

Usage:
    python scripts/comprehensive_hnsw_vs_nonhnsw_5000_validation.py
    python scripts/comprehensive_hnsw_vs_nonhnsw_5000_validation.py --skip-setup
    python scripts/comprehensive_hnsw_vs_nonhnsw_5000_validation.py --fast-mode
"""

import os
import sys
import logging
import time
import json
import argparse
import psutil
import numpy as np
import threading
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.iris_connector import get_iris_connection # Updated import
from common.utils import get_embedding_func, get_llm_func, get_colbert_query_encoder_func, get_colbert_doc_encoder_func_adapted # Updated import
from data.loader_fixed import load_documents_to_iris # Path remains correct
from data.pmc_processor import process_pmc_files # Path remains correct

# Import all RAG pipelines
from iris_rag.pipelines.basic import BasicRAGPipeline # Updated import
from iris_rag.pipelines.hyde import HyDERAGPipeline # Updated import
from iris_rag.pipelines.crag import CRAGPipeline # Updated import
from iris_rag.pipelines.colbert import ColBERTRAGPipeline # Updated import
from iris_rag.pipelines.noderag import NodeRAGPipeline # Updated import
from iris_rag.pipelines.graphrag import GraphRAGPipeline # Updated import
from iris_rag.pipelines.hybrid_ifind import HybridIFindRAGPipeline # Updated import

# Import chunking service
from chunking.enhanced_chunking_service import EnhancedChunkingService # Path remains correct

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hnsw_vs_nonhnsw_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for HNSW vs non-HNSW comparison"""
    technique_name: str
    approach: str  # 'hnsw' or 'varchar'
    query_count: int
    success_count: int
    success_rate: float
    avg_response_time_ms: float
    median_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    avg_documents_retrieved: float
    avg_similarity_score: float
    total_execution_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    queries_per_second: float
    error_details: List[str]

@dataclass
class ComparisonResult:
    """Results comparing HNSW vs non-HNSW for a technique"""
    technique_name: str
    hnsw_metrics: PerformanceMetrics
    varchar_metrics: PerformanceMetrics
    speed_improvement_factor: float
    response_time_improvement_ms: float
    retrieval_quality_difference: float
    memory_overhead_mb: float
    statistical_significance: bool
    recommendation: str

class SystemMonitor:
    """Enhanced system monitoring for HNSW comparison"""
    
    def __init__(self):
        self.monitoring = False
        self.metrics = []
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start system monitoring in background thread"""
        self.monitoring = True
        self.metrics = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("🔍 Enhanced system monitoring started for HNSW comparison")
        
    def stop_monitoring(self):
        """Stop system monitoring and return metrics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info(f"📊 System monitoring stopped - collected {len(self.metrics)} data points")
        return self.metrics
        
    def _monitor_loop(self):
        """Background monitoring loop with enhanced metrics"""
        while self.monitoring:
            try:
                memory = psutil.virtual_memory()
                cpu = psutil.cpu_percent(interval=1)
                disk = psutil.disk_usage('/')
                
                # Enhanced metrics for HNSW comparison
                self.metrics.append({
                    'timestamp': time.time(),
                    'memory_used_gb': memory.used / (1024**3),
                    'memory_percent': memory.percent,
                    'memory_available_gb': memory.available / (1024**3),
                    'cpu_percent': cpu,
                    'disk_used_gb': disk.used / (1024**3),
                    'disk_percent': (disk.used / disk.total) * 100,
                    'disk_io_read_mb': psutil.disk_io_counters().read_bytes / (1024**2) if psutil.disk_io_counters() else 0,
                    'disk_io_write_mb': psutil.disk_io_counters().write_bytes / (1024**2) if psutil.disk_io_counters() else 0
                })
                
                # Log critical resource usage
                if memory.percent > 90:
                    logger.warning(f"⚠️ High memory usage: {memory.percent:.1f}%")
                if cpu > 90:
                    logger.warning(f"⚠️ High CPU usage: {cpu:.1f}%")
                    
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                
            time.sleep(2)  # More frequent monitoring for HNSW comparison

class HNSWSchemaManager:
    """Manages HNSW schema deployment and configuration"""

    def __init__(self, connection):
        self.connection = connection

    def deploy_hnsw_schema(self) -> bool:
        """Deploy HNSW schema with native VECTOR types and indexes"""
        logger.info("🚀 Deploying HNSW vector database schema...")
        try:
            cursor = self.connection.cursor()

            # Drop existing schema and tables
            drop_statements = [
                "DROP TABLE IF EXISTS RAG_HNSW.DocumentChunks CASCADE",
                "DROP TABLE IF EXISTS RAG_HNSW.DocumentTokenEmbeddings CASCADE",
                "DROP TABLE IF EXISTS RAG_HNSW.KnowledgeGraphEdges CASCADE",
                "DROP TABLE IF EXISTS RAG_HNSW.KnowledgeGraphNodes CASCADE",
                "DROP TABLE IF EXISTS RAG_HNSW.SourceDocuments CASCADE",
                "DROP SCHEMA IF EXISTS RAG_HNSW CASCADE"
            ]
            for stmt in drop_statements:
                try:
                    cursor.execute(stmt)
                    logger.debug(f"Executed: {stmt}")
                except Exception as e_drop: # nosec
                    logger.debug(f"Could not execute '{stmt}': {e_drop} (might not exist)")
            
            cursor.execute("CREATE SCHEMA RAG_HNSW")
            logger.info("✅ RAG_HNSW schema created.")

            # Create SourceDocuments table
            create_sourcedocs_sql = """
CREATE TABLE RAG_HNSW.SourceDocuments (
    doc_id VARCHAR(255) PRIMARY KEY,
    title VARCHAR(1000),
    text_content CLOB,
    metadata CLOB,
    embedding_model VARCHAR(255),
    embedding_dimensions INTEGER,
    embedding_vector VECTOR(FLOAT, 768),
    embedding_str VARCHAR(60000),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)"""
            cursor.execute(create_sourcedocs_sql.strip())
            logger.info("✅ RAG_HNSW.SourceDocuments table created.")

            # Create additional tables and indexes using helper methods
            self._create_additional_tables(cursor)
            self._create_hnsw_indexes(cursor)
            self._create_standard_indexes(cursor)
            
            self.connection.commit() # Commit all schema changes
            cursor.close()
            logger.info("✅ HNSW schema deployment completed successfully.")
            return True
        except Exception as e:
            logger.error(f"❌ HNSW schema deployment failed: {e}")
            try:
                self.connection.rollback() # Rollback on error
            except Exception as rb_e:
                logger.error(f"Rollback failed: {rb_e}")
            return False

    def _create_additional_tables(self, cursor):
        """Create additional tables for HNSW schema"""
        logger.info("🔧 Creating additional HNSW tables...")
        additional_tables_sql = """
        CREATE TABLE RAG_HNSW.DocumentChunks (
            chunk_id VARCHAR(255) PRIMARY KEY,
            doc_id VARCHAR(255),
            chunk_text CLOB,
            chunk_type VARCHAR(50),
            strategy_name VARCHAR(50),
            start_position INTEGER,
            end_position INTEGER,
            token_count INTEGER,
            embedding_vector VECTOR(FLOAT, 768),
            embedding_str VARCHAR(60000),
            semantic_coherence_score DECIMAL(5,4),
            boundary_strength DECIMAL(5,4),
            biomedical_density DECIMAL(5,4),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (doc_id) REFERENCES RAG_HNSW.SourceDocuments(doc_id)
        );
        
        CREATE TABLE RAG_HNSW.DocumentTokenEmbeddings (
            doc_id VARCHAR(255),
            token_sequence_index INTEGER,
            token_text VARCHAR(255),
            token_embedding_vector VECTOR(FLOAT, 128),
            token_embedding_str VARCHAR(30000),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (doc_id, token_sequence_index),
            FOREIGN KEY (doc_id) REFERENCES RAG_HNSW.SourceDocuments(doc_id)
        );
        
        CREATE TABLE RAG_HNSW.KnowledgeGraphNodes (
            node_id VARCHAR(255) PRIMARY KEY,
            node_name VARCHAR(500),
            node_type VARCHAR(100),
            properties CLOB,
            embedding_vector VECTOR(FLOAT, 768),
            embedding_str VARCHAR(60000),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE RAG_HNSW.KnowledgeGraphEdges (
            edge_id VARCHAR(255),
            source_node_id VARCHAR(255),
            target_node_id VARCHAR(255),
            relationship_type VARCHAR(100),
            relationship_strength DECIMAL(5,4),
            properties CLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (edge_id),
            FOREIGN KEY (source_node_id) REFERENCES RAG_HNSW.KnowledgeGraphNodes(node_id),
            FOREIGN KEY (target_node_id) REFERENCES RAG_HNSW.KnowledgeGraphNodes(node_id)
        );
        """
        for statement in additional_tables_sql.split(';'):
            if statement.strip():
                cursor.execute(statement.strip())
        logger.info("✅ Additional HNSW tables created.")
    
    def _create_hnsw_indexes(self, cursor):
        """Create HNSW indexes on VECTOR columns"""
        logger.info("🔧 Creating HNSW indexes...")
        hnsw_indexes = [
            "CREATE INDEX idx_hnsw_source_embeddings ON RAG_HNSW.SourceDocuments (embedding_vector) AS HNSW(M=16, efConstruction=200, Distance='COSINE')",
            "CREATE INDEX idx_hnsw_chunk_embeddings ON RAG_HNSW.DocumentChunks (embedding_vector) AS HNSW(M=16, efConstruction=200, Distance='COSINE')",
            "CREATE INDEX idx_hnsw_token_embeddings ON RAG_HNSW.DocumentTokenEmbeddings (token_embedding_vector) AS HNSW(M=16, efConstruction=200, Distance='COSINE')",
            "CREATE INDEX idx_hnsw_kg_node_embeddings ON RAG_HNSW.KnowledgeGraphNodes (embedding_vector) AS HNSW(M=16, efConstruction=200, Distance='COSINE')"
        ]
        for index_sql in hnsw_indexes:
            try:
                cursor.execute(index_sql)
                logger.info(f"✅ HNSW index created: {index_sql.splitlines()[1].strip().split()[2]}")
            except Exception as e:
                logger.warning(f"⚠️ HNSW index creation failed for '{index_sql.splitlines()[1].strip().split()[2]}' (may not be supported or already exists): {e}")
    
    def _create_standard_indexes(self, cursor):
        """Create standard indexes for performance optimization"""
        logger.info("🔧 Creating standard performance indexes...")
        standard_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_source_docs_title ON RAG_HNSW.SourceDocuments(title)",
            "CREATE INDEX IF NOT EXISTS idx_source_docs_model ON RAG_HNSW.SourceDocuments(embedding_model)",
            "CREATE INDEX IF NOT EXISTS idx_source_docs_created ON RAG_HNSW.SourceDocuments(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON RAG_HNSW.DocumentChunks(doc_id)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_type ON RAG_HNSW.DocumentChunks(chunk_type)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_strategy ON RAG_HNSW.DocumentChunks(strategy_name)",
            "CREATE INDEX IF NOT EXISTS idx_kg_nodes_type ON RAG_HNSW.KnowledgeGraphNodes(node_type)",
            "CREATE INDEX IF NOT EXISTS idx_kg_nodes_name ON RAG_HNSW.KnowledgeGraphNodes(node_name)",
            "CREATE INDEX IF NOT EXISTS idx_kg_edges_type ON RAG_HNSW.KnowledgeGraphEdges(relationship_type)",
            "CREATE INDEX IF NOT EXISTS idx_token_embeddings_doc ON RAG_HNSW.DocumentTokenEmbeddings(doc_id)"
        ]
        for index_sql in standard_indexes:
            try:
                cursor.execute(index_sql)
                logger.info(f"✅ Standard index processed: {index_sql.split('ON')[0].split()[-1]}") # Log index name
            except Exception as e:
                logger.warning(f"⚠️ Standard index creation/check failed for '{index_sql.split('ON')[0].split()[-1]}': {e}")

class ComprehensiveHNSWComparison:
    """Comprehensive HNSW vs non-HNSW comparison framework"""
    
    def __init__(self, target_docs: int = 5000):
        self.target_docs = target_docs
        self.connection = None
        self.embedding_func = None
        self.llm_func = None
        self.chunking_service = None
        self.schema_manager = None
        self.monitor = SystemMonitor()
        self.results: List[ComparisonResult] = []
        self.start_time = time.time()
        
        # Test queries for statistical significance
        self.test_queries = [
            "diabetes treatment and management strategies in clinical practice",
            "machine learning applications in medical diagnosis and imaging",
            "cancer immunotherapy and personalized medicine approaches",
            "genetic mutations and disease susceptibility analysis",
            "artificial intelligence in healthcare systems and patient care",
            "cardiovascular disease prevention and intervention methods",
            "neurological disorders and brain function research",
            "infectious disease epidemiology and control measures",
            "metabolic syndrome and obesity research findings",
            "respiratory system diseases and treatment protocols",
            "biomarker discovery and validation in precision medicine",
            "drug discovery and pharmaceutical development processes",
            "clinical trial design and statistical analysis methods",
            "medical imaging techniques and diagnostic accuracy",
            "genomics and proteomics in disease research"
        ]
    
    def setup_environment(self) -> bool:
        """Setup complete environment for HNSW comparison"""
        logger.info("🔧 Setting up comprehensive HNSW comparison environment...")
        
        # Setup database connection
        if not self._setup_database():
            return False
        
        # Setup models
        if not self._setup_models():
            return False
        
        # Setup chunking service
        if not self._setup_chunking():
            return False
        
        # Setup HNSW schema manager
        self.schema_manager = HNSWSchemaManager(self.connection)
        
        return True
    
    def _setup_database(self) -> bool:
        """Setup database connection and verify connectivity"""
        try:
            self.connection = get_iris_connection()
            if not self.connection:
                raise Exception("Failed to establish database connection")
            
            cursor = self.connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments_V2")
            current_docs = cursor.fetchone()[0]
            cursor.close()
            
            logger.info(f"✅ Database connected: {current_docs} documents available")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database setup failed: {e}")
            return False
    
    def _setup_models(self) -> bool:
        """Setup embedding and LLM models"""
        try:
            # Setup optimized embedding model
            self.embedding_func = get_embedding_func(
                model_name="intfloat/e5-base-v2", 
                mock=False
            )
            
            # Test embedding
            test_embedding = self.embedding_func(["HNSW performance test"])[0]
            logger.info(f"✅ Embedding model: {len(test_embedding)} dimensions")
            
            # Setup LLM
            self.llm_func = get_llm_func(provider="openai", model_name="gpt-3.5-turbo")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Model setup failed: {e}")
            return False
    
    def _setup_chunking(self) -> bool:
        """Setup enhanced chunking service"""
        try:
            self.chunking_service = EnhancedChunkingService(
                connection=self.connection,
                embedding_func=self.embedding_func
            )
            logger.info("✅ Enhanced chunking service initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Chunking service setup failed: {e}")
            return False
def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Comprehensive HNSW vs non-HNSW Performance Comparison")
    parser.add_argument("--skip-setup", action="store_true", help="Skip HNSW infrastructure setup")
    parser.add_argument("--fast-mode", action="store_true", help="Run with reduced query set for faster testing")
    parser.add_argument("--target-docs", type=int, default=5000, help="Target number of documents to test with")
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting Comprehensive HNSW vs Non-HNSW Performance Comparison")
    logger.info(f"📊 Target documents: {args.target_docs}")
    logger.info(f"⚡ Fast mode: {args.fast_mode}")
    logger.info(f"⏭️ Skip setup: {args.skip_setup}")
    
    # Initialize comparison framework
    comparison = ComprehensiveHNSWComparison(target_docs=args.target_docs)
    
    try:
        # Setup environment
        if not comparison.setup_environment():
            logger.error("❌ Environment setup failed")
            return 1
        
        # Deploy HNSW infrastructure
        if not comparison.deploy_hnsw_infrastructure(skip_setup=args.skip_setup):
            logger.error("❌ HNSW infrastructure deployment failed")
            return 1
        
        # Run comprehensive comparison
        if not comparison.run_comprehensive_comparison(fast_mode=args.fast_mode):
            logger.error("❌ Comprehensive comparison failed")
            return 1
        
        # Generate report
        results_file = comparison.generate_comprehensive_report()
        
        # Print summary
        logger.info("🎉 COMPREHENSIVE HNSW VS NON-HNSW COMPARISON COMPLETED!")
        logger.info(f"📊 Results saved to: {results_file}")
        logger.info(f"🔬 Techniques tested: {len(comparison.results)}")
        
        # Print quick summary
        if comparison.results:
            hnsw_advantages = len([r for r in comparison.results if r.speed_improvement_factor > 1.1])
            logger.info(f"✅ Techniques with HNSW advantage: {hnsw_advantages}/{len(comparison.results)}")
            
            best_improvement = max(comparison.results, key=lambda x: x.speed_improvement_factor)
            logger.info(f"🏆 Best HNSW improvement: {best_improvement.technique_name} ({best_improvement.speed_improvement_factor:.2f}x faster)")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Comprehensive comparison failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())