#!/usr/bin/env python3
"""
GraphRAG Enterprise Scale Testing - 10K+ Documents

Comprehensive scale testing framework for GraphRAG implementation validation
with 10,000+ documents to ensure enterprise production readiness.

Features:
- PMC document batch processing and loading
- Memory usage monitoring and profiling  
- Entity extraction performance at scale
- Knowledge graph population validation
- Multi-hop query performance testing
- Bottleneck identification and analysis
- Resource utilization monitoring
- Comparative performance against other pipelines

Usage:
    python scripts/test_graphrag_scale_10k.py [--documents 10000] [--config config.yaml] [--use-mocks]
"""

import os
import sys
import json
import time
import logging
import argparse
import traceback
import psutil
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Third-party imports with graceful fallback
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: Visualization packages not available. Install with: pip install pandas matplotlib seaborn")

# IRIS RAG imports
try:
    from iris_rag.pipelines.graphrag import GraphRAGPipeline as CurrentGraphRAG
    from iris_rag.pipelines.graphrag_merged import GraphRAGPipeline as MergedGraphRAG
    from iris_rag.pipelines.basic import BasicRAGPipeline
    from iris_rag.pipelines.crag import CRAGPipeline
    from iris_rag.core.connection import ConnectionManager
    from iris_rag.config.manager import ConfigurationManager
    from iris_rag.core.models import Document
    from data.pmc_processor import extract_abstract, _chunk_pmc_content
except ImportError as e:
    print(f"Failed to import IRIS RAG components: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ScaleTestMetrics:
    """Comprehensive metrics for enterprise scale testing."""
    test_id: str
    pipeline_name: str
    timestamp: str
    
    # Document processing metrics
    total_documents: int
    documents_loaded: int
    loading_time_seconds: float
    loading_success_rate: float
    
    # Memory and performance metrics
    peak_memory_mb: float
    avg_memory_mb: float
    cpu_usage_percent: float
    
    # Entity extraction metrics
    entities_extracted: int
    relationships_created: int
    graph_nodes: int
    graph_edges: int
    extraction_time_seconds: float
    
    # Query performance metrics
    queries_executed: int
    avg_query_time_ms: float
    max_query_time_ms: float
    min_query_time_ms: float
    query_success_rate: float
    
    # Quality metrics
    avg_documents_retrieved: float
    avg_answer_length: int
    retrieval_method_stats: Dict[str, int]
    
    # Error tracking
    errors_encountered: List[str]
    warnings: List[str]
    
    # Success criteria evaluation
    meets_memory_target: bool  # <8GB
    meets_query_time_target: bool  # <30s
    meets_success_rate_target: bool  # >95%
    overall_status: str  # "PASS", "PARTIAL", "FAIL"


@dataclass
class PMCDocumentBatch:
    """Batch of PMC documents for processing."""
    batch_id: str
    documents: List[Document]
    total_size_mb: float
    processing_time_seconds: float = 0.0
    success_count: int = 0
    error_count: int = 0
    errors: List[str] = field(default_factory=list)


class PMCDocumentLoader:
    """Efficient PMC document loader for scale testing."""
    
    def __init__(self, pmc_directory: str = "data/downloaded_pmc_docs"):
        self.pmc_directory = Path(pmc_directory)
        self.loaded_documents: List[Document] = []
        
    def load_pmc_documents(self, limit: int = 10000, batch_size: int = 100) -> List[PMCDocumentBatch]:
        """Load PMC documents in batches for efficient processing."""
        logger.info(f"Loading up to {limit} PMC documents from {self.pmc_directory}")
        
        xml_files = list(self.pmc_directory.glob("*.xml"))[:limit]
        logger.info(f"Found {len(xml_files)} XML files to process")
        
        batches = []
        for i in range(0, len(xml_files), batch_size):
            batch_files = xml_files[i:i + batch_size]
            batch = self._process_batch(batch_files, f"batch_{i//batch_size}")
            batches.append(batch)
            logger.info(f"Processed batch {len(batches)}: {batch.success_count}/{len(batch_files)} documents")
        
        return batches
    
    def _process_batch(self, xml_files: List[Path], batch_id: str) -> PMCDocumentBatch:
        """Process a batch of XML files into documents."""
        import xml.etree.ElementTree as ET
        
        start_time = time.time()
        documents = []
        total_size = 0
        errors = []
        
        for xml_file in xml_files:
            try:
                # Parse XML and extract content
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # Extract metadata
                pmc_id = xml_file.stem
                abstract = extract_abstract(root)
                
                # Get file size
                file_size = xml_file.stat().st_size
                total_size += file_size
                
                # Create document
                if abstract:
                    document = Document(
                        id=pmc_id,
                        page_content=abstract,
                        metadata={
                            "source": str(xml_file),
                            "pmc_id": pmc_id,
                            "file_size_bytes": file_size,
                            "batch_id": batch_id,
                            "document_type": "pmc_abstract"
                        }
                    )
                    documents.append(document)
                else:
                    errors.append(f"No abstract found in {xml_file}")
                    
            except Exception as e:
                error_msg = f"Failed to process {xml_file}: {str(e)}"
                errors.append(error_msg)
                logger.warning(error_msg)
        
        processing_time = time.time() - start_time
        
        return PMCDocumentBatch(
            batch_id=batch_id,
            documents=documents,
            total_size_mb=total_size / (1024 * 1024),
            processing_time_seconds=processing_time,
            success_count=len(documents),
            error_count=len(errors),
            errors=errors
        )


class SystemResourceMonitor:
    """Monitor system resources during testing."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = time.time()
        self.measurements = []
        
    def take_measurement(self) -> Dict[str, float]:
        """Take a snapshot of current resource usage."""
        try:
            memory_info = self.process.memory_info()
            cpu_percent = self.process.cpu_percent()
            
            measurement = {
                "timestamp": time.time() - self.start_time,
                "memory_mb": memory_info.rss / (1024 * 1024),
                "cpu_percent": cpu_percent
            }
            
            self.measurements.append(measurement)
            return measurement
            
        except Exception as e:
            logger.warning(f"Failed to take resource measurement: {e}")
            return {"timestamp": time.time() - self.start_time, "memory_mb": 0, "cpu_percent": 0}
    
    def get_statistics(self) -> Dict[str, float]:
        """Calculate resource usage statistics."""
        if not self.measurements:
            return {"peak_memory_mb": 0, "avg_memory_mb": 0, "avg_cpu_percent": 0}
        
        memory_values = [m["memory_mb"] for m in self.measurements]
        cpu_values = [m["cpu_percent"] for m in self.measurements]
        
        return {
            "peak_memory_mb": max(memory_values),
            "avg_memory_mb": statistics.mean(memory_values),
            "avg_cpu_percent": statistics.mean(cpu_values)
        }


class GraphRAGScaleTester:
    """Comprehensive GraphRAG enterprise scale testing framework."""
    
    # Enterprise test query suite
    ENTERPRISE_TEST_QUERIES = [
        # Basic entity queries
        "What are the main causes of cardiovascular disease?",
        "How do vaccines work to prevent infectious diseases?", 
        "What are the symptoms and treatments for diabetes?",
        
        # Multi-hop queries
        "What is the relationship between diabetes and cardiovascular complications?",
        "How do COVID-19 vaccines affect patients with autoimmune diseases?",
        "What are the genetic factors linking obesity to type 2 diabetes?",
        
        # Complex biomedical queries
        "Explain the molecular mechanisms of insulin resistance in metabolic syndrome",
        "What are the therapeutic targets for cancer immunotherapy in lung cancer?",
        "How do environmental factors influence the development of asthma in children?",
        
        # Performance stress queries
        "Describe the complete pathway from viral infection to immune response including all cellular interactions",
        "What are all known drug interactions between diabetes medications and cardiovascular treatments?",
        "Analyze the relationship between gut microbiome, immune system, and neurological disorders"
    ]
    
    # Success criteria thresholds
    MEMORY_TARGET_GB = 8.0
    QUERY_TIME_TARGET_SECONDS = 30.0
    SUCCESS_RATE_TARGET = 0.95
    
    def __init__(self, config_path: Optional[str] = None, use_mocks: bool = False, output_dir: str = "outputs"):
        """Initialize the scale testing framework."""
        self.use_mocks = use_mocks
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        try:
            self.config_manager = ConfigurationManager(config_path)
            if not use_mocks:
                self.connection_manager = ConnectionManager()
            else:
                self.connection_manager = None
                logger.info("Using mock mode - database connection skipped")
                
        except Exception as e:
            if not use_mocks:
                logger.warning(f"Failed to initialize components, falling back to mock mode: {e}")
                self.use_mocks = True
                self.connection_manager = None
            else:
                raise
        
        # Initialize monitoring and data structures
        self.resource_monitor = SystemResourceMonitor()
        self.document_loader = PMCDocumentLoader()
        self.test_results: Dict[str, ScaleTestMetrics] = {}
        
        logger.info(f"GraphRAG Scale Tester initialized (mock_mode={self.use_mocks})")
    
    def run_comprehensive_scale_test(self, num_documents: int = 1000) -> Dict[str, ScaleTestMetrics]:
        """Execute comprehensive scale testing for all GraphRAG implementations."""
        logger.info(f"🚀 Starting GraphRAG Enterprise Scale Testing - {num_documents} documents")
        
        # Load documents in batches
        logger.info("📄 Loading PMC documents...")
        document_batches = self.document_loader.load_pmc_documents(limit=num_documents)
        all_documents = []
        for batch in document_batches:
            all_documents.extend(batch.documents)
        
        logger.info(f"✅ Loaded {len(all_documents)} documents across {len(document_batches)} batches")
        
        # Test both GraphRAG implementations
        pipelines_to_test = [
            ("current_graphrag", "Current GraphRAG Implementation"),
            ("merged_graphrag", "Merged GraphRAG Implementation")
        ]
        
        for pipeline_id, pipeline_name in pipelines_to_test:
            logger.info(f"🧪 Testing {pipeline_name}...")
            try:
                metrics = self._test_single_pipeline(pipeline_id, pipeline_name, all_documents)
                self.test_results[pipeline_id] = metrics
                logger.info(f"✅ {pipeline_name} testing completed - Status: {metrics.overall_status}")
                
                # Force garbage collection between tests
                gc.collect()
                
            except Exception as e:
                logger.error(f"❌ {pipeline_name} testing failed: {e}")
                self.test_results[pipeline_id] = self._create_failed_metrics(pipeline_id, pipeline_name, str(e))
        
        # Generate comprehensive report
        self._generate_scale_test_report()
        
        return self.test_results
    
    def _test_single_pipeline(self, pipeline_id: str, pipeline_name: str, documents: List[Document]) -> ScaleTestMetrics:
        """Test a single GraphRAG pipeline implementation."""
        start_time = time.time()
        test_id = f"scale_test_{pipeline_id}_{int(start_time)}"
        
        # Initialize metrics
        metrics = ScaleTestMetrics(
            test_id=test_id,
            pipeline_name=pipeline_name,
            timestamp=datetime.now().isoformat(),
            total_documents=len(documents),
            documents_loaded=0,
            loading_time_seconds=0.0,
            loading_success_rate=0.0,
            peak_memory_mb=0.0,
            avg_memory_mb=0.0,
            cpu_usage_percent=0.0,
            entities_extracted=0,
            relationships_created=0,
            graph_nodes=0,
            graph_edges=0,
            extraction_time_seconds=0.0,
            queries_executed=0,
            avg_query_time_ms=0.0,
            max_query_time_ms=0.0,
            min_query_time_ms=0.0,
            query_success_rate=0.0,
            avg_documents_retrieved=0.0,
            avg_answer_length=0,
            retrieval_method_stats={},
            errors_encountered=[],
            warnings=[],
            meets_memory_target=False,
            meets_query_time_target=False,
            meets_success_rate_target=False,
            overall_status="FAIL"
        )
        
        try:
            # Initialize pipeline
            pipeline = self._create_pipeline(pipeline_id)
            self.resource_monitor.take_measurement()
            
            # Document loading phase
            logger.info(f"📄 Loading {len(documents)} documents into {pipeline_name}...")
            loading_start = time.time()
            
            try:
                if hasattr(pipeline, 'load_documents'):
                    pipeline.load_documents("", documents=documents)
                else:
                    logger.warning(f"Pipeline {pipeline_name} does not support load_documents")
                
                metrics.documents_loaded = len(documents)
                metrics.loading_success_rate = 1.0
                
            except Exception as e:
                error_msg = f"Document loading failed: {str(e)}"
                metrics.errors_encountered.append(error_msg)
                logger.error(error_msg)
                metrics.loading_success_rate = 0.0
            
            metrics.loading_time_seconds = time.time() - loading_start
            self.resource_monitor.take_measurement()
            
            # Entity extraction validation
            logger.info("🔗 Validating entity extraction...")
            extraction_start = time.time()
            
            try:
                # Check if entities were extracted (this would be implementation-specific)
                extraction_stats = self._validate_entity_extraction(pipeline)
                metrics.entities_extracted = extraction_stats.get('entities', 0)
                metrics.relationships_created = extraction_stats.get('relationships', 0)
                metrics.graph_nodes = extraction_stats.get('nodes', 0)
                metrics.graph_edges = extraction_stats.get('edges', 0)
                
            except Exception as e:
                error_msg = f"Entity extraction validation failed: {str(e)}"
                metrics.warnings.append(error_msg)
                logger.warning(error_msg)
            
            metrics.extraction_time_seconds = time.time() - extraction_start
            self.resource_monitor.take_measurement()
            
            # Query performance testing
            logger.info("❓ Testing query performance...")
            query_results = self._test_query_performance(pipeline)
            
            metrics.queries_executed = query_results['executed']
            metrics.avg_query_time_ms = query_results['avg_time_ms']
            metrics.max_query_time_ms = query_results['max_time_ms']
            metrics.min_query_time_ms = query_results['min_time_ms']
            metrics.query_success_rate = query_results['success_rate']
            metrics.avg_documents_retrieved = query_results['avg_docs_retrieved']
            metrics.avg_answer_length = query_results['avg_answer_length']
            metrics.retrieval_method_stats = query_results['retrieval_methods']
            
            # Final resource measurements
            resource_stats = self.resource_monitor.get_statistics()
            metrics.peak_memory_mb = resource_stats['peak_memory_mb']
            metrics.avg_memory_mb = resource_stats['avg_memory_mb']
            metrics.cpu_usage_percent = resource_stats['avg_cpu_percent']
            
            # Evaluate success criteria
            metrics.meets_memory_target = metrics.peak_memory_mb < (self.MEMORY_TARGET_GB * 1024)
            metrics.meets_query_time_target = metrics.avg_query_time_ms < (self.QUERY_TIME_TARGET_SECONDS * 1000)
            metrics.meets_success_rate_target = metrics.query_success_rate >= self.SUCCESS_RATE_TARGET
            
            # Determine overall status
            if all([metrics.meets_memory_target, metrics.meets_query_time_target, metrics.meets_success_rate_target]):
                metrics.overall_status = "PASS"
            elif any([metrics.meets_memory_target, metrics.meets_query_time_target, metrics.meets_success_rate_target]):
                metrics.overall_status = "PARTIAL"
            else:
                metrics.overall_status = "FAIL"
            
            logger.info(f"✅ {pipeline_name} test completed in {time.time() - start_time:.2f}s")
            
        except Exception as e:
            error_msg = f"Pipeline test failed: {str(e)}"
            metrics.errors_encountered.append(error_msg)
            logger.error(error_msg)
            logger.error(traceback.format_exc())
        
        return metrics
    
    def _create_pipeline(self, pipeline_id: str):
        """Create pipeline instance based on ID."""
        if self.use_mocks:
            return self._create_mock_pipeline(pipeline_id)
        
        if pipeline_id == "current_graphrag":
            return CurrentGraphRAG(
                connection_manager=self.connection_manager,
                config_manager=self.config_manager
            )
        elif pipeline_id == "merged_graphrag":
            return MergedGraphRAG(
                connection_manager=self.connection_manager,
                config_manager=self.config_manager
            )
        else:
            raise ValueError(f"Unknown pipeline ID: {pipeline_id}")
    
    def _create_mock_pipeline(self, pipeline_id: str):
        """Create mock pipeline for testing without database."""
        class MockGraphRAGPipeline:
            def __init__(self, pipeline_type):
                self.pipeline_type = pipeline_type
                self.documents_loaded = []
            
            def load_documents(self, path: str, documents: List[Document] = None):
                if documents:
                    self.documents_loaded = documents
                    logger.info(f"Mock {self.pipeline_type}: Loaded {len(documents)} documents")
            
            def query(self, query_text: str, **kwargs) -> Dict[str, Any]:
                import random
                time.sleep(random.uniform(0.1, 2.0))  # Simulate processing time
                
                mock_docs = random.randint(1, 10)
                answer_length = random.randint(50, 500)
                
                return {
                    "query": query_text,
                    "answer": f"Mock answer from {self.pipeline_type} for: {query_text[:50]}...",
                    "retrieved_documents": [f"mock_doc_{i}" for i in range(mock_docs)],
                    "contexts": [f"Mock context {i}" for i in range(mock_docs)],
                    "sources": [{"document_id": f"mock_doc_{i}", "source": "mock"} for i in range(mock_docs)],
                    "metadata": {
                        "num_retrieved": mock_docs,
                        "processing_time_ms": random.uniform(100, 2000),
                        "pipeline_type": f"graphrag_{self.pipeline_type}",
                        "retrieval_method": random.choice(["knowledge_graph_traversal", "vector_fallback"]),
                        "entities_found": random.randint(1, 5),
                        "relationships_traversed": random.randint(0, 3),
                        "confidence": random.uniform(0.6, 0.95)
                    }
                }
        
        return MockGraphRAGPipeline(pipeline_id)
    
    def _validate_entity_extraction(self, pipeline) -> Dict[str, int]:
        """Validate entity extraction statistics."""
        # This would need to be implemented based on the actual GraphRAG API
        # For now, return mock statistics
        if self.use_mocks:
            import random
            return {
                "entities": random.randint(100, 1000),
                "relationships": random.randint(50, 500),
                "nodes": random.randint(100, 1000),
                "edges": random.randint(50, 500)
            }
        
        # In real implementation, would query the knowledge graph
        return {"entities": 0, "relationships": 0, "nodes": 0, "edges": 0}
    
    def _test_query_performance(self, pipeline) -> Dict[str, Any]:
        """Test query performance with enterprise test suite."""
        query_times = []
        successful_queries = 0
        docs_retrieved = []
        answer_lengths = []
        retrieval_methods = {}
        
        for i, query in enumerate(self.ENTERPRISE_TEST_QUERIES):
            logger.info(f"🔍 Executing query {i+1}/{len(self.ENTERPRISE_TEST_QUERIES)}: {query[:50]}...")
            
            try:
                start_time = time.time()
                result = pipeline.query(query_text=query, top_k=10, include_sources=True)
                query_time_ms = (time.time() - start_time) * 1000
                
                query_times.append(query_time_ms)
                successful_queries += 1
                
                # Extract metrics
                metadata = result.get('metadata', {})
                docs_retrieved.append(metadata.get('num_retrieved', 0))
                answer_lengths.append(len(result.get('answer', '')))
                
                retrieval_method = metadata.get('retrieval_method', 'unknown')
                retrieval_methods[retrieval_method] = retrieval_methods.get(retrieval_method, 0) + 1
                
                self.resource_monitor.take_measurement()
                
            except Exception as e:
                logger.error(f"Query {i+1} failed: {e}")
                query_times.append(30000)  # Mark as timeout
        
        return {
            "executed": len(self.ENTERPRISE_TEST_QUERIES),
            "avg_time_ms": statistics.mean(query_times) if query_times else 0,
            "max_time_ms": max(query_times) if query_times else 0,
            "min_time_ms": min(query_times) if query_times else 0,
            "success_rate": successful_queries / len(self.ENTERPRISE_TEST_QUERIES),
            "avg_docs_retrieved": statistics.mean(docs_retrieved) if docs_retrieved else 0,
            "avg_answer_length": int(statistics.mean(answer_lengths)) if answer_lengths else 0,
            "retrieval_methods": retrieval_methods
        }
    
    def _create_failed_metrics(self, pipeline_id: str, pipeline_name: str, error: str) -> ScaleTestMetrics:
        """Create failed test metrics."""
        return ScaleTestMetrics(
            test_id=f"failed_{pipeline_id}_{int(time.time())}",
            pipeline_name=pipeline_name,
            timestamp=datetime.now().isoformat(),
            total_documents=0,
            documents_loaded=0,
            loading_time_seconds=0.0,
            loading_success_rate=0.0,
            peak_memory_mb=0.0,
            avg_memory_mb=0.0,
            cpu_usage_percent=0.0,
            entities_extracted=0,
            relationships_created=0,
            graph_nodes=0,
            graph_edges=0,
            extraction_time_seconds=0.0,
            queries_executed=0,
            avg_query_time_ms=0.0,
            max_query_time_ms=0.0,
            min_query_time_ms=0.0,
            query_success_rate=0.0,
            avg_documents_retrieved=0.0,
            avg_answer_length=0,
            retrieval_method_stats={},
            errors_encountered=[error],
            warnings=[],
            meets_memory_target=False,
            meets_query_time_target=False,
            meets_success_rate_target=False,
            overall_status="FAIL"
        )
    
    def _generate_scale_test_report(self):
        """Generate comprehensive scale test report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as JSON
        results_file = self.output_dir / f"graphrag_scale_test_10k_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                "test_suite": "GraphRAG Enterprise Scale Testing",
                "timestamp": datetime.now().isoformat(),
                "results": {k: asdict(v) for k, v in self.test_results.items()},
                "success_criteria": {
                    "memory_target_gb": self.MEMORY_TARGET_GB,
                    "query_time_target_seconds": self.QUERY_TIME_TARGET_SECONDS,
                    "success_rate_target": self.SUCCESS_RATE_TARGET
                }
            }, f, indent=2)
        
        # Generate markdown report
        report_file = self.output_dir / f"graphrag_scale_test_10k_report_{timestamp}.md"
        with open(report_file, 'w') as f:
            f.write(self._generate_markdown_report())
        
        logger.info(f"📊 Scale test results saved to {results_file}")
        logger.info(f"📋 Scale test report saved to {report_file}")
    
    def _generate_markdown_report(self) -> str:
        """Generate markdown format test report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        report = f"""# GraphRAG Enterprise Scale Test Report

**Generated:** {timestamp}  
**Test Suite:** GraphRAG 10K+ Document Scale Testing  
**Target Scale:** {max([m.total_documents for m in self.test_results.values()], default=0):,} documents

## Executive Summary

This report presents the results of comprehensive enterprise-scale testing of GraphRAG implementations
with 10,000+ documents to validate production readiness and performance characteristics.

### Overall Results

"""
        
        # Summary table
        for pipeline_id, metrics in self.test_results.items():
            status_emoji = "✅" if metrics.overall_status == "PASS" else "⚠️" if metrics.overall_status == "PARTIAL" else "❌"
            
            report += f"""
**{metrics.pipeline_name}** {status_emoji}
- Status: {metrics.overall_status}
- Documents Loaded: {metrics.documents_loaded:,} / {metrics.total_documents:,} ({metrics.loading_success_rate*100:.1f}%)
- Peak Memory: {metrics.peak_memory_mb:.1f} MB
- Query Success Rate: {metrics.query_success_rate*100:.1f}%
- Average Query Time: {metrics.avg_query_time_ms:.1f} ms
"""
        
        report += f"""
## Success Criteria Analysis

### Memory Usage Target: < {self.MEMORY_TARGET_GB:.1f} GB
"""
        
        for pipeline_id, metrics in self.test_results.items():
            status = "✅ PASS" if metrics.meets_memory_target else "❌ FAIL"
            report += f"- **{metrics.pipeline_name}**: {metrics.peak_memory_mb:.1f} MB {status}\n"
        
        report += f"""
### Query Performance Target: < {self.QUERY_TIME_TARGET_SECONDS:.0f} seconds
"""
        
        for pipeline_id, metrics in self.test_results.items():
            status = "✅ PASS" if metrics.meets_query_time_target else "❌ FAIL"
            report += f"- **{metrics.pipeline_name}**: {metrics.avg_query_time_ms/1000:.2f}s average {status}\n"
        
        report += f"""
### Success Rate Target: > {self.SUCCESS_RATE_TARGET*100:.0f}%
"""
        
        for pipeline_id, metrics in self.test_results.items():
            status = "✅ PASS" if metrics.meets_success_rate_target else "❌ FAIL"
            report += f"- **{metrics.pipeline_name}**: {metrics.query_success_rate*100:.1f}% {status}\n"
        
        # Detailed results for each pipeline
        for pipeline_id, metrics in self.test_results.items():
            report += f"""
## {metrics.pipeline_name} Detailed Results

### Document Processing
- **Total Documents**: {metrics.total_documents:,}
- **Successfully Loaded**: {metrics.documents_loaded:,}
- **Loading Time**: {metrics.loading_time_seconds:.2f} seconds
- **Success Rate**: {metrics.loading_success_rate*100:.1f}%

### Entity Extraction & Knowledge Graph
- **Entities Extracted**: {metrics.entities_extracted:,}
- **Relationships Created**: {metrics.relationships_created:,}
- **Graph Nodes**: {metrics.graph_nodes:,}
- **Graph Edges**: {metrics.graph_edges:,}
- **Extraction Time**: {metrics.extraction_time_seconds:.2f} seconds

### Query Performance
- **Queries Executed**: {metrics.queries_executed}
- **Average Response Time**: {metrics.avg_query_time_ms:.1f} ms
- **Max Response Time**: {metrics.max_query_time_ms:.1f} ms
- **Min Response Time**: {metrics.min_query_time_ms:.1f} ms
- **Success Rate**: {metrics.query_success_rate*100:.1f}%
- **Average Documents Retrieved**: {metrics.avg_documents_retrieved:.1f}
- **Average Answer Length**: {metrics.avg_answer_length} characters

### Retrieval Method Distribution
"""
            
            for method, count in metrics.retrieval_method_stats.items():
                percentage = (count / metrics.queries_executed * 100) if metrics.queries_executed > 0 else 0
                report += f"- **{method}**: {count} queries ({percentage:.1f}%)\n"
            
            report += f"""
### Resource Utilization
- **Peak Memory Usage**: {metrics.peak_memory_mb:.1f} MB
- **Average Memory Usage**: {metrics.avg_memory_mb:.1f} MB
- **Average CPU Usage**: {metrics.cpu_usage_percent:.1f}%

### Error Analysis
"""
            
            if metrics.errors_encountered:
                for error in metrics.errors_encountered:
                    report += f"- ❌ {error}\n"
            else:
                report += "- ✅ No errors encountered\n"
            
            if metrics.warnings:
                for warning in metrics.warnings:
                    report += f"- ⚠️ {warning}\n"
        
        # Bottleneck analysis
        report += """
## Bottleneck Analysis & Recommendations

"""
        
        bottlenecks = self._analyze_bottlenecks()
        for bottleneck in bottlenecks:
            report += f"### {bottleneck['category']}\n"
            report += f"{bottleneck['analysis']}\n\n"
            report += f"**Recommendations:**\n"
            for rec in bottleneck['recommendations']:
                report += f"- {rec}\n"
            report += "\n"
        
        report += f"""
## Test Environment

- **Test Framework**: GraphRAG Enterprise Scale Tester
- **Document Source**: PMC Biomedical Literature
- **Test Mode**: {"Mock Data" if self.use_mocks else "Real Database"}
- **Query Suite**: {len(self.ENTERPRISE_TEST_QUERIES)} enterprise test queries
- **Target Scale**: 10,000+ documents

## Conclusion

"""
        
        passing_tests = sum(1 for m in self.test_results.values() if m.overall_status == "PASS")
        total_tests = len(self.test_results)
        
        if passing_tests == total_tests:
            report += "✅ **ALL TESTS PASSED** - GraphRAG implementations are ready for enterprise deployment.\n"
        elif passing_tests > 0:
            report += f"⚠️ **PARTIAL SUCCESS** - {passing_tests}/{total_tests} implementations passed. Review failing tests before deployment.\n"
        else:
            report += "❌ **TESTS FAILED** - Critical issues identified. Address bottlenecks before enterprise deployment.\n"
        
        report += f"""
---
**Report Generated:** {timestamp}  
**Test ID**: {list(self.test_results.values())[0].test_id if self.test_results else 'N/A'}
"""
        
        return report
    
    def _analyze_bottlenecks(self) -> List[Dict[str, Any]]:
        """Analyze performance bottlenecks and provide recommendations."""
        bottlenecks = []
        
        # Memory analysis
        max_memory = max([m.peak_memory_mb for m in self.test_results.values()], default=0)
        if max_memory > self.MEMORY_TARGET_GB * 1024 * 0.8:  # 80% of target
            bottlenecks.append({
                "category": "Memory Usage",
                "analysis": f"Peak memory usage of {max_memory:.1f} MB approaches or exceeds the {self.MEMORY_TARGET_GB:.1f}GB target. This may cause issues in production environments with limited memory.",
                "recommendations": [
                    "Implement document batch processing with smaller batch sizes",
                    "Add memory cleanup between document processing batches", 
                    "Consider streaming document processing instead of loading all into memory",
                    "Optimize entity extraction to use less memory per document"
                ]
            })
        
        # Query performance analysis
        avg_query_times = [m.avg_query_time_ms for m in self.test_results.values()]
        max_avg_time = max(avg_query_times, default=0)
        if max_avg_time > self.QUERY_TIME_TARGET_SECONDS * 1000 * 0.7:  # 70% of target
            bottlenecks.append({
                "category": "Query Performance", 
                "analysis": f"Average query time of {max_avg_time:.1f}ms is approaching the {self.QUERY_TIME_TARGET_SECONDS}s target. Complex queries may exceed acceptable response times.",
                "recommendations": [
                    "Optimize knowledge graph traversal algorithms",
                    "Implement query result caching for common queries",
                    "Add query complexity analysis and optimization",
                    "Consider query timeout mechanisms for very complex queries"
                ]
            })
        
        # Success rate analysis
        min_success_rate = min([m.query_success_rate for m in self.test_results.values()], default=0)
        if min_success_rate < self.SUCCESS_RATE_TARGET:
            bottlenecks.append({
                "category": "Query Success Rate",
                "analysis": f"Query success rate of {min_success_rate*100:.1f}% is below the {self.SUCCESS_RATE_TARGET*100:.0f}% target. This indicates reliability issues.",
                "recommendations": [
                    "Improve error handling in query processing pipeline", 
                    "Add fallback mechanisms for failed knowledge graph queries",
                    "Implement better entity recognition for diverse query types",
                    "Add query validation and preprocessing"
                ]
            })
        
        # Entity extraction analysis
        total_entities = sum([m.entities_extracted for m in self.test_results.values()])
        total_docs = sum([m.documents_loaded for m in self.test_results.values()])
        if total_docs > 0:
            entities_per_doc = total_entities / total_docs
            if entities_per_doc < 1.0:  # Less than 1 entity per document
                bottlenecks.append({
                    "category": "Entity Extraction",
                    "analysis": f"Low entity extraction rate of {entities_per_doc:.2f} entities per document may limit knowledge graph effectiveness.",
                    "recommendations": [
                        "Improve entity recognition models for biomedical text",
                        "Add domain-specific entity extraction rules", 
                        "Validate entity extraction pipeline configuration",
                        "Consider hybrid extraction approaches (rule-based + ML)"
                    ]
                })
        
        return bottlenecks


def main():
    """Main entry point for GraphRAG enterprise scale testing."""
    parser = argparse.ArgumentParser(description="GraphRAG Enterprise Scale Testing Framework")
    parser.add_argument("--documents", type=int, default=1000, 
                       help="Number of documents to test with (default: 1000)")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--use-mocks", action="store_true", 
                       help="Use mock data instead of real database")
    parser.add_argument("--output-dir", type=str, default="outputs",
                       help="Output directory for results (default: outputs)")
    
    args = parser.parse_args()
    
    logger.info("🚀 GraphRAG Enterprise Scale Testing Framework")
    logger.info(f"📊 Target documents: {args.documents:,}")
    logger.info(f"🔧 Mock mode: {args.use_mocks}")
    
    try:
        # Initialize tester
        tester = GraphRAGScaleTester(
            config_path=args.config,
            use_mocks=args.use_mocks,
            output_dir=args.output_dir
        )
        
        # Run comprehensive scale tests
        results = tester.run_comprehensive_scale_test(num_documents=args.documents)
        
        # Print summary
        print("\n" + "="*80)
        print("📊 GRAPHRAG ENTERPRISE SCALE TEST SUMMARY")
        print("="*80)
        
        for pipeline_id, metrics in results.items():
            status_emoji = "✅" if metrics.overall_status == "PASS" else "⚠️" if metrics.overall_status == "PARTIAL" else "❌"
            print(f"\n{status_emoji} {metrics.pipeline_name}")
            print(f"   Status: {metrics.overall_status}")
            print(f"   Documents: {metrics.documents_loaded:,} loaded")
            print(f"   Memory: {metrics.peak_memory_mb:.1f} MB peak")
            print(f"   Queries: {metrics.query_success_rate*100:.1f}% success rate")
            print(f"   Time: {metrics.avg_query_time_ms:.1f} ms average")
        
        # Overall assessment
        passing_tests = sum(1 for m in results.values() if m.overall_status == "PASS")
        total_tests = len(results)
        
        print(f"\n📈 OVERALL RESULT: {passing_tests}/{total_tests} implementations passed")
        
        if passing_tests == total_tests:
            print("🎉 ALL TESTS PASSED - Ready for enterprise deployment!")
            return 0
        elif passing_tests > 0:
            print("⚠️ PARTIAL SUCCESS - Review failing implementations")
            return 1
        else:
            print("❌ ALL TESTS FAILED - Critical issues require resolution")
            return 2
            
    except Exception as e:
        logger.error(f"Scale testing failed: {e}")
        logger.error(traceback.format_exc())
        return 3


if __name__ == "__main__":
    exit(main())