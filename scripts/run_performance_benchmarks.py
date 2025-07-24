#!/usr/bin/env python3
"""
Performance Benchmarking for RAG Templates
Generates detailed performance metrics and identifies bottlenecks.
"""

import sys
import time
import json
import psutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple
from dataclasses import dataclass, asdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.database_schema_manager import get_schema_manager
from common.iris_connector import get_iris_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for a single operation."""
    operation: str
    duration_ms: float
    memory_mb: float
    cpu_percent: float
    rows_processed: int = 0
    throughput_per_sec: float = 0.0
    
    def __post_init__(self):
        if self.rows_processed > 0 and self.duration_ms > 0:
            self.throughput_per_sec = (self.rows_processed / self.duration_ms) * 1000

@dataclass 
class SystemMetrics:
    """System-level performance metrics."""
    timestamp: str
    total_memory_gb: float
    available_memory_gb: float
    memory_usage_percent: float
    cpu_count: int
    cpu_usage_percent: float
    disk_usage_percent: float

class PerformanceBenchmarker:
    """Comprehensive performance benchmarking system."""
    
    def __init__(self):
        self.schema = get_schema_manager()
        self.connection = None
        self.metrics = []
        self.system_metrics = []
        self.start_time = datetime.now()
        
        # Get initial system state
        self._capture_system_metrics("benchmark_start")
    
    def _capture_system_metrics(self, label: str) -> SystemMetrics:
        """Capture current system performance metrics."""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        metrics = SystemMetrics(
            timestamp=datetime.now().isoformat(),
            total_memory_gb=round(memory.total / (1024**3), 2),
            available_memory_gb=round(memory.available / (1024**3), 2),
            memory_usage_percent=memory.percent,
            cpu_count=psutil.cpu_count(),
            cpu_usage_percent=psutil.cpu_percent(interval=1),
            disk_usage_percent=disk.percent
        )
        
        self.system_metrics.append((label, metrics))
        return metrics
    
    def _time_operation(self, operation_name: str, func, *args, **kwargs) -> Tuple[Any, PerformanceMetrics]:
        """Time an operation and capture performance metrics."""
        # Capture initial state
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**2)  # MB
        
        # Run operation
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Capture final state
        final_memory = process.memory_info().rss / (1024**2)  # MB
        cpu_percent = process.cpu_percent()
        
        # Calculate metrics
        duration_ms = (end_time - start_time) * 1000
        memory_mb = max(final_memory - initial_memory, 0)
        
        metrics = PerformanceMetrics(
            operation=operation_name,
            duration_ms=duration_ms,
            memory_mb=memory_mb,
            cpu_percent=cpu_percent
        )
        
        self.metrics.append(metrics)
        logger.info(f"⏱️  {operation_name}: {duration_ms:.1f}ms, {memory_mb:.1f}MB")
        
        return result, metrics
    
    def benchmark_database_operations(self) -> Dict[str, Any]:
        """Benchmark core database operations."""
        logger.info("🔍 Benchmarking Database Operations...")
        
        try:
            self.connection = get_iris_connection()
            cursor = self.connection.cursor()
            
            benchmarks = {}
            
            # 1. Simple SELECT performance
            def simple_select():
                table_name = self.schema.get_table_name('source_documents', fully_qualified=True)
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                return cursor.fetchone()[0]
            
            count, metrics = self._time_operation("simple_count_query", simple_select)
            metrics.rows_processed = 1
            benchmarks['simple_count'] = asdict(metrics)
            
            # 2. Complex SELECT with WHERE
            def complex_select():
                table_name = self.schema.get_table_name('source_documents', fully_qualified=True)
                cursor.execute(f"SELECT doc_id, title FROM {table_name} WHERE title LIKE '%diabetes%' LIMIT 100")
                return cursor.fetchall()
            
            results, metrics = self._time_operation("complex_select_query", complex_select)
            metrics.rows_processed = len(results)
            benchmarks['complex_select'] = asdict(metrics)
            
            # 3. Token embeddings query (large table)
            def token_query():
                table_name = self.schema.get_table_name('document_token_embeddings', fully_qualified=True)
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                return cursor.fetchone()[0]
            
            token_count, metrics = self._time_operation("token_count_query", token_query)
            metrics.rows_processed = 1
            benchmarks['token_count'] = asdict(metrics)
            
            # 4. JOIN operation
            def join_query():
                docs_table = self.schema.get_table_name('source_documents', fully_qualified=True)
                tokens_table = self.schema.get_table_name('document_token_embeddings', fully_qualified=True)
                cursor.execute(f"""
                    SELECT d.doc_id, d.title, COUNT(t.token_index) as token_count
                    FROM {docs_table} d
                    LEFT JOIN {tokens_table} t ON d.doc_id = t.doc_id
                    GROUP BY d.doc_id, d.title
                    LIMIT 50
                """)
                return cursor.fetchall()
            
            join_results, metrics = self._time_operation("join_query", join_query)
            metrics.rows_processed = len(join_results)
            benchmarks['join_query'] = asdict(metrics)
            
            return {
                'database_benchmarks': benchmarks,
                'total_documents': count,
                'total_tokens': token_count
            }
            
        except Exception as e:
            logger.error(f"Database benchmark failed: {e}")
            return {'error': str(e)}
    
    def benchmark_vector_operations(self) -> Dict[str, Any]:
        """Benchmark vector-specific operations."""
        logger.info("🧮 Benchmarking Vector Operations...")
        
        try:
            cursor = self.connection.cursor()
            benchmarks = {}
            
            # 1. Vector similarity search simulation
            def vector_similarity():
                tokens_table = self.schema.get_table_name('document_token_embeddings', fully_qualified=True)
                # Simulate vector similarity by selecting tokens for a specific document
                cursor.execute(f"""
                    SELECT doc_id, token_text, token_embedding 
                    FROM {tokens_table} 
                    WHERE doc_id = 'sample'
                    LIMIT 100
                """)
                return cursor.fetchall()
            
            vector_results, metrics = self._time_operation("vector_similarity_simulation", vector_similarity)
            metrics.rows_processed = len(vector_results)
            benchmarks['vector_similarity'] = asdict(metrics)
            
            # 2. Embedding retrieval
            def embedding_retrieval():
                docs_table = self.schema.get_table_name('source_documents', fully_qualified=True)
                cursor.execute(f"""
                    SELECT doc_id, embedding 
                    FROM {docs_table}
                    WHERE embedding IS NOT NULL
                    LIMIT 50
                """)
                return cursor.fetchall()
            
            embedding_results, metrics = self._time_operation("embedding_retrieval", embedding_retrieval)
            metrics.rows_processed = len(embedding_results)
            benchmarks['embedding_retrieval'] = asdict(metrics)
            
            return {'vector_benchmarks': benchmarks}
            
        except Exception as e:
            logger.error(f"Vector benchmark failed: {e}")
            return {'error': str(e)}
    
    def benchmark_pipeline_readiness(self) -> Dict[str, Any]:
        """Benchmark pipeline readiness checks."""
        logger.info("🚰 Benchmarking Pipeline Readiness...")
        
        pipeline_requirements = {
            'BasicRAG': ['source_documents'],
            'HyDE': ['source_documents'],
            'CRAG': ['source_documents', 'document_chunks'],
            'GraphRAG': ['source_documents', 'document_entities'],
            'ColBERT': ['source_documents', 'document_token_embeddings'],
            'NodeRAG': ['source_documents', 'document_chunks'],
            'HybridIFind': ['source_documents', 'ifind_index']
        }
        
        def check_all_pipelines():
            cursor = self.connection.cursor()
            readiness = {}
            
            for pipeline, required_tables in pipeline_requirements.items():
                pipeline_ready = True
                table_counts = {}
                
                for table_key in required_tables:
                    try:
                        table_name = self.schema.get_table_name(table_key, fully_qualified=True)
                        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                        count = cursor.fetchone()[0]
                        table_counts[table_key] = count
                        
                        if count == 0:
                            pipeline_ready = False
                    except Exception:
                        pipeline_ready = False
                        table_counts[table_key] = -1
                
                readiness[pipeline] = {
                    'ready': pipeline_ready,
                    'table_counts': table_counts
                }
            
            return readiness
        
        readiness, metrics = self._time_operation("pipeline_readiness_check", check_all_pipelines)
        metrics.rows_processed = len(pipeline_requirements)
        
        return {
            'readiness_benchmark': asdict(metrics),
            'pipeline_readiness': readiness
        }
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        logger.info("📊 Generating Performance Report...")
        
        # Capture final system metrics
        self._capture_system_metrics("benchmark_end")
        
        # Run all benchmarks
        db_results = self.benchmark_database_operations()
        vector_results = self.benchmark_vector_operations()
        pipeline_results = self.benchmark_pipeline_readiness()
        
        # Calculate summary statistics
        all_metrics = [m for m in self.metrics]
        
        if all_metrics:
            avg_duration = sum(m.duration_ms for m in all_metrics) / len(all_metrics)
            max_duration = max(m.duration_ms for m in all_metrics)
            min_duration = min(m.duration_ms for m in all_metrics)
            total_memory = sum(m.memory_mb for m in all_metrics)
        else:
            avg_duration = max_duration = min_duration = total_memory = 0
        
        # Identify bottlenecks
        bottlenecks = []
        for metric in all_metrics:
            if metric.duration_ms > avg_duration * 2:
                bottlenecks.append({
                    'operation': metric.operation,
                    'duration_ms': metric.duration_ms,
                    'severity': 'high' if metric.duration_ms > avg_duration * 3 else 'medium'
                })
        
        return {
            'benchmark_metadata': {
                'timestamp': self.start_time.isoformat(),
                'duration_seconds': (datetime.now() - self.start_time).total_seconds(),
                'schema_version': 'config-driven',
                'total_operations': len(all_metrics)
            },
            'summary_statistics': {
                'avg_operation_time_ms': round(avg_duration, 2),
                'max_operation_time_ms': round(max_duration, 2),
                'min_operation_time_ms': round(min_duration, 2),
                'total_memory_used_mb': round(total_memory, 2)
            },
            'bottlenecks': bottlenecks,
            'system_metrics': {
                label: asdict(metrics) for label, metrics in self.system_metrics
            },
            'detailed_results': {
                **db_results,
                **vector_results,
                **pipeline_results
            },
            'all_operation_metrics': [asdict(m) for m in all_metrics]
        }
    
    def save_report(self, report: Dict[str, Any], output_file: str = None) -> str:
        """Save performance report to file."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"benchmarks/performance_report_{timestamp}.json"
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"💾 Performance report saved to: {output_path}")
        return str(output_path)
    
    def print_summary(self, report: Dict[str, Any]):
        """Print performance summary to console."""
        print("\n" + "="*70)
        print("⚡ RAG TEMPLATES PERFORMANCE BENCHMARK REPORT")
        print("="*70)
        
        metadata = report['benchmark_metadata']
        summary = report['summary_statistics']
        bottlenecks = report['bottlenecks']
        
        print(f"📅 Timestamp: {metadata['timestamp']}")
        print(f"⏱️  Duration: {metadata['duration_seconds']:.1f}s")
        print(f"🔢 Operations: {metadata['total_operations']}")
        
        print(f"\n📊 Performance Summary:")
        print(f"  Avg Operation Time: {summary['avg_operation_time_ms']:.1f}ms")
        print(f"  Max Operation Time: {summary['max_operation_time_ms']:.1f}ms")
        print(f"  Min Operation Time: {summary['min_operation_time_ms']:.1f}ms")
        print(f"  Total Memory Used:  {summary['total_memory_used_mb']:.1f}MB")
        
        if bottlenecks:
            print(f"\n⚠️  Identified Bottlenecks ({len(bottlenecks)}):")
            for bottleneck in bottlenecks:
                severity_icon = "🔴" if bottleneck['severity'] == 'high' else "🟡"
                print(f"  {severity_icon} {bottleneck['operation']}: {bottleneck['duration_ms']:.1f}ms")
        else:
            print(f"\n✅ No significant bottlenecks detected!")
        
        # Pipeline readiness summary
        pipeline_data = report['detailed_results'].get('pipeline_readiness', {})
        if pipeline_data:
            ready_count = sum(1 for p in pipeline_data.values() if p.get('ready', False))
            total_count = len(pipeline_data)
            print(f"\n🚰 Pipeline Readiness: {ready_count}/{total_count} ready")
        
        print("="*70)

def main():
    """Main execution function."""
    benchmarker = PerformanceBenchmarker()
    
    # Generate comprehensive performance report
    report = benchmarker.generate_performance_report()
    
    # Save and display results
    output_file = benchmarker.save_report(report)
    benchmarker.print_summary(report)
    
    logger.info(f"✅ Performance benchmark completed! Report: {output_file}")

if __name__ == "__main__":
    main()