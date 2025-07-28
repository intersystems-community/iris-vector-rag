"""
System Validator for RAG Templates

Provides comprehensive system validation and integrity checking.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json

from ..config.manager import ConfigurationManager
from .health_monitor import HealthMonitor
from common.iris_connection_manager import get_iris_connection

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of a validation test."""
    test_name: str
    success: bool
    message: str
    details: Dict[str, Any]
    duration_ms: float
    timestamp: datetime

class SystemValidator:
    """
    Comprehensive system validation for the RAG templates system.
    
    Validates data integrity, pipeline functionality, and system configuration.
    """
    
    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        """
        Initialize the system validator.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigurationManager()
        self.health_monitor = HealthMonitor(self.config_manager)
    
    def validate_data_integrity(self) -> ValidationResult:
        """Validate data integrity in the database."""
        start_time = time.time()
        test_name = "data_integrity"
        
        try:
            connection = get_iris_connection()
            issues = []
            details = {}
            
            with connection.cursor() as cursor:
                # Check for duplicate documents
                cursor.execute("""
                    SELECT doc_id, COUNT(*) as count
                    FROM RAG.SourceDocuments
                    GROUP BY doc_id
                    HAVING COUNT(*) > 1
                """)
                duplicates = cursor.fetchall()
                
                if duplicates:
                    issues.append(f"Found {len(duplicates)} duplicate document IDs")
                    details['duplicate_doc_ids'] = [row[0] for row in duplicates[:10]]  # First 10
                
                # Check for null embeddings in documents that should have them
                cursor.execute("""
                    SELECT COUNT(*) as null_embeddings
                    FROM RAG.SourceDocuments
                    WHERE content IS NOT NULL AND LENGTH(content) > 100 AND embedding IS NULL
                """)
                null_embeddings = cursor.fetchone()[0]
                details['documents_without_embeddings'] = null_embeddings
                
                if null_embeddings > 0:
                    issues.append(f"Found {null_embeddings} documents without embeddings")
                
                # Check for orphaned chunks (if DocumentChunks table exists)
                try:
                    cursor.execute("""
                        SELECT COUNT(*) as orphaned_chunks
                        FROM RAG.DocumentChunks dc
                        LEFT JOIN RAG.SourceDocuments sd ON dc.doc_id = sd.doc_id
                        WHERE sd.doc_id IS NULL
                    """)
                    orphaned_chunks = cursor.fetchone()[0]
                    details['orphaned_chunks'] = orphaned_chunks
                    
                    if orphaned_chunks > 0:
                        issues.append(f"Found {orphaned_chunks} orphaned chunks")
                except:
                    # DocumentChunks table might not exist
                    details['orphaned_chunks'] = 'N/A - DocumentChunks table not found'
                
                # Check embedding dimensions consistency
                cursor.execute("""
                    SELECT DISTINCT VECTOR_DIMENSION(embedding) as dimension
                    FROM RAG.SourceDocuments
                    WHERE embedding IS NOT NULL
                """)
                dimensions = [row[0] for row in cursor.fetchall()]
                details['embedding_dimensions'] = dimensions
                
                if len(dimensions) > 1:
                    issues.append(f"Inconsistent embedding dimensions: {dimensions}")
                
                # Check for empty content
                cursor.execute("""
                    SELECT COUNT(*) as empty_content
                    FROM RAG.SourceDocuments
                    WHERE content IS NULL OR LENGTH(TRIM(content)) = 0
                """)
                empty_content = cursor.fetchone()[0]
                details['documents_with_empty_content'] = empty_content
                
                if empty_content > 0:
                    issues.append(f"Found {empty_content} documents with empty content")
            
            success = len(issues) == 0
            message = "Data integrity validation passed" if success else f"Found {len(issues)} issues: {'; '.join(issues)}"
            
        except Exception as e:
            success = False
            message = f"Data integrity validation failed: {e}"
            details = {'error': str(e)}
        
        duration_ms = (time.time() - start_time) * 1000
        
        return ValidationResult(
            test_name=test_name,
            success=success,
            message=message,
            details=details,
            duration_ms=duration_ms,
            timestamp=datetime.now()
        )
    
    def validate_pipeline_functionality(self, test_queries: Optional[List[str]] = None) -> ValidationResult:
        """Validate that RAG pipelines are functioning correctly."""
        start_time = time.time()
        test_name = "pipeline_functionality"
        
        if test_queries is None:
            test_queries = [
                "What is machine learning?",
                "Explain neural networks",
                "How does deep learning work?"
            ]
        
        try:
            from ..pipelines.basic import BasicRAGPipeline
            
            pipeline = BasicRAGPipeline(config_manager=self.config_manager)
            results = []
            issues = []
            
            for query in test_queries:
                try:
                    query_start = time.time()
                    result = pipeline.execute(query)
                    query_time = (time.time() - query_start) * 1000
                    
                    # Validate result structure
                    required_keys = ['query', 'answer', 'retrieved_documents']
                    missing_keys = [key for key in required_keys if key not in result]
                    
                    if missing_keys:
                        issues.append(f"Query '{query}' missing keys: {missing_keys}")
                    
                    if not result.get('answer') or len(result.get('answer', '').strip()) == 0:
                        issues.append(f"Query '{query}' returned empty answer")
                    
                    if not result.get('retrieved_documents') or len(result.get('retrieved_documents', [])) == 0:
                        issues.append(f"Query '{query}' returned no documents")
                    
                    results.append({
                        'query': query,
                        'success': len(missing_keys) == 0,
                        'execution_time_ms': query_time,
                        'answer_length': len(result.get('answer', '')),
                        'documents_retrieved': len(result.get('retrieved_documents', []))
                    })
                    
                except Exception as e:
                    issues.append(f"Query '{query}' failed: {e}")
                    results.append({
                        'query': query,
                        'success': False,
                        'error': str(e)
                    })
            
            success = len(issues) == 0
            message = "Pipeline functionality validation passed" if success else f"Found {len(issues)} issues"
            
            details = {
                'test_queries_count': len(test_queries),
                'successful_queries': len([r for r in results if r.get('success', False)]),
                'failed_queries': len([r for r in results if not r.get('success', False)]),
                'issues': issues,
                'query_results': results
            }
            
        except Exception as e:
            success = False
            message = f"Pipeline functionality validation failed: {e}"
            details = {'error': str(e)}
        
        duration_ms = (time.time() - start_time) * 1000
        
        return ValidationResult(
            test_name=test_name,
            success=success,
            message=message,
            details=details,
            duration_ms=duration_ms,
            timestamp=datetime.now()
        )
    
    def validate_vector_operations(self) -> ValidationResult:
        """Validate vector operations and HNSW performance."""
        start_time = time.time()
        test_name = "vector_operations"
        
        try:
            connection = get_iris_connection()
            details = {}
            issues = []
            
            with connection.cursor() as cursor:
                # Test basic vector operations
                cursor.execute("SELECT TO_VECTOR('[0.1, 0.2, 0.3]') AS test_vector")
                vector_result = cursor.fetchone()
                
                if not vector_result:
                    issues.append("Basic vector creation failed")
                else:
                    details['basic_vector_operations'] = 'working'
                
                # Test vector similarity
                cursor.execute("""
                    SELECT VECTOR_COSINE(TO_VECTOR('[0.1, 0.2, 0.3]'), TO_VECTOR('[0.1, 0.2, 0.3]')) AS similarity
                """)
                similarity_result = cursor.fetchone()
                
                if not similarity_result or abs(similarity_result[0] - 1.0) > 0.001:
                    issues.append("Vector similarity calculation failed")
                else:
                    details['vector_similarity'] = 'working'
                
                # Test vector query performance
                cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NOT NULL")
                embedded_count = cursor.fetchone()[0]
                details['embedded_documents'] = embedded_count
                
                if embedded_count > 0:
                    test_vector = "[" + ",".join(["0.1"] * 384) + "]"
                    
                    # Test query performance
                    query_start = time.time()
                    cursor.execute("""
                        SELECT TOP 10 doc_id, VECTOR_COSINE(embedding, TO_VECTOR(?)) AS similarity
                        FROM RAG.SourceDocuments 
                        WHERE embedding IS NOT NULL
                        ORDER BY similarity DESC
                    """, (test_vector,))
                    
                    results = cursor.fetchall()
                    query_time_ms = (time.time() - query_start) * 1000
                    
                    details['vector_query_time_ms'] = query_time_ms
                    details['vector_query_results'] = len(results)
                    
                    if query_time_ms > 5000:  # 5 seconds
                        issues.append(f"Vector query performance poor: {query_time_ms:.1f}ms")
                    
                    # Check HNSW index existence
                    cursor.execute("""
                        SELECT INDEX_NAME, INDEX_TYPE
                        FROM INFORMATION_SCHEMA.INDEXES
                        WHERE TABLE_NAME = 'SourceDocuments' AND TABLE_SCHEMA = 'RAG'
                        AND COLUMN_NAME = 'embedding'
                    """)
                    indexes = cursor.fetchall()
                    details['vector_indexes'] = [{'name': idx[0], 'type': idx[1]} for idx in indexes]
                    
                    hnsw_indexes = [idx for idx in indexes if 'HNSW' in idx[1]]
                    if not hnsw_indexes:
                        issues.append("No HNSW indexes found on embedding column")
                else:
                    issues.append("No embedded documents found for vector testing")
            
            success = len(issues) == 0
            message = "Vector operations validation passed" if success else f"Found {len(issues)} issues"
            
        except Exception as e:
            success = False
            message = f"Vector operations validation failed: {e}"
            details = {'error': str(e)}
            issues = [str(e)]
        
        duration_ms = (time.time() - start_time) * 1000
        
        return ValidationResult(
            test_name=test_name,
            success=success,
            message=message,
            details=details,
            duration_ms=duration_ms,
            timestamp=datetime.now()
        )
    
    def validate_system_configuration(self) -> ValidationResult:
        """Validate system configuration and dependencies."""
        start_time = time.time()
        test_name = "system_configuration"
        
        try:
            details = {}
            issues = []
            
            # Check required Python packages
            required_packages = [
                'psutil', 'docker', 'numpy', 'sentence_transformers'
            ]
            
            missing_packages = []
            for package in required_packages:
                try:
                    __import__(package)
                    details[f'package_{package}'] = 'available'
                except ImportError:
                    missing_packages.append(package)
                    details[f'package_{package}'] = 'missing'
            
            if missing_packages:
                issues.append(f"Missing required packages: {missing_packages}")
            
            # Check configuration files
            config_files = [
                'config/monitoring.json',
                'config/database.json'
            ]
            
            for config_file in config_files:
                try:
                    with open(config_file, 'r') as f:
                        json.load(f)
                    details[f'config_{config_file}'] = 'valid'
                except FileNotFoundError:
                    details[f'config_{config_file}'] = 'missing'
                except json.JSONDecodeError:
                    issues.append(f"Invalid JSON in {config_file}")
                    details[f'config_{config_file}'] = 'invalid'
            
            # Check log directories
            log_dirs = [
                'logs/performance',
                'logs/health_checks',
                'logs/errors'
            ]
            
            for log_dir in log_dirs:
                import os
                if os.path.exists(log_dir):
                    details[f'log_dir_{log_dir}'] = 'exists'
                else:
                    issues.append(f"Missing log directory: {log_dir}")
                    details[f'log_dir_{log_dir}'] = 'missing'
            
            # Run health check
            health_results = self.health_monitor.run_comprehensive_health_check()
            overall_health = self.health_monitor.get_overall_health_status(health_results)
            details['overall_health_status'] = overall_health
            
            if overall_health == 'critical':
                issues.append("System health check indicates critical issues")
            elif overall_health == 'warning':
                issues.append("System health check indicates warnings")
            
            success = len(issues) == 0
            message = "System configuration validation passed" if success else f"Found {len(issues)} issues"
            
        except Exception as e:
            success = False
            message = f"System configuration validation failed: {e}"
            details = {'error': str(e)}
        
        duration_ms = (time.time() - start_time) * 1000
        
        return ValidationResult(
            test_name=test_name,
            success=success,
            message=message,
            details=details,
            duration_ms=duration_ms,
            timestamp=datetime.now()
        )
    
    def run_comprehensive_validation(self) -> Dict[str, ValidationResult]:
        """Run all validation tests."""
        logger.info("Starting comprehensive system validation...")
        
        validations = {
            'data_integrity': self.validate_data_integrity,
            'pipeline_functionality': self.validate_pipeline_functionality,
            'vector_operations': self.validate_vector_operations,
            'system_configuration': self.validate_system_configuration
        }
        
        results = {}
        
        for validation_name, validation_func in validations.items():
            logger.info(f"Running {validation_name} validation...")
            try:
                results[validation_name] = validation_func()
            except Exception as e:
                logger.error(f"Validation {validation_name} failed with exception: {e}")
                results[validation_name] = ValidationResult(
                    test_name=validation_name,
                    success=False,
                    message=f"Validation failed with exception: {e}",
                    details={'error': str(e)},
                    duration_ms=0,
                    timestamp=datetime.now()
                )
        
        # Log summary
        successful_validations = sum(1 for r in results.values() if r.success)
        total_validations = len(results)
        
        logger.info(f"Validation complete: {successful_validations}/{total_validations} passed")
        
        return results
    
    def generate_validation_report(self, results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """Generate a comprehensive validation report."""
        successful_validations = [r for r in results.values() if r.success]
        failed_validations = [r for r in results.values() if not r.success]
        
        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'summary': {
                'total_validations': len(results),
                'successful_validations': len(successful_validations),
                'failed_validations': len(failed_validations),
                'success_rate': len(successful_validations) / len(results) * 100,
                'overall_status': 'PASS' if len(failed_validations) == 0 else 'FAIL'
            },
            'validation_results': {
                name: {
                    'success': result.success,
                    'message': result.message,
                    'duration_ms': result.duration_ms,
                    'timestamp': result.timestamp.isoformat(),
                    'details': result.details
                }
                for name, result in results.items()
            },
            'recommendations': self._generate_recommendations(results)
        }
        
        return report
    
    def _generate_recommendations(self, results: Dict[str, ValidationResult]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        for name, result in results.items():
            if not result.success:
                if name == 'data_integrity':
                    recommendations.append("Run data cleanup and re-embedding for documents with integrity issues")
                elif name == 'pipeline_functionality':
                    recommendations.append("Check pipeline configuration and dependencies")
                elif name == 'vector_operations':
                    recommendations.append("Verify HNSW indexes and vector data quality")
                elif name == 'system_configuration':
                    recommendations.append("Install missing dependencies and fix configuration issues")
        
        if not recommendations:
            recommendations.append("System validation passed - no immediate actions required")
        
        return recommendations