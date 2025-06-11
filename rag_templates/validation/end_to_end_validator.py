"""
End-to-End Validator for RAG System.

Tests real query execution for all pipelines, validates response quality,
monitors performance, and integrates with the iris_rag package.
"""

import logging
import time
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class ExternalConnectionWrapper:
    """
    Wrapper to make external database connections compatible with iris_rag ConnectionManager interface.
    """
    
    def __init__(self, external_connection, config_manager):
        """
        Initialize wrapper with external connection.
        
        Args:
            external_connection: External database connection (e.g., from get_iris_connection())
            config_manager: Configuration manager instance
        """
        self.external_connection = external_connection
        self.config_manager = config_manager
        self._connection_type = "external_dbapi"
    
    def get_connection(self, backend_name: str = "iris"):
        """
        Return the external connection.
        
        Args:
            backend_name: Backend name (ignored, returns external connection)
            
        Returns:
            The external database connection
        """
        return self.external_connection
    
    def execute(self, query: str, params=None):
        """
        Execute a query using the external connection.
        
        Args:
            query: SQL query to execute
            params: Optional query parameters
            
        Returns:
            Query results
        """
        cursor = self.external_connection.cursor()
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # Return results for SELECT queries
            if query.strip().upper().startswith('SELECT'):
                return cursor.fetchall()
            else:
                self.external_connection.commit()
                return cursor.rowcount
        finally:
            cursor.close()
    
    @property
    def connection_type(self):
        """Return connection type."""
        return self._connection_type


class EndToEndValidator:
    """
    Performs end-to-end validation of all RAG pipelines.
    """
    def __init__(self, config=None, db_connection=None):
        """
        Initializes the EndToEndValidator.

        Args:
            config: Optional configuration object.
            db_connection: Database connection object.
        """
        self.config = config or {}
        self.db_connection = db_connection
        self.results = {}
        
        # Pipeline definitions with their module paths
        self.pipeline_definitions = {
            "basic_rag": {
                "module": "iris_rag.pipelines.basic",
                "class": "BasicRAGPipeline",
                "description": "Basic RAG with vector similarity search"
            },
            "colbert_rag": {
                "module": "iris_rag.pipelines.colbert",
                "class": "ColBERTRAGPipeline",
                "description": "ColBERT-based RAG with token-level embeddings"
            },
            "hyde_rag": {
                "module": "iris_rag.pipelines.hyde",
                "class": "HyDERAGPipeline",
                "description": "HyDE (Hypothetical Document Embeddings) RAG"
            },
            "crag": {
                "module": "iris_rag.pipelines.crag",
                "class": "CRAGPipeline",
                "description": "Corrective RAG with self-reflection"
            },
            "hybrid_ifind_rag": {
                "module": "iris_rag.pipelines.hybrid_ifind",
                "class": "HybridIFindRAGPipeline",
                "description": "Hybrid RAG with iFind integration"
            },
            "graph_rag": {
                "module": "iris_rag.pipelines.graphrag",
                "class": "GraphRAGPipeline",
                "description": "Graph-based RAG with entity relationships"
            },
            "node_rag": {
                "module": "iris_rag.pipelines.noderag",
                "class": "NodeRAGPipeline",
                "description": "Node-based RAG with knowledge graphs"
            }
        }
        
        # Initialize pipeline instances
        self.pipelines_to_test = {}
        self._initialize_pipelines()

    def _initialize_pipelines(self):
        """
        Initializes all pipeline instances for testing.
        """
        logger.info("Initializing RAG pipelines for testing...")
        
        for pipeline_name, pipeline_def in self.pipeline_definitions.items():
            try:
                # Import the pipeline module
                module = __import__(pipeline_def["module"], fromlist=[pipeline_def["class"]])
                pipeline_class = getattr(module, pipeline_def["class"])
                
                # Initialize pipeline with connection and config
                pipeline_instance = self._create_pipeline_instance(pipeline_class, pipeline_name)
                
                if pipeline_instance:
                    self.pipelines_to_test[pipeline_name] = pipeline_instance
                    logger.info(f"Successfully initialized {pipeline_name}")
                else:
                    logger.warning(f"Failed to create instance for {pipeline_name}")
                    self.pipelines_to_test[pipeline_name] = None
                    
            except Exception as e:
                logger.error(f"Failed to initialize {pipeline_name}: {e}")
                self.pipelines_to_test[pipeline_name] = None

    def _create_pipeline_instance(self, pipeline_class, pipeline_name):
        """
        Creates a pipeline instance with proper configuration.
        """
        try:
            # Import required iris_rag classes
            from iris_rag.core.connection import ConnectionManager
            from iris_rag.config.manager import ConfigurationManager
            from common.utils import get_llm_func
            
            # Create configuration manager
            config_manager = ConfigurationManager()
            
            # Create connection manager with external connection if available
            if self.db_connection:
                # Create a wrapper for external connection
                connection_manager = ExternalConnectionWrapper(self.db_connection, config_manager)
            else:
                # Use standard connection manager
                connection_manager = ConnectionManager(config_manager)
            
            # Get LLM function and embedding function
            llm_func = get_llm_func()
            
            # Import embedding function for pipelines that need it
            try:
                from common.utils import get_embedding_func
                embedding_func = get_embedding_func()
            except Exception as e:
                logger.warning(f"Could not get embedding function: {e}")
                embedding_func = None
            
            # Check if pipeline needs embedding_func parameter
            import inspect
            sig = inspect.signature(pipeline_class.__init__)
            params = list(sig.parameters.keys())
            
            # Create pipeline instance with correct parameters
            if 'embedding_func' in params:
                pipeline_instance = pipeline_class(
                    connection_manager=connection_manager,
                    config_manager=config_manager,
                    embedding_func=embedding_func,
                    llm_func=llm_func
                )
            else:
                pipeline_instance = pipeline_class(
                    connection_manager=connection_manager,
                    config_manager=config_manager,
                    llm_func=llm_func
                )
            return pipeline_instance
            
        except Exception as e:
            logger.error(f"Error creating pipeline instance for {pipeline_name}: {e}")
            return None

    def test_all_pipelines(self, sample_queries):
        """
        Tests all 7 RAG pipelines with a set of sample queries.

        Args:
            sample_queries: A list of strings, representing queries.
        """
        try:
            logger.info(f"Starting end-to-end testing of {len(self.pipelines_to_test)} pipelines...")
            
            if not sample_queries:
                sample_queries = self._get_default_test_queries()
            
            all_pipelines_passed = True
            pipeline_results = {}
            
            for pipeline_name, pipeline_instance in self.pipelines_to_test.items():
                logger.info(f"Testing pipeline: {pipeline_name}")
                
                if pipeline_instance is None:
                    pipeline_results[pipeline_name] = {
                        'status': 'skipped',
                        'details': 'Pipeline not instantiated',
                        'queries': {},
                        'overall_score': 0.0
                    }
                    all_pipelines_passed = False
                    continue

                pipeline_passed, pipeline_result = self._test_single_pipeline(
                    pipeline_name, pipeline_instance, sample_queries
                )
                
                pipeline_results[pipeline_name] = pipeline_result
                
                if not pipeline_passed:
                    all_pipelines_passed = False
                    logger.error(f"Pipeline {pipeline_name} failed testing")
                else:
                    logger.info(f"Pipeline {pipeline_name} passed testing")
            
            # Calculate overall results
            self.results['pipeline_results'] = pipeline_results
            self.results['overall_e2e_status'] = 'pass' if all_pipelines_passed else 'fail'
            self.results['pipelines_tested'] = len(self.pipelines_to_test)
            self.results['pipelines_passed'] = sum(1 for r in pipeline_results.values() if r['status'] == 'pass')
            self.results['success_rate'] = (self.results['pipelines_passed'] / self.results['pipelines_tested']) * 100
            
            logger.info(f"End-to-end testing completed. Success rate: {self.results['success_rate']:.1f}%")
            return all_pipelines_passed
            
        except Exception as e:
            logger.error(f"Error during end-to-end testing: {e}")
            self.results['overall_e2e_status'] = 'error'
            self.results['error'] = str(e)
            return False

    def _get_default_test_queries(self):
        """
        Returns default test queries for medical/scientific domain.
        """
        return [
            "What is COVID-19 and how does it affect the respiratory system?",
            "Explain the mechanism of action of ACE2 inhibitors.",
            "What are the latest treatments for Alzheimer's disease?",
            "How do mRNA vaccines work?",
            "What is the role of inflammation in cardiovascular disease?"
        ]

    def _test_single_pipeline(self, name, pipeline, queries):
        """
        Helper to test a single pipeline with multiple queries.
        """
        try:
            pipeline_start_time = time.time()
            query_results = {}
            total_score = 0.0
            successful_queries = 0
            
            for query in queries:
                logger.info(f"Testing query: '{query[:50]}...' on {name}")
                
                query_start_time = time.time()
                
                try:
                    # Execute query on pipeline
                    response = self._execute_pipeline_query(pipeline, query)
                    query_duration = time.time() - query_start_time
                    
                    if response:
                        # Validate response quality
                        quality_score = self.validate_response_quality(name, query, response)
                        
                        # Monitor performance
                        performance_metrics = self.monitor_performance(name, query, query_duration, response)
                        
                        query_results[query] = {
                            'status': 'success',
                            'response': response,
                            'quality_score': quality_score,
                            'performance_metrics': performance_metrics,
                            'execution_time': query_duration
                        }
                        
                        total_score += quality_score
                        successful_queries += 1
                        
                    else:
                        query_results[query] = {
                            'status': 'failed',
                            'details': 'Pipeline returned empty response',
                            'execution_time': query_duration
                        }
                        
                except Exception as e:
                    query_duration = time.time() - query_start_time
                    query_results[query] = {
                        'status': 'error',
                        'details': f'Error executing query: {str(e)}',
                        'execution_time': query_duration
                    }
                    logger.error(f"Error executing query '{query}' on {name}: {e}")
            
            # Calculate pipeline overall score
            pipeline_duration = time.time() - pipeline_start_time
            average_score = total_score / len(queries) if queries else 0.0
            success_rate = (successful_queries / len(queries)) * 100 if queries else 0.0
            
            pipeline_passed = success_rate >= 70.0 and average_score >= 0.6  # 70% success rate, 60% quality threshold
            
            pipeline_result = {
                'status': 'pass' if pipeline_passed else 'fail',
                'details': f'Pipeline tested with {len(queries)} queries',
                'queries': query_results,
                'overall_score': average_score,
                'success_rate': success_rate,
                'successful_queries': successful_queries,
                'total_queries': len(queries),
                'total_execution_time': pipeline_duration
            }
            
            return pipeline_passed, pipeline_result
            
        except Exception as e:
            logger.error(f"Error testing pipeline {name}: {e}")
            return False, {
                'status': 'error',
                'details': f'Error during pipeline testing: {str(e)}',
                'queries': {},
                'overall_score': 0.0
            }

    def _execute_pipeline_query(self, pipeline, query):
        """
        Executes a query on a pipeline and returns the response.
        """
        try:
            # Different pipelines may have different interfaces
            # Try common method names
            if hasattr(pipeline, 'query'):
                response = pipeline.query(query)
            elif hasattr(pipeline, 'run'):
                response = pipeline.run(query)
            elif hasattr(pipeline, 'execute'):
                response = pipeline.execute(query)
            elif hasattr(pipeline, '__call__'):
                response = pipeline(query)
            else:
                # Fallback: try to call the pipeline directly
                response = pipeline.query(query) if hasattr(pipeline, 'query') else None
            
            # Extract response text if it's a complex object
            if isinstance(response, dict):
                response_text = response.get('answer', response.get('response', response.get('result', str(response))))
            elif hasattr(response, 'content'):
                response_text = response.content
            elif hasattr(response, 'text'):
                response_text = response.text
            else:
                response_text = str(response) if response else None
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error executing pipeline query: {e}")
            return None

    def validate_response_quality(self, pipeline_name, query, response):
        """
        Validates the quality of a pipeline's response using predefined metrics.
        Returns a quality score between 0.0 and 1.0.
        """
        try:
            if not response or not isinstance(response, str):
                return 0.0
            
            quality_score = 0.0
            
            # Basic quality checks
            checks = {
                'non_empty': len(response.strip()) > 0,
                'sufficient_length': len(response.strip()) >= 20,
                'not_error_message': not any(error_word in response.lower() 
                                           for error_word in ['error', 'failed', 'exception', 'traceback']),
                'contains_relevant_terms': self._check_relevance(query, response),
                'proper_formatting': self._check_formatting(response)
            }
            
            # Calculate weighted score
            weights = {
                'non_empty': 0.2,
                'sufficient_length': 0.2,
                'not_error_message': 0.2,
                'contains_relevant_terms': 0.3,
                'proper_formatting': 0.1
            }
            
            for check, passed in checks.items():
                if passed:
                    quality_score += weights[check]
            
            # Store detailed quality results
            if pipeline_name not in self.results:
                self.results[pipeline_name] = {}
            if 'quality_checks' not in self.results[pipeline_name]:
                self.results[pipeline_name]['quality_checks'] = {}
            
            self.results[pipeline_name]['quality_checks'][query] = {
                'score': quality_score,
                'checks': checks,
                'response_length': len(response)
            }
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Error validating response quality: {e}")
            return 0.0

    def _check_relevance(self, query, response):
        """
        Checks if the response contains terms relevant to the query.
        """
        try:
            query_words = set(query.lower().split())
            response_words = set(response.lower().split())
            
            # Remove common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'what', 'how', 'when', 'where', 'why', 'who'}
            
            query_content_words = query_words - stop_words
            response_content_words = response_words - stop_words
            
            if not query_content_words:
                return True  # If no content words in query, assume relevant
            
            # Check for overlap
            overlap = query_content_words.intersection(response_content_words)
            relevance_ratio = len(overlap) / len(query_content_words)
            
            return relevance_ratio >= 0.3  # At least 30% of query terms should appear in response
            
        except Exception as e:
            logger.error(f"Error checking relevance: {e}")
            return False

    def _check_formatting(self, response):
        """
        Checks if the response is properly formatted.
        """
        try:
            # Basic formatting checks
            has_proper_sentences = '.' in response or '!' in response or '?' in response
            not_all_caps = not response.isupper()
            not_all_lower = not response.islower()
            reasonable_length = 10 <= len(response) <= 5000
            
            return has_proper_sentences and not_all_caps and reasonable_length
            
        except Exception as e:
            logger.error(f"Error checking formatting: {e}")
            return False

    def monitor_performance(self, pipeline_name, query, execution_time, response):
        """
        Monitors and reports performance metrics for pipeline execution.
        """
        try:
            performance_metrics = {
                'execution_time_seconds': execution_time,
                'response_length': len(response) if response else 0,
                'tokens_per_second': (len(response.split()) / execution_time) if response and execution_time > 0 else 0,
                'performance_category': self._categorize_performance(execution_time)
            }
            
            # Store performance results
            if pipeline_name not in self.results:
                self.results[pipeline_name] = {}
            if 'performance_metrics' not in self.results[pipeline_name]:
                self.results[pipeline_name]['performance_metrics'] = {}
            
            self.results[pipeline_name]['performance_metrics'][query] = performance_metrics
            
            return performance_metrics
            
        except Exception as e:
            logger.error(f"Error monitoring performance: {e}")
            return {
                'execution_time_seconds': execution_time,
                'error': str(e)
            }

    def _categorize_performance(self, execution_time):
        """
        Categorizes performance based on execution time.
        """
        if execution_time < 2.0:
            return 'excellent'
        elif execution_time < 5.0:
            return 'good'
        elif execution_time < 10.0:
            return 'acceptable'
        else:
            return 'slow'

    def generate_performance_report(self):
        """
        Generates a comprehensive performance report.
        """
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_pipelines': len(self.pipelines_to_test),
                    'successful_pipelines': self.results.get('pipelines_passed', 0),
                    'success_rate': self.results.get('success_rate', 0.0),
                    'overall_status': self.results.get('overall_e2e_status', 'unknown')
                },
                'pipeline_details': {}
            }
            
            for pipeline_name, pipeline_result in self.results.get('pipeline_results', {}).items():
                report['pipeline_details'][pipeline_name] = {
                    'status': pipeline_result.get('status', 'unknown'),
                    'overall_score': pipeline_result.get('overall_score', 0.0),
                    'success_rate': pipeline_result.get('success_rate', 0.0),
                    'total_execution_time': pipeline_result.get('total_execution_time', 0.0),
                    'query_count': pipeline_result.get('total_queries', 0)
                }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {'error': str(e)}

    def get_results(self):
        """
        Returns the results of the end-to-end validation.
        """
        return self.results

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        from common.iris_connection_manager import get_iris_connection
        connection = get_iris_connection()
        
        validator = EndToEndValidator(db_connection=connection)
        sample_queries = [
            "What is COVID-19?", 
            "Tell me about gene therapy.",
            "How do vaccines work?",
            "What causes cancer?",
            "Explain diabetes treatment."
        ]
        
        success = validator.test_all_pipelines(sample_queries)
        
        print("End-to-End Validation Results:")
        print(json.dumps(validator.get_results(), indent=2, default=str))
        
        if success:
            print("\n✅ End-to-end validation PASSED")
        else:
            print("\n❌ End-to-end validation FAILED")
            
        # Generate performance report
        report = validator.generate_performance_report()
        print("\nPerformance Report:")
        print(json.dumps(report, indent=2, default=str))
        
    except Exception as e:
        print(f"Error running end-to-end validation: {e}")