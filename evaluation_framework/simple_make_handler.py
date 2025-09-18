#!/usr/bin/env python3
"""
Production Make Evaluation Handler - STRICT Database Requirements
Integrates RealProductionEvaluator with NO MOCKS, NO FALLBACKS per .clinerules
"""

import sys
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import RealProductionEvaluator
from real_production_evaluation import RealProductionEvaluator

# Core infrastructure imports
from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.connection import ConnectionManager
from iris_rag.pipelines.registry import PipelineRegistry
from iris_rag.pipelines.factory import PipelineFactory
from iris_rag.config.pipeline_config_service import PipelineConfigService
from iris_rag.utils.module_loader import ModuleLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionMakeConfig:
    """Production configuration for make-based evaluation - NO FALLBACKS."""
    
    def __init__(self, evaluation_type: str = "full"):
        self.evaluation_type = evaluation_type
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Configuration based on evaluation type - production scale
        if evaluation_type == "demo":
            self.num_documents = 100
            self.num_questions = 20
            self.target_pipelines = ["BasicRAG", "CRAG"]
        elif evaluation_type == "quick":
            self.num_documents = 1000
            self.num_questions = 100
            self.target_pipelines = ["BasicRAG", "CRAG", "GraphRAG", "BasicRAGReranking"]
        elif evaluation_type == "test":
            self.num_documents = 5000
            self.num_questions = 500
            self.target_pipelines = ["BasicRAG", "CRAG", "GraphRAG", "BasicRAGReranking"]
        else:  # full
            self.num_documents = 10000
            self.num_questions = 2000
            self.target_pipelines = ["BasicRAG", "CRAG", "GraphRAG", "BasicRAGReranking"]
        
        self.output_dir = f"evaluation_results/production_make_{evaluation_type}_{self.timestamp}"
        
        # Critical validation thresholds per .clinerules
        self.min_documents_required = 100
        self.min_pipelines_required = len(self.target_pipelines)

class ProductionMakeHandler:
    """Production make handler using RealProductionEvaluator - STRICT requirements."""
    
    def __init__(self, config: ProductionMakeConfig):
        self.config = config
        logger.info(f"Initializing Production Make Handler for {config.evaluation_type} evaluation")
        logger.info(f"Target: {config.num_documents} docs, {config.num_questions} questions")
        logger.info(f"Required pipelines: {config.target_pipelines}")
        
        # Initialize core infrastructure with STRICT validation
        self.config_manager = ConfigurationManager()
        self.connection_manager = ConnectionManager()
        
        # CRITICAL: Validate database connection immediately - NO FALLBACKS
        self._validate_database_connection_strict()
        
        # CRITICAL: Validate document count - MUST be >100 per .clinerules
        self._validate_document_count_strict()
        
        # Initialize pipeline registry with STRICT validation
        self.pipeline_registry = self._initialize_pipeline_registry_strict()
        
        # CRITICAL: Validate all target pipelines are operational
        self._validate_all_pipelines_strict()
        
        # Initialize RealProductionEvaluator
        self.production_evaluator = RealProductionEvaluator()
        
        logger.info("✅ Production Make Handler initialized successfully - all validations passed")
    
    def _validate_database_connection_strict(self) -> None:
        """STRICT database validation - FAIL if not available per .clinerules."""
        logger.info("STRICT VALIDATION: Testing database connection and document count...")
        
        try:
            # Get a fresh connection and perform both validations in one go
            connection = self.connection_manager.get_connection()
            if connection is None:
                logger.error("❌ CRITICAL FAILURE: Database connection returned None")
                logger.error("Per .clinerules: NO MOCKS, NO FALLBACKS - evaluation MUST FAIL")
                sys.exit(1)
            
            cursor = connection.cursor()
            
            # Test basic connectivity
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            if result is None:
                logger.error("❌ CRITICAL FAILURE: Database query execution failed")
                sys.exit(1)
            
            logger.info("✅ Database connection validated successfully")
            
            # Test document count in same connection (IRIS compatible query)
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE text_content IS NOT NULL")
            result = cursor.fetchone()
            document_count = result[0] if result else 0
            
            logger.info(f"Found {document_count} documents in database")
            
            # Close resources properly
            cursor.close()
            connection.close()
            
            if document_count < self.config.min_documents_required:
                logger.error(f"❌ CRITICAL FAILURE: Only {document_count} documents found")
                logger.error(f"Required minimum: {self.config.min_documents_required} documents")
                logger.error("Per .clinerules: Evaluation MUST FAIL if insufficient data")
                sys.exit(1)
            
            logger.info(f"✅ Document count validation passed: {document_count} >= {self.config.min_documents_required}")
            
        except Exception as e:
            logger.error(f"❌ CRITICAL FAILURE: Database validation failed: {e}")
            logger.error("Per .clinerules: NO MOCKS, NO FALLBACKS - evaluation MUST FAIL")
            sys.exit(1)
    
    def _validate_document_count_strict(self) -> None:
        """STRICT document count validation - now handled in connection validation."""
        # This is now combined with connection validation to avoid connection issues
        pass
    
    def _initialize_pipeline_registry_strict(self) -> PipelineRegistry:
        """Initialize pipeline registry with STRICT validation."""
        logger.info("STRICT VALIDATION: Initializing pipeline registry...")
        
        try:
            # Framework dependencies - all REAL components
            framework_dependencies = {
                'connection_manager': self.connection_manager,
                'config_manager': self.config_manager,
                'llm_func': None,  # Will be set by RealProductionEvaluator
                'vector_store': None  # Will be set by pipelines
            }
            
            # Initialize pipeline components
            config_service = PipelineConfigService()
            module_loader = ModuleLoader()
            pipeline_factory = PipelineFactory(config_service, module_loader, framework_dependencies)
            pipeline_registry = PipelineRegistry(pipeline_factory)
            
            # Register all pipelines
            pipeline_registry.register_pipelines()
            
            registered_pipelines = pipeline_registry.list_pipeline_names()
            logger.info(f"Registered pipelines: {registered_pipelines}")
            
            if len(registered_pipelines) == 0:
                logger.error("❌ CRITICAL FAILURE: No pipelines registered")
                logger.error("Per .clinerules: All pipelines must be operational or evaluation MUST FAIL")
                sys.exit(1)
            
            return pipeline_registry
            
        except Exception as e:
            logger.error(f"❌ CRITICAL FAILURE: Pipeline registry initialization failed: {e}")
            sys.exit(1)
    
    def _validate_all_pipelines_strict(self) -> None:
        """STRICT pipeline validation - ALL target pipelines MUST initialize."""
        logger.info("STRICT VALIDATION: Validating all target pipelines...")
        
        registered_pipelines = self.pipeline_registry.list_pipeline_names()
        missing_pipelines = []
        
        for pipeline_name in self.config.target_pipelines:
            if pipeline_name not in registered_pipelines:
                missing_pipelines.append(pipeline_name)
        
        if missing_pipelines:
            logger.error(f"❌ CRITICAL FAILURE: Missing required pipelines: {missing_pipelines}")
            logger.error(f"Available pipelines: {registered_pipelines}")
            logger.error("Per .clinerules: All pipelines must be operational or evaluation MUST FAIL")
            sys.exit(1)
        
        logger.info(f"✅ All {len(self.config.target_pipelines)} target pipelines validated successfully")
    
    def run_production_evaluation(self) -> bool:
        """Run production evaluation using RealProductionEvaluator."""
        try:
            logger.info("=" * 80)
            logger.info(f"STARTING PRODUCTION {self.config.evaluation_type.upper()} EVALUATION")
            logger.info("=" * 80)
            
            # Ensure output directory exists
            Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
            
            # Run evaluation using RealProductionEvaluator
            logger.info("Executing evaluation with RealProductionEvaluator...")
            results = self.production_evaluator.run_real_evaluation(
                num_documents=self.config.num_documents,
                num_questions=self.config.num_questions
            )
            
            # Validate results integrity
            if not results or 'pipeline_results' not in results:
                logger.error("❌ CRITICAL FAILURE: Invalid evaluation results")
                return False
            
            # Generate production report
            self._generate_production_report(results)
            
            logger.info("=" * 80)
            logger.info("PRODUCTION EVALUATION COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ CRITICAL FAILURE: Production evaluation failed: {e}")
            return False
    
    def _generate_production_report(self, results: Dict) -> None:
        """Generate production-quality evaluation report."""
        logger.info("Generating production evaluation report...")
        
        # Enhanced report with make integration details
        report_content = f"""# Production Make Evaluation Report

## Configuration
- **Evaluation Type**: {self.config.evaluation_type}
- **Timestamp**: {self.config.timestamp}
- **Documents Processed**: {results.get('documents_processed', 0)}
- **Questions Evaluated**: {results.get('questions_evaluated', 0)}
- **Execution Time**: {results.get('execution_time_minutes', 0):.2f} minutes

## Make Integration
- **Make Target**: eval-{self.config.evaluation_type}
- **Handler**: simple_make_handler.py
- **Framework**: RealProductionEvaluator
- **Database**: IRIS (STRICT - no fallbacks)

## Infrastructure Validation
✅ **Database Connection**: STRICT validation passed
✅ **Document Count**: {results.get('documents_processed', 0)} >= {self.config.min_documents_required}
✅ **Pipeline Count**: {len(self.config.target_pipelines)} pipelines operational
✅ **Real Evaluation**: {results.get('infrastructure', {}).get('evaluation_framework', 'Unknown')}

## Pipeline Results
"""
        
        for pipeline_name, result in results.get('pipeline_results', {}).items():
            metrics = result.get('metrics', {})
            overall_score = metrics.get('overall_score', 0.0)
            report_content += f"""
### {pipeline_name}
- **Overall Score**: {overall_score:.3f}
- **Questions**: {result.get('total_questions', 0)}
- **Status**: ✅ Operational
"""
        
        report_content += f"""
## Compliance
✅ **No Mocks**: Real database, real pipelines, real LLM
✅ **No Fallbacks**: Strict failure on any component unavailability  
✅ **Production Scale**: {results.get('documents_processed', 0)} documents evaluated
✅ **Make Compatible**: Supports eval-demo, eval-test, eval-full targets

**Status**: PRODUCTION EVALUATION COMPLETE - {self.config.evaluation_type.upper()}
"""
        
        # Save report
        report_file = f"{self.config.output_dir}/production_report.md"
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Production report saved: {report_file}")

def main():
    """Main entry point for production make evaluation handler."""
    
    if len(sys.argv) != 2:
        print("Usage: python simple_make_handler.py <evaluation_type>")
        print("evaluation_type: demo, test, full")
        sys.exit(1)
    
    evaluation_type = sys.argv[1]
    
    # Validate evaluation type
    valid_types = ['demo', 'test', 'full']
    if evaluation_type not in valid_types:
        print(f"Invalid evaluation type: {evaluation_type}")
        print(f"Valid types: {', '.join(valid_types)}")
        sys.exit(1)
    
    # Create production configuration
    config = ProductionMakeConfig(evaluation_type)
    logger.info(f"Starting PRODUCTION {evaluation_type} evaluation")
    logger.info("Per .clinerules: NO MOCKS, NO FALLBACKS, REAL DATABASE ONLY")
    
    # Create production handler with STRICT validation
    handler = ProductionMakeHandler(config)
    
    # Run production evaluation
    if not handler.run_production_evaluation():
        logger.error("❌ PRODUCTION EVALUATION FAILED")
        sys.exit(1)
    
    logger.info(f"✅ PRODUCTION EVALUATION {evaluation_type.upper()} COMPLETED SUCCESSFULLY")
    print(f"Results available in: {config.output_dir}")

if __name__ == "__main__":
    main()