"""
State Observer Component for Reconciliation Framework

This module contains the StateObserver class responsible for observing current
system state and determining desired state configurations. It's extracted from
the main ReconciliationController to improve modularity and testability.
"""

import logging
from datetime import datetime
from typing import Optional

from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.connection import ConnectionManager
from iris_rag.validation.embedding_validator import EmbeddingValidator
from iris_rag.controllers.reconciliation_components.models import (
    SystemState,
    DesiredState,
    CompletenessRequirements,
    QualityIssues
)

logger = logging.getLogger(__name__)


class StateObserver:
    """
    Component responsible for observing current system state and determining desired state.
    
    This class encapsulates the logic for:
    - Observing current system state from the database
    - Determining desired state from configuration
    - Quality analysis using EmbeddingValidator
    """
    
    def __init__(self, config_manager: ConfigurationManager, connection_manager: ConnectionManager, embedding_validator: EmbeddingValidator):
        """
        Initialize the StateObserver.
        
        Args:
            config_manager: Configuration manager instance for accessing settings
            connection_manager: Connection manager for database access
            embedding_validator: Validator for embedding quality analysis
        """
        self.config_manager = config_manager
        self.connection_manager = connection_manager
        self.embedding_validator = embedding_validator
        
        logger.debug("StateObserver initialized")
    
    def observe_current_state(self) -> SystemState:
        """
        Observe and analyze the current system state.
        
        Returns:
            SystemState: Current observed state of the system
        """
        logger.debug("Observing current system state")
        
        try:
            # Get database connection
            iris_connector = self.connection_manager.get_connection("iris")
            cursor = iris_connector.cursor()
            
            # Query total documents
            doc_query = "SELECT COUNT(*) as total_docs FROM RAG.SourceDocuments"
            cursor.execute(doc_query)
            doc_result = cursor.fetchone()
            total_documents = doc_result[0] if doc_result else 0
            
            # Query total token embeddings
            token_query = "SELECT COUNT(*) as total_tokens FROM RAG.DocumentTokenEmbeddings"
            cursor.execute(token_query)
            token_result = cursor.fetchone()
            total_token_embeddings = token_result[0] if token_result else 0

            # Query for documents with no token embeddings at all
            docs_missing_all_embeddings_query = """
                SELECT COUNT(DISTINCT sd.id)
                FROM RAG.SourceDocuments sd
                LEFT JOIN RAG.DocumentTokenEmbeddings dte ON sd.id = dte.doc_id
                WHERE dte.doc_id IS NULL
            """
            cursor.execute(docs_missing_all_embeddings_query)
            missing_all_result = cursor.fetchone()
            documents_without_any_embeddings = missing_all_result[0] if missing_all_result else 0

            # Query for documents with an insufficient number of token embeddings (e.g., < 5)
            # This is a heuristic. A more robust solution might involve RAG.DocumentChunks
            # or a per-document expected token count.
            # For now, assume a document is incomplete if it has > 0 and < 5 embeddings.
            # This query counts documents that have some embeddings but fewer than 5.
            # It excludes documents that have NO embeddings at all (handled by documents_without_any_embeddings).
            docs_with_few_embeddings_query = """
                SELECT COUNT(doc_id)
                FROM (
                    SELECT sd.id as doc_id, COUNT(dte.id) as embedding_count
                    FROM RAG.SourceDocuments sd
                    JOIN RAG.DocumentTokenEmbeddings dte ON sd.id = dte.doc_id
                    GROUP BY sd.id
                    HAVING COUNT(dte.id) > 0 AND COUNT(dte.id) < 5
                ) AS subquery
            """
            # The value '5' is a placeholder/heuristic for "expected minimum tokens if not complete"
            # It should ideally come from config or be more dynamic.
            cursor.execute(docs_with_few_embeddings_query)
            incomplete_result = cursor.fetchone()
            documents_with_incomplete_embeddings_count = incomplete_result[0] if incomplete_result else 0
            
            # Calculate average embedding size
            avg_embedding_size = 0.0
            if total_token_embeddings > 0:
                # Sample a few embeddings to estimate average size
                sample_query = """
                    SELECT TOP 5 token_embedding
                    FROM RAG.DocumentTokenEmbeddings
                    WHERE token_embedding IS NOT NULL
                """
                cursor.execute(sample_query)
                sample_results = cursor.fetchall()
                if sample_results:
                    sizes = []
                    for row in sample_results:
                        embedding_str = str(row[0])
                        if ',' in embedding_str:
                            sizes.append(len(embedding_str.split(',')))
                    avg_embedding_size = sum(sizes) / len(sizes) if sizes else 0.0
            
            # Use EmbeddingValidator to analyze quality issues
            sample_embeddings = self.embedding_validator.sample_embeddings_from_database(
                table_name="RAG.DocumentTokenEmbeddings",
                sample_size=50
            )
            quality_issues = self.embedding_validator.analyze_quality(sample_embeddings)
            
            current_state = SystemState(
                total_documents=total_documents,
                total_token_embeddings=total_token_embeddings,
                avg_embedding_size=avg_embedding_size,
                quality_issues=quality_issues,
                documents_without_any_embeddings=documents_without_any_embeddings,
                documents_with_incomplete_embeddings_count=documents_with_incomplete_embeddings_count
            )
            
            logger.info(f"Current state observed: {total_documents} documents, "
                       f"{total_token_embeddings} token embeddings")
            
            return current_state
            
        except Exception as e:
            logger.error(f"Error observing current state: {e}")
            # Return empty state on error
            return SystemState(
                total_documents=0,
                total_token_embeddings=0,
                avg_embedding_size=0.0,
                quality_issues=QualityIssues(),
                documents_without_any_embeddings=0,
                documents_with_incomplete_embeddings_count=0
            )
    
    def get_desired_state(self, pipeline_type: str = "colbert") -> DesiredState:
        """
        Get the desired target state from configuration.
        
        Args:
            pipeline_type: The pipeline type to get desired state for
            
        Returns:
            DesiredState: Target state configuration
        """
        logger.debug(f"Getting desired state from configuration for pipeline: {pipeline_type}")
        
        # Use the new ConfigurationManager methods to get structured config
        embedding_config = self.config_manager.get_desired_embedding_state(pipeline_type)
        reconciliation_config = self.config_manager.get_reconciliation_config()
        
        # Extract values from structured configuration
        target_document_count = embedding_config.get("target_document_count", 1000)
        embedding_model = embedding_config.get("model_name", "fjmgAI/reason-colBERT-150M-GTE-ModernColBERT")
        
        # Handle both token_dimension (ColBERT) and vector_dimensions (other pipelines)
        vector_dimensions = embedding_config.get("token_dimension") or embedding_config.get("vector_dimensions", 768)
        
        # Get validation settings
        validation_config = embedding_config.get("validation", {})
        diversity_threshold = validation_config.get("diversity_threshold", 0.7)
        mock_detection_enabled = validation_config.get("mock_detection_enabled", True)
        min_embedding_quality_score = validation_config.get("min_embedding_quality_score", 0.8)
        
        # Get completeness requirements
        completeness_config = embedding_config.get("completeness", {})
        require_all_docs = completeness_config.get("require_all_docs", True)
        require_token_embeddings = completeness_config.get("require_token_embeddings", pipeline_type.lower() == "colbert")
        min_completeness_percent = completeness_config.get("min_completeness_percent", 95.0)
        min_embeddings_per_doc = completeness_config.get("min_embeddings_per_doc", 5) # Read from config
        
        # Create completeness requirements with enhanced configuration
        completeness_requirements = CompletenessRequirements(
            require_all_docs=require_all_docs,
            require_token_embeddings=require_token_embeddings,
            min_embedding_quality_score=min_embedding_quality_score,
            min_embeddings_per_doc=min_embeddings_per_doc # Pass to constructor
        )
        
        # Create desired state with comprehensive configuration
        desired_state = DesiredState(
            target_document_count=target_document_count,
            embedding_model=embedding_model,
            vector_dimensions=vector_dimensions,
            completeness_requirements=completeness_requirements,
            diversity_threshold=diversity_threshold
        )
        
        logger.info(f"Desired state for {pipeline_type}: {target_document_count} documents, "
                   f"model: {embedding_model}, dimensions: {vector_dimensions}, "
                   f"require_token_embeddings: {require_token_embeddings}")
        
        return desired_state