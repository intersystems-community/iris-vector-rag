"""
Remediation Engine for Reconciliation Controller.

This module contains the RemediationEngine class that handles the active remediation
of drift issues, including embedding generation and document processing.
"""

import logging
import time
from typing import List, Tuple, Optional
from transformers import AutoTokenizer, AutoModel
import torch

from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.connection import ConnectionManager
from iris_rag.controllers.reconciliation_components.document_service import DocumentService
from iris_rag.validation.embedding_validator import EmbeddingValidator
from iris_rag.controllers.reconciliation_components.models import (
    DriftAnalysis,
    DesiredState,
    SystemState,
    ReconciliationAction
)
from common.utils import get_embedding_func

logger = logging.getLogger(__name__)


class RemediationEngine:
    """
    Engine responsible for executing remediation actions to fix drift issues.
    
    This class handles the active remediation of detected drift issues, including
    embedding generation, document processing, and coordination of remediation actions.
    """
    
    def __init__(self, config_manager: ConfigurationManager, 
                 connection_manager: ConnectionManager,
                 document_service: DocumentService,
                 embedding_validator: EmbeddingValidator):
        """
        Initialize the RemediationEngine.
        
        Args:
            config_manager: Configuration manager instance
            connection_manager: Connection manager for database access
            document_service: Service for document operations
            embedding_validator: Validator for embedding quality
        """
        self.config_manager = config_manager
        self.connection_manager = connection_manager
        self.document_service = document_service
        self.embedding_validator = embedding_validator
        
        # Initialize embedding model and tokenizer for remediation
        self._initialize_embedding_model()
        
        logger.info("RemediationEngine initialized")
    
    def _initialize_embedding_model(self):
        """Initialize the embedding model and tokenizer for generating embeddings."""
        try:
            # Get model configuration from config manager
            reconciliation_config = self.config_manager.get_reconciliation_config()
            model_name = reconciliation_config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
            
            # Initialize model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            
            # Set to evaluation mode
            self.model.eval()
            
            logger.info(f"Embedding model initialized: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            # Fall back to None - will use get_embedding_func() instead
            self.model = None
            self.tokenizer = None
    
    def remediate_drift(self, drift_analysis: DriftAnalysis, desired_state: DesiredState, 
                       current_state: SystemState) -> List[ReconciliationAction]:
        """
        Execute remediation actions for detected drift issues.
        
        Args:
            drift_analysis: Analysis of detected drift issues
            desired_state: Target state configuration
            current_state: Current observed system state
            
        Returns:
            List[ReconciliationAction]: Actions taken to remediate drift
        """
        logger.debug("Executing drift remediation")
        
        actions = []
        
        for issue in drift_analysis.issues:
            logger.info(f"Remediating issue: {issue.issue_type} - {issue.description}")
            
            try:
                if issue.issue_type == "mock_contamination":
                    # Get affected document IDs and clear/regenerate embeddings
                    affected_doc_ids = self.document_service.get_documents_with_mock_embeddings()
                    action = self._clear_and_regenerate_embeddings(affected_doc_ids, desired_state)
                    actions.append(action)
                    
                elif issue.issue_type == "low_diversity" or issue.issue_type == "low_diversity_embeddings":
                    # Get affected document IDs and regenerate embeddings
                    affected_doc_ids = self.document_service.get_documents_with_low_diversity_embeddings()
                    action = self._regenerate_low_diversity_embeddings(affected_doc_ids, desired_state)
                    actions.append(action)
                    
                elif issue.issue_type == "missing_embeddings":
                    # Get documents without embeddings
                    affected_doc_ids = self.document_service.get_documents_without_embeddings()
                    action = self._generate_missing_embeddings(affected_doc_ids, desired_state)
                    actions.append(action)
                    
                elif issue.issue_type == "incomplete_token_embeddings":
                    # Get documents with incomplete token embeddings
                    affected_doc_ids = self.document_service.get_documents_with_incomplete_embeddings(
                        desired_state.completeness_requirements.min_embeddings_per_doc
                    )
                    action = self._generate_missing_embeddings(affected_doc_ids, desired_state)
                    actions.append(action)
                    
                else:
                    # Log unhandled issue types
                    action = ReconciliationAction(
                        action_type="log_unhandled",
                        description=f"Unhandled drift issue: {issue.issue_type}",
                        estimated_duration_seconds=0.1,
                        parameters={"issue_type": issue.issue_type, "severity": issue.severity}
                    )
                    actions.append(action)
                    logger.warning(f"Unhandled drift issue type: {issue.issue_type}")
                    
            except Exception as e:
                logger.error(f"Error remediating issue {issue.issue_type}: {e}")
                action = ReconciliationAction(
                    action_type="error",
                    description=f"Failed to remediate {issue.issue_type}: {e}",
                    estimated_duration_seconds=0.0,
                    parameters={"issue_type": issue.issue_type, "error": str(e)}
                )
                actions.append(action)
        
        logger.info(f"Drift remediation completed with {len(actions)} actions executed")
        return actions
    
    def _clear_and_regenerate_embeddings(self, affected_doc_ids: List[str], desired_state: DesiredState = None) -> ReconciliationAction:
        """
        Clear existing token embeddings and regenerate them for affected documents.
        
        Args:
            affected_doc_ids: List of document IDs to process
            
        Returns:
            ReconciliationAction describing what was done
        """
        start_time = time.time()
        logger.info(f"Clearing and regenerating embeddings for {len(affected_doc_ids)} documents")
        
        if not affected_doc_ids:
            return ReconciliationAction(
                action_type="clear_and_regenerate",
                description="No documents to process",
                estimated_duration_seconds=0.0
            )
        
        try:
            iris_connector = self.connection_manager.get_connection("iris")
            embedding_func = get_embedding_func()
            
            # Extract target dimension from desired state
            target_dimension = desired_state.vector_dimensions if desired_state else None
            
            processed_count = 0
            error_count = 0
            
            # Process in batches
            batch_size = 10
            for i in range(0, len(affected_doc_ids), batch_size):
                batch_doc_ids = affected_doc_ids[i:i + batch_size]
                
                cursor = iris_connector.cursor()
                try:
                    for doc_id in batch_doc_ids:
                        # Get document content and process embeddings
                        text_content = self._get_document_text_content(doc_id, cursor)
                        if text_content and self._remediate_single_document_embeddings(
                            doc_id, text_content, embedding_func, target_dimension
                        ):
                            processed_count += 1
                        else:
                            error_count += 1
                    
                    iris_connector.commit()
                    logger.debug(f"Processed batch {i//batch_size + 1}: {len(batch_doc_ids)} documents")
                    
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    iris_connector.rollback()
                    error_count += len(batch_doc_ids)
                finally:
                    cursor.close()
            
            duration = time.time() - start_time
            
            return ReconciliationAction(
                action_type="clear_and_regenerate",
                description=f"Cleared and regenerated embeddings for {processed_count} documents ({error_count} errors)",
                estimated_duration_seconds=duration,
                parameters={
                    "processed_count": processed_count,
                    "error_count": error_count,
                    "total_documents": len(affected_doc_ids)
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Failed to clear and regenerate embeddings: {e}")
            return ReconciliationAction(
                action_type="clear_and_regenerate",
                description=f"Failed to clear and regenerate embeddings: {e}",
                estimated_duration_seconds=duration,
                parameters={"error": str(e)}
            )
    
    def _regenerate_low_diversity_embeddings(self, affected_doc_ids: List[str], desired_state: DesiredState = None) -> ReconciliationAction:
        """
        Regenerate embeddings for documents with low diversity.
        
        Args:
            affected_doc_ids: List of document IDs to process
            
        Returns:
            ReconciliationAction describing what was done
        """
        # For now, this uses the same logic as clear_and_regenerate
        # In the future, this could use more sophisticated regeneration strategies
        action = self._clear_and_regenerate_embeddings(affected_doc_ids, desired_state)
        action.action_type = "regenerate_low_diversity"
        action.description = action.description.replace("Cleared and regenerated", "Regenerated low-diversity")
        return action
    
    def _generate_missing_embeddings(self, affected_doc_ids: List[str], desired_state: DesiredState = None) -> ReconciliationAction:
        """
        Generate embeddings for documents that are missing them.
        
        Args:
            affected_doc_ids: List of document IDs to process
            
        Returns:
            ReconciliationAction describing what was done
        """
        # For now, this uses the same logic as clear_and_regenerate
        # The difference is that we don't need to clear existing embeddings first
        action = self._clear_and_regenerate_embeddings(affected_doc_ids, desired_state)
        action.action_type = "generate_missing"
        action.description = action.description.replace("Cleared and regenerated", "Generated missing")
        return action
    
    def _get_document_text_content(self, doc_id: str, cursor) -> Optional[str]:
        """
        Get and process document text content, handling CLOB streams.
        
        Args:
            doc_id: Document ID to get content for
            cursor: Database cursor
            
        Returns:
            Document text content as string, or None if not found
        """
        try:
            # Get document text content
            cursor.execute(
                "SELECT text_content FROM RAG.SourceDocuments WHERE ID = ?",
                [doc_id]
            )
            result = cursor.fetchone()
            
            if not result or not result[0]:
                logger.warning(f"No text content found for document {doc_id}")
                return None
            
            text_content_raw = result[0]
            text_content = ""
            
            if hasattr(text_content_raw, 'read') and callable(text_content_raw.read):
                logger.debug(f"Detected stream-like object for text_content for doc {doc_id} (type: {type(text_content_raw)}), attempting to read fully.")
                char_list = []
                try:
                    while True:
                        # Read one character/byte at a time as per IRISInputStream.read()
                        char_code = text_content_raw.read()
                        if char_code == -1 or char_code is None:  # Check for EOF or None if stream behaves differently
                            break
                        
                        # Assuming char_code is an int representing a character
                        char_list.append(chr(char_code))
                        
                    text_content = "".join(char_list)
                    logger.debug(f"Fully read stream for doc {doc_id}. Content length: {len(text_content)}. Starts: {text_content[:200]}...")
                except Exception as e_read:
                    logger.error(f"Error reading CLOB stream for document {doc_id}: {e_read}")
                    # text_content remains "" or partially filled; subsequent logic will handle empty tokens.
                finally:
                    if hasattr(text_content_raw, 'close') and callable(text_content_raw.close):
                        try:
                            text_content_raw.close()
                            logger.debug(f"Closed CLOB stream for doc {doc_id}.")
                        except Exception as e_close:
                            logger.error(f"Error closing CLOB stream for doc {doc_id}: {e_close}")
            elif text_content_raw is not None:  # If not a stream, try to treat as string
                text_content = str(text_content_raw)
                logger.debug(f"text_content for doc {doc_id} was not a stream, used as string. Content: {text_content[:200]}...")
            else:
                # This case is already covered by the `if not result or not result[0]:` check before this block.
                # If result[0] (text_content_raw) is None, text_content remains ""
                logger.warning(f"text_content_raw was None for doc {doc_id} upon direct assignment.")
            
            return text_content if text_content.strip() else None
            
        except Exception as e:
            logger.error(f"Error getting document text content for {doc_id}: {e}")
            return None
    
    def _remediate_single_document_embeddings(self, doc_id: str, text_content: str,
                                            embedding_func, target_dimension: int = None) -> bool:
        """
        Process document embeddings for a single document during remediation.
        
        Args:
            doc_id: Document ID to process
            text_content: Document text content
            embedding_func: Embedding function to use
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Simple tokenization (space-splitting as interim solution)
            tokens = text_content.strip().split()
            if not tokens:
                logger.warning(f"No tokens found for document {doc_id}")
                return True  # Not an error, just empty content
            
            # Limit tokens for performance
            max_tokens = 512  # TODO: Make this configurable from pipeline settings
            if len(tokens) > max_tokens:
                tokens = tokens[:max_tokens]
            
            # Generate embeddings for tokens
            raw_embeddings = embedding_func(tokens)  # This is List[List[float]]
            logger.debug(f"Generated {len(raw_embeddings)} raw_embeddings for {len(tokens)} tokens.")
            if raw_embeddings and isinstance(raw_embeddings[0], list):
                logger.debug(f"Dimension of first raw_embedding: {len(raw_embeddings[0])}")
            
            # Use configured target dimension or fall back to actual embedding dimension
            if target_dimension is None:
                target_dimension = len(raw_embeddings[0]) if raw_embeddings and raw_embeddings[0] else 384  # Default dimension
            
            return self.document_service.save_document_embeddings(
                doc_id=doc_id,
                tokens=tokens,
                embeddings=raw_embeddings,
                target_dimension=target_dimension
            )
            
        except Exception as e:
            logger.error(f"Error processing document embeddings for {doc_id}: {e}")
            return False