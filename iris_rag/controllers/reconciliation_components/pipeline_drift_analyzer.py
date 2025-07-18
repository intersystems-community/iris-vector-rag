"""
Pipeline-Specific Drift Analyzer Extension

This module extends the basic drift analyzer to detect pipeline-specific drift issues
that the generic analyzer misses, such as GraphRAG graph underpopulation, 
ColBERT token embedding gaps, and HybridIFind table synchronization issues.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .models import SystemState, DesiredState, DriftIssue, DriftAnalysis
from .drift_analyzer import DriftAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class PipelineSpecificState:
    """Extended state information for pipeline-specific analysis."""
    # Graph-related state for GraphRAG/NodeRAG
    total_entities: int = 0
    total_graph_nodes: int = 0
    total_graph_edges: int = 0
    entity_coverage_ratio: float = 0.0
    
    # Table synchronization state for HybridIFind
    ifind_table_count: int = 0
    main_table_count: int = 0
    ifind_sync_ratio: float = 0.0
    
    # Token embedding state for ColBERT
    documents_with_token_embeddings: int = 0
    token_embedding_coverage_ratio: float = 0.0
    avg_tokens_per_document: float = 0.0
    
    # Chunk state
    documents_with_chunks: int = 0
    chunk_coverage_ratio: float = 0.0


class PipelineDriftAnalyzer(DriftAnalyzer):
    """
    Extended drift analyzer that detects pipeline-specific drift issues.
    
    This analyzer extends the base DriftAnalyzer to catch drift issues specific
    to individual RAG pipeline types that the generic analyzer misses.
    """
    
    def __init__(self, connection_manager=None):
        """
        Initialize the pipeline-specific drift analyzer.
        
        Args:
            connection_manager: Database connection manager for detailed state analysis
        """
        super().__init__()
        self.connection_manager = connection_manager
    
    def analyze_pipeline_drift(self, current_state: SystemState, desired_state: DesiredState, 
                              pipeline_type: str) -> DriftAnalysis:
        """
        Analyze drift with pipeline-specific considerations.
        
        Args:
            current_state: Current system state
            desired_state: Desired target state
            pipeline_type: Type of pipeline to analyze ("graphrag", "noderag", "colbert", "hybrid_ifind", etc.)
            
        Returns:
            DriftAnalysis with pipeline-specific drift issues
        """
        logger.info(f"Analyzing pipeline-specific drift for {pipeline_type}")
        
        # Start with base drift analysis
        base_analysis = super().analyze_drift(current_state, desired_state)
        
        # Get extended state information
        pipeline_state = self._get_pipeline_specific_state(current_state)
        
        # Add pipeline-specific drift detection
        additional_issues = []
        
        if pipeline_type in ["graphrag", "noderag"]:
            additional_issues.extend(self._analyze_graph_drift(pipeline_state, desired_state))
        
        if pipeline_type == "colbert":
            additional_issues.extend(self._analyze_colbert_drift(pipeline_state, desired_state))
        
        if pipeline_type == "hybrid_ifind":
            additional_issues.extend(self._analyze_hybrid_ifind_drift(pipeline_state, desired_state))
        
        # Combine base and pipeline-specific issues
        all_issues = base_analysis.issues + additional_issues
        has_drift = len(all_issues) > 0
        
        return DriftAnalysis(
            has_drift=has_drift,
            issues=all_issues
        )
    
    def _get_pipeline_specific_state(self, current_state: SystemState) -> PipelineSpecificState:
        """Get extended state information for pipeline-specific analysis."""
        pipeline_state = PipelineSpecificState()
        
        if not self.connection_manager:
            logger.warning("No connection manager provided - using default values for pipeline state")
            return pipeline_state
        
        try:
            connection = self.connection_manager.get_connection()
            cursor = connection.cursor()
            
            # Get graph data state
            try:
                cursor.execute("SELECT COUNT(*) FROM RAG.DocumentEntities")
                pipeline_state.total_entities = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM RAG.KnowledgeGraphNodes")
                pipeline_state.total_graph_nodes = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM RAG.KnowledgeGraphEdges")
                pipeline_state.total_graph_edges = cursor.fetchone()[0]
                
                if current_state.total_documents > 0:
                    pipeline_state.entity_coverage_ratio = pipeline_state.total_entities / current_state.total_documents
            except Exception as e:
                logger.warning(f"Could not get graph state: {e}")
            
            # Get table synchronization state
            try:
                cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocumentsIFind")
                pipeline_state.ifind_table_count = cursor.fetchone()[0]
                pipeline_state.main_table_count = current_state.total_documents
                
                if current_state.total_documents > 0:
                    pipeline_state.ifind_sync_ratio = pipeline_state.ifind_table_count / current_state.total_documents
            except Exception as e:
                logger.warning(f"Could not get IFind state: {e}")
            
            # Get token embedding coverage
            try:
                cursor.execute("SELECT COUNT(DISTINCT doc_id) FROM RAG.DocumentTokenEmbeddings")
                pipeline_state.documents_with_token_embeddings = cursor.fetchone()[0]
                
                if current_state.total_documents > 0:
                    pipeline_state.token_embedding_coverage_ratio = pipeline_state.documents_with_token_embeddings / current_state.total_documents
                
                # Get average tokens per document
                if pipeline_state.documents_with_token_embeddings > 0:
                    cursor.execute("""
                        SELECT AVG(token_count) FROM (
                            SELECT doc_id, COUNT(*) as token_count 
                            FROM RAG.DocumentTokenEmbeddings 
                            GROUP BY doc_id
                        )
                    """)
                    result = cursor.fetchone()
                    pipeline_state.avg_tokens_per_document = float(result[0]) if result and result[0] else 0.0
            except Exception as e:
                logger.warning(f"Could not get token embedding state: {e}")
            
            # Get chunk coverage
            try:
                cursor.execute("SELECT COUNT(DISTINCT doc_id) FROM RAG.DocumentChunks")
                pipeline_state.documents_with_chunks = cursor.fetchone()[0]
                
                if current_state.total_documents > 0:
                    pipeline_state.chunk_coverage_ratio = pipeline_state.documents_with_chunks / current_state.total_documents
            except Exception as e:
                logger.warning(f"Could not get chunk state: {e}")
            
            cursor.close()
            
        except Exception as e:
            logger.error(f"Failed to get pipeline-specific state: {e}")
        
        return pipeline_state
    
    def _analyze_graph_drift(self, pipeline_state: PipelineSpecificState, desired_state: DesiredState) -> List[DriftIssue]:
        """Analyze drift issues specific to graph-based pipelines (GraphRAG, NodeRAG)."""
        issues = []
        
        # Check entity coverage ratio
        min_entity_coverage = 0.1  # At least 0.1 entities per document for basic functionality
        optimal_entity_coverage = 0.5  # 0.5+ entities per document for good performance
        
        if pipeline_state.entity_coverage_ratio < min_entity_coverage:
            issues.append(DriftIssue(
                issue_type="graph_severely_underpopulated",
                severity="critical",
                description=f"Graph severely underpopulated: {pipeline_state.entity_coverage_ratio:.3f} entities/document (need ≥{min_entity_coverage})",
                affected_count=int((min_entity_coverage - pipeline_state.entity_coverage_ratio) * desired_state.target_document_count),
                recommended_action="Run entity extraction to populate graph with entities and relationships"
            ))
        elif pipeline_state.entity_coverage_ratio < optimal_entity_coverage:
            issues.append(DriftIssue(
                issue_type="graph_underpopulated",
                severity="high",
                description=f"Graph underpopulated: {pipeline_state.entity_coverage_ratio:.3f} entities/document (optimal ≥{optimal_entity_coverage})",
                affected_count=int((optimal_entity_coverage - pipeline_state.entity_coverage_ratio) * desired_state.target_document_count),
                recommended_action="Enhance entity extraction to improve graph coverage and query performance"
            ))
        
        # Check absolute entity count
        min_entities = max(50, desired_state.target_document_count * 0.05)  # At least 50 entities or 5% of docs
        if pipeline_state.total_entities < min_entities:
            issues.append(DriftIssue(
                issue_type="insufficient_graph_entities",
                severity="high",
                description=f"Insufficient entities: {pipeline_state.total_entities} entities (need ≥{min_entities})",
                affected_count=int(min_entities - pipeline_state.total_entities),
                recommended_action="Extract more entities from document content to build comprehensive knowledge graph"
            ))
        
        # Check graph connectivity
        if pipeline_state.total_graph_nodes > 0 and pipeline_state.total_graph_edges == 0:
            issues.append(DriftIssue(
                issue_type="graph_no_relationships",
                severity="medium", 
                description=f"Graph has {pipeline_state.total_graph_nodes} nodes but no relationships",
                affected_count=pipeline_state.total_graph_nodes,
                recommended_action="Create relationships between graph nodes to enable graph traversal"
            ))
        
        return issues
    
    def _analyze_colbert_drift(self, pipeline_state: PipelineSpecificState, desired_state: DesiredState) -> List[DriftIssue]:
        """Analyze drift issues specific to ColBERT pipeline."""
        issues = []
        
        # Check token embedding coverage
        min_token_coverage = 0.9  # 90% of documents should have token embeddings
        if pipeline_state.token_embedding_coverage_ratio < min_token_coverage:
            issues.append(DriftIssue(
                issue_type="incomplete_token_embeddings",
                severity="high",
                description=f"Incomplete token embeddings: {pipeline_state.token_embedding_coverage_ratio:.1%} coverage (need ≥{min_token_coverage:.0%})",
                affected_count=int((min_token_coverage - pipeline_state.token_embedding_coverage_ratio) * desired_state.target_document_count),
                recommended_action="Generate token embeddings for documents missing ColBERT token representations"
            ))
        
        # Check average tokens per document
        min_tokens_per_doc = 10  # Expect at least 10 tokens per document on average
        expected_tokens_per_doc = 50  # Good coverage would be ~50 tokens per document
        
        if pipeline_state.avg_tokens_per_document < min_tokens_per_doc:
            issues.append(DriftIssue(
                issue_type="insufficient_token_density",
                severity="medium",
                description=f"Low token density: {pipeline_state.avg_tokens_per_document:.1f} tokens/document (need ≥{min_tokens_per_doc})",
                affected_count=pipeline_state.documents_with_token_embeddings,
                recommended_action="Re-process documents with better tokenization to increase token embedding density"
            ))
        elif pipeline_state.avg_tokens_per_document < expected_tokens_per_doc:
            issues.append(DriftIssue(
                issue_type="suboptimal_token_density",
                severity="low",
                description=f"Suboptimal token density: {pipeline_state.avg_tokens_per_document:.1f} tokens/document (optimal ≥{expected_tokens_per_doc})",
                affected_count=pipeline_state.documents_with_token_embeddings,
                recommended_action="Consider improving tokenization strategy for better ColBERT performance"
            ))
        
        return issues
    
    def _analyze_hybrid_ifind_drift(self, pipeline_state: PipelineSpecificState, desired_state: DesiredState) -> List[DriftIssue]:
        """Analyze drift issues specific to HybridIFind pipeline."""
        issues = []
        
        # Check IFind table synchronization
        expected_sync_ratio = 1.0  # IFind table should match main table exactly
        tolerance = 0.05  # 5% tolerance
        
        if abs(pipeline_state.ifind_sync_ratio - expected_sync_ratio) > tolerance:
            if pipeline_state.ifind_sync_ratio < expected_sync_ratio:
                issues.append(DriftIssue(
                    issue_type="ifind_table_undersynchronized",
                    severity="critical",
                    description=f"IFind table undersynchronized: {pipeline_state.ifind_sync_ratio:.1%} of main table (need 100%)",
                    affected_count=int((expected_sync_ratio - pipeline_state.ifind_sync_ratio) * desired_state.target_document_count),
                    recommended_action="Synchronize IFind table with main SourceDocuments table"
                ))
            else:
                issues.append(DriftIssue(
                    issue_type="ifind_table_oversynchronized",
                    severity="medium",
                    description=f"IFind table oversynchronized: {pipeline_state.ifind_sync_ratio:.1%} of main table (expect 100%)",
                    affected_count=int((pipeline_state.ifind_sync_ratio - expected_sync_ratio) * desired_state.target_document_count),
                    recommended_action="Clean up extra records in IFind table"
                ))
        
        # Check if IFind table is completely empty
        if pipeline_state.ifind_table_count == 0 and pipeline_state.main_table_count > 0:
            issues.append(DriftIssue(
                issue_type="ifind_table_empty",
                severity="critical",
                description=f"IFind table is empty but main table has {pipeline_state.main_table_count} documents",
                affected_count=pipeline_state.main_table_count,
                recommended_action="Populate IFind table and create text search indexes"
            ))
        
        return issues
    
    def generate_pipeline_readiness_report(self, current_state: SystemState, 
                                         pipeline_types: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Generate a comprehensive pipeline readiness report with drift analysis.
        
        Args:
            current_state: Current system state
            pipeline_types: List of pipeline types to analyze
            
        Returns:
            Dictionary with readiness status for each pipeline type
        """
        # Create a default desired state for analysis
        desired_state = DesiredState(
            target_document_count=max(1000, current_state.total_documents),
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            vector_dimensions=384,
            completeness_requirements=current_state.quality_issues.__class__()
        )
        
        report = {}
        
        for pipeline_type in pipeline_types:
            try:
                drift_analysis = self.analyze_pipeline_drift(current_state, desired_state, pipeline_type)
                
                # Categorize issues by severity
                critical_issues = [i for i in drift_analysis.issues if i.severity == "critical"]
                high_issues = [i for i in drift_analysis.issues if i.severity == "high"]
                
                # Determine readiness
                ready = len(critical_issues) == 0
                performance_optimal = len(critical_issues) == 0 and len(high_issues) == 0
                
                report[pipeline_type] = {
                    "ready": ready,
                    "performance_optimal": performance_optimal,
                    "drift_detected": drift_analysis.has_drift,
                    "total_issues": len(drift_analysis.issues),
                    "critical_issues": len(critical_issues),
                    "high_issues": len(high_issues),
                    "issues": [
                        {
                            "type": issue.issue_type,
                            "severity": issue.severity,
                            "description": issue.description,
                            "recommended_action": issue.recommended_action
                        }
                        for issue in drift_analysis.issues
                    ]
                }
                
            except Exception as e:
                logger.error(f"Failed to analyze {pipeline_type}: {e}")
                report[pipeline_type] = {
                    "ready": False,
                    "performance_optimal": False,
                    "drift_detected": True,
                    "error": str(e)
                }
        
        return report


def analyze_pipeline_drift(connection_manager, current_state: SystemState, 
                          pipeline_type: str) -> DriftAnalysis:
    """
    Convenience function to analyze pipeline-specific drift.
    
    Args:
        connection_manager: Database connection manager
        current_state: Current system state
        pipeline_type: Type of pipeline to analyze
        
    Returns:
        DriftAnalysis with pipeline-specific issues
    """
    analyzer = PipelineDriftAnalyzer(connection_manager)
    
    # Create a reasonable desired state
    desired_state = DesiredState(
        target_document_count=max(1000, current_state.total_documents),
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        vector_dimensions=384,
        completeness_requirements=current_state.quality_issues.__class__()
    )
    
    return analyzer.analyze_pipeline_drift(current_state, desired_state, pipeline_type)