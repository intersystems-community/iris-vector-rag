#!/usr/bin/env python3
"""
Database State Validator for RAG Templates.

This module provides comprehensive validation of database state consistency,
table integrity, and pipeline readiness. It pressure tests the schema manager
concept and ensures all tables are properly synchronized.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of a validation check."""

    check_name: str
    severity: ValidationSeverity
    status: str  # "pass", "fail", "warning"
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class TableState:
    """State information for a database table."""

    name: str
    row_count: int
    exists: bool
    schema_valid: bool
    last_updated: Optional[str] = None
    dependencies: List[str] = None


class DatabaseStateValidator:
    """
    Comprehensive database state validator for RAG systems.

    This validator pressure tests the schema manager concept by:
    1. Validating table consistency across all pipeline requirements
    2. Checking referential integrity between related tables
    3. Ensuring data completeness for each pipeline type
    4. Detecting silent failures and inconsistent states
    """

    def __init__(self, connection_manager, schema_manager):
        """
        Initialize the database state validator.

        Args:
            connection_manager: Database connection manager
            schema_manager: Schema manager instance to pressure test
        """
        self.connection_manager = connection_manager
        self.schema_manager = schema_manager
        self.validation_results = []

        # Define expected table relationships and constraints
        self.table_dependencies = {
            "RAG.SourceDocuments": [],  # Base table, no dependencies
            "RAG.SourceDocumentsIFind": ["RAG.SourceDocuments"],  # Should match SourceDocuments 1:1
            "RAG.DocumentChunks": ["RAG.SourceDocuments"],  # Many chunks per document
            "RAG.DocumentTokenEmbeddings": ["RAG.SourceDocuments"],  # Many tokens per document
            "RAG.DocumentEntities": ["RAG.SourceDocuments"],  # Many entities per document
            "RAG.KnowledgeGraphNodes": [],  # Independent entities
            "RAG.KnowledgeGraphEdges": ["RAG.KnowledgeGraphNodes"],  # Edges between nodes
            "RAG.EntityRelationships": ["RAG.DocumentEntities"],  # Relationships between entities
        }

        # Define pipeline requirements
        self.pipeline_requirements = {
            "basic": {"required_tables": ["RAG.SourceDocuments"], "minimum_rows": {"RAG.SourceDocuments": 1}},
            "colbert": {
                "required_tables": ["RAG.SourceDocuments", "RAG.DocumentTokenEmbeddings"],
                "minimum_rows": {"RAG.SourceDocuments": 1, "RAG.DocumentTokenEmbeddings": 1},
            },
            "graphrag": {
                "required_tables": ["RAG.SourceDocuments", "RAG.DocumentEntities", "RAG.KnowledgeGraphNodes"],
                "minimum_rows": {"RAG.SourceDocuments": 1, "RAG.DocumentEntities": 10, "RAG.KnowledgeGraphNodes": 5},
            },
            "noderag": {
                "required_tables": ["RAG.SourceDocuments", "RAG.DocumentEntities", "RAG.KnowledgeGraphNodes"],
                "minimum_rows": {"RAG.SourceDocuments": 1, "RAG.DocumentEntities": 10, "RAG.KnowledgeGraphNodes": 5},
            },
            "hybrid_ifind": {
                "required_tables": ["RAG.SourceDocuments", "RAG.SourceDocumentsIFind"],
                "minimum_rows": {"RAG.SourceDocuments": 1, "RAG.SourceDocumentsIFind": 1},
                "consistency_requirements": [("RAG.SourceDocuments", "RAG.SourceDocumentsIFind", 1.0)],  # 1:1 ratio
            },
            "crag": {"required_tables": ["RAG.SourceDocuments"], "minimum_rows": {"RAG.SourceDocuments": 1}},
            "hyde": {"required_tables": ["RAG.SourceDocuments"], "minimum_rows": {"RAG.SourceDocuments": 1}},
        }

    def validate_complete_state(self) -> List[ValidationResult]:
        """
        Perform comprehensive database state validation.

        Returns:
            List of validation results
        """
        logger.info("🔍 Starting comprehensive database state validation...")
        self.validation_results = []

        # 1. Pressure test schema manager
        self._validate_schema_manager()

        # 2. Check table existence and basic state
        table_states = self._get_table_states()

        # 3. Validate table consistency
        self._validate_table_consistency(table_states)

        # 4. Check referential integrity
        self._validate_referential_integrity(table_states)

        # 5. Validate pipeline readiness
        self._validate_pipeline_readiness(table_states)

        # 6. Check for silent failure conditions
        self._detect_silent_failures(table_states)

        # 7. Validate data quality
        self._validate_data_quality(table_states)

        self._log_validation_summary()
        return self.validation_results

    def _validate_schema_manager(self):
        """Pressure test the schema manager concept."""
        logger.info("🔥 Pressure testing schema manager...")

        try:
            # Test schema manager configuration loading
            base_dim = self.schema_manager.get_base_embedding_dimension()
            colbert_dim = self.schema_manager.get_colbert_token_dimension()

            if base_dim <= 0:
                self._add_result(
                    "schema_manager_base_dimension",
                    ValidationSeverity.ERROR,
                    "fail",
                    f"Invalid base embedding dimension: {base_dim}",
                )

            if colbert_dim <= 0:
                self._add_result(
                    "schema_manager_colbert_dimension",
                    ValidationSeverity.ERROR,
                    "fail",
                    f"Invalid ColBERT token dimension: {colbert_dim}",
                )

            # Test dimension consistency
            if base_dim >= colbert_dim:
                self._add_result(
                    "schema_manager_dimension_logic",
                    ValidationSeverity.WARNING,
                    "warning",
                    f"Unusual: Base dimension ({base_dim}) >= ColBERT dimension ({colbert_dim})",
                )

            # Test ColBERT configuration
            colbert_config = self.schema_manager.get_colbert_config()
            if colbert_config["backend"] not in ["native", "pylate"]:
                self._add_result(
                    "schema_manager_colbert_backend",
                    ValidationSeverity.ERROR,
                    "fail",
                    f"Invalid ColBERT backend: {colbert_config['backend']}",
                )

            self._add_result(
                "schema_manager_pressure_test",
                ValidationSeverity.INFO,
                "pass",
                "Schema manager pressure test completed",
            )

        except Exception as e:
            self._add_result(
                "schema_manager_pressure_test",
                ValidationSeverity.CRITICAL,
                "fail",
                f"Schema manager pressure test failed: {e}",
            )

    def _get_table_states(self) -> Dict[str, TableState]:
        """Get current state of all relevant tables."""
        logger.info("📊 Collecting table state information...")

        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()
        table_states = {}

        for table_name in self.table_dependencies.keys():
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]

                table_states[table_name] = TableState(
                    name=table_name,
                    row_count=row_count,
                    exists=True,
                    schema_valid=True,  # Assume valid if we can query it
                    dependencies=self.table_dependencies[table_name],
                )

            except Exception as e:
                table_states[table_name] = TableState(name=table_name, row_count=0, exists=False, schema_valid=False)
                self._add_result(
                    f"table_existence_{table_name}",
                    ValidationSeverity.ERROR,
                    "fail",
                    f"Table {table_name} does not exist or is not accessible: {e}",
                )

        cursor.close()
        return table_states

    def _validate_table_consistency(self, table_states: Dict[str, TableState]):
        """Validate consistency between related tables."""
        logger.info("🔗 Validating table consistency...")

        # Check critical consistency issues
        source_docs = table_states.get("RAG.SourceDocuments", TableState("", 0, False, False))
        ifind_docs = table_states.get("RAG.SourceDocumentsIFind", TableState("", 0, False, False))

        if source_docs.exists and ifind_docs.exists:
            if source_docs.row_count != ifind_docs.row_count:
                ratio = ifind_docs.row_count / source_docs.row_count if source_docs.row_count > 0 else 0
                severity = ValidationSeverity.CRITICAL if ratio < 0.5 else ValidationSeverity.ERROR

                self._add_result(
                    "table_consistency_ifind",
                    severity,
                    "fail",
                    f"IFind table severely out of sync: {ifind_docs.row_count} vs {source_docs.row_count} ({ratio:.1%})",
                    {"source_count": source_docs.row_count, "ifind_count": ifind_docs.row_count, "ratio": ratio},
                )

        # Check graph data consistency
        entities = table_states.get("RAG.DocumentEntities", TableState("", 0, False, False))
        nodes = table_states.get("RAG.KnowledgeGraphNodes", TableState("", 0, False, False))

        if source_docs.exists and entities.exists:
            if source_docs.row_count > 100 and entities.row_count < 50:
                self._add_result(
                    "table_consistency_graph",
                    ValidationSeverity.ERROR,
                    "fail",
                    f"Graph data severely underpopulated: {entities.row_count} entities for {source_docs.row_count} documents",
                )

    def _validate_referential_integrity(self, table_states: Dict[str, TableState]):
        """Validate referential integrity between tables."""
        logger.info("🔗 Validating referential integrity...")

        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()

        # Check that all chunks reference valid documents
        try:
            cursor.execute(
                """
                SELECT COUNT(*) FROM RAG.DocumentChunks c
                LEFT JOIN RAG.SourceDocuments d ON c.doc_id = d.doc_id
                WHERE d.doc_id IS NULL
            """
            )
            orphaned_chunks = cursor.fetchone()[0]

            if orphaned_chunks > 0:
                self._add_result(
                    "referential_integrity_chunks",
                    ValidationSeverity.ERROR,
                    "fail",
                    f"Found {orphaned_chunks} orphaned chunks with no corresponding documents",
                )

        except Exception as e:
            self._add_result(
                "referential_integrity_chunks",
                ValidationSeverity.WARNING,
                "warning",
                f"Could not validate chunk integrity: {e}",
            )

        # Check that all token embeddings reference valid documents
        try:
            cursor.execute(
                """
                SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings t
                LEFT JOIN RAG.SourceDocuments d ON t.doc_id = d.doc_id
                WHERE d.doc_id IS NULL
            """
            )
            orphaned_tokens = cursor.fetchone()[0]

            if orphaned_tokens > 0:
                self._add_result(
                    "referential_integrity_tokens",
                    ValidationSeverity.ERROR,
                    "fail",
                    f"Found {orphaned_tokens} orphaned token embeddings with no corresponding documents",
                )

        except Exception as e:
            self._add_result(
                "referential_integrity_tokens",
                ValidationSeverity.WARNING,
                "warning",
                f"Could not validate token embedding integrity: {e}",
            )

        cursor.close()

    def _validate_pipeline_readiness(self, table_states: Dict[str, TableState]):
        """Validate readiness for each pipeline type."""
        logger.info("🚀 Validating pipeline readiness...")

        for pipeline_name, requirements in self.pipeline_requirements.items():
            pipeline_ready = True
            issues = []

            # Check required tables exist
            for table in requirements["required_tables"]:
                if not table_states.get(table, TableState("", 0, False, False)).exists:
                    pipeline_ready = False
                    issues.append(f"Missing table: {table}")

            # Check minimum row counts
            for table, min_rows in requirements["minimum_rows"].items():
                table_state = table_states.get(table, TableState("", 0, False, False))
                if table_state.row_count < min_rows:
                    pipeline_ready = False
                    issues.append(f"{table} has {table_state.row_count} rows, needs >= {min_rows}")

            # Check consistency requirements
            if "consistency_requirements" in requirements:
                for table1, table2, expected_ratio in requirements["consistency_requirements"]:
                    state1 = table_states.get(table1, TableState("", 0, False, False))
                    state2 = table_states.get(table2, TableState("", 0, False, False))

                    if state1.row_count > 0:
                        actual_ratio = state2.row_count / state1.row_count
                        if abs(actual_ratio - expected_ratio) > 0.1:  # 10% tolerance
                            pipeline_ready = False
                            issues.append(f"{table1}:{table2} ratio is {actual_ratio:.2f}, expected {expected_ratio}")

            # Record pipeline readiness
            if pipeline_ready:
                self._add_result(
                    f"pipeline_readiness_{pipeline_name}",
                    ValidationSeverity.INFO,
                    "pass",
                    f"{pipeline_name} pipeline is ready",
                )
            else:
                severity = (
                    ValidationSeverity.CRITICAL if pipeline_name in ["basic", "colbert"] else ValidationSeverity.ERROR
                )
                self._add_result(
                    f"pipeline_readiness_{pipeline_name}",
                    severity,
                    "fail",
                    f"{pipeline_name} pipeline is NOT ready: {'; '.join(issues)}",
                )

    def _detect_silent_failures(self, table_states: Dict[str, TableState]):
        """Detect conditions that could lead to silent failures."""
        logger.info("🕵️ Detecting silent failure conditions...")

        source_docs = table_states.get("RAG.SourceDocuments", TableState("", 0, False, False))

        # Detect partial IFind population (silent degradation)
        ifind_docs = table_states.get("RAG.SourceDocumentsIFind", TableState("", 0, False, False))
        if source_docs.exists and ifind_docs.exists and source_docs.row_count > 0:
            ratio = ifind_docs.row_count / source_docs.row_count
            if 0.1 < ratio < 0.9:  # Between 10% and 90% - suspicious
                self._add_result(
                    "silent_failure_ifind_partial",
                    ValidationSeverity.CRITICAL,
                    "fail",
                    f"IFind table partially populated ({ratio:.1%}) - will cause silent degradation in HybridIFind",
                )

        # Detect minimal graph data (silent GraphRAG failure)
        entities = table_states.get("RAG.DocumentEntities", TableState("", 0, False, False))
        if source_docs.row_count > 100 and entities.row_count < 20:
            self._add_result(
                "silent_failure_graph_minimal",
                ValidationSeverity.CRITICAL,
                "fail",
                f"Minimal graph data ({entities.row_count} entities) will cause GraphRAG to silently fall back",
            )

        # Detect token embedding gaps
        tokens = table_states.get("RAG.DocumentTokenEmbeddings", TableState("", 0, False, False))
        if source_docs.exists and tokens.exists and source_docs.row_count > 0:
            # Rough estimate: expect ~90 tokens per document on average
            expected_tokens = source_docs.row_count * 90
            if tokens.row_count < expected_tokens * 0.5:
                self._add_result(
                    "silent_failure_token_gaps",
                    ValidationSeverity.WARNING,
                    "warning",
                    f"Token embeddings may be incomplete: {tokens.row_count} vs expected ~{expected_tokens}",
                )

    def _validate_data_quality(self, table_states: Dict[str, TableState]):
        """Validate data quality in critical tables."""
        logger.info("✨ Validating data quality...")

        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()

        # Check for empty content in source documents
        try:
            cursor.execute(
                "SELECT COUNT(*) FROM RAG.SourceDocuments WHERE text_content IS NULL OR LENGTH(text_content) < 10"
            )
            empty_docs = cursor.fetchone()[0]

            if empty_docs > 0:
                self._add_result(
                    "data_quality_empty_content",
                    ValidationSeverity.WARNING,
                    "warning",
                    f"Found {empty_docs} documents with empty or very short content",
                )
        except Exception as e:
            self._add_result(
                "data_quality_empty_content",
                ValidationSeverity.WARNING,
                "warning",
                f"Could not validate content quality: {e}",
            )

        # Check for null embeddings
        try:
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NULL")
            null_embeddings = cursor.fetchone()[0]

            if null_embeddings > 0:
                self._add_result(
                    "data_quality_null_embeddings",
                    ValidationSeverity.ERROR,
                    "fail",
                    f"Found {null_embeddings} documents with null embeddings",
                )
        except Exception as e:
            self._add_result(
                "data_quality_null_embeddings",
                ValidationSeverity.WARNING,
                "warning",
                f"Could not validate embedding quality: {e}",
            )

        cursor.close()

    def _add_result(
        self, check_name: str, severity: ValidationSeverity, status: str, message: str, details: Optional[Dict] = None
    ):
        """Add a validation result."""
        result = ValidationResult(check_name, severity, status, message, details)
        self.validation_results.append(result)

        # Log based on severity
        if severity == ValidationSeverity.CRITICAL:
            logger.error(f"🔴 CRITICAL: {message}")
        elif severity == ValidationSeverity.ERROR:
            logger.error(f"🟠 ERROR: {message}")
        elif severity == ValidationSeverity.WARNING:
            logger.warning(f"🟡 WARNING: {message}")
        else:
            logger.info(f"🟢 {message}")

    def _log_validation_summary(self):
        """Log a summary of validation results."""
        total = len(self.validation_results)
        passed = len([r for r in self.validation_results if r.status == "pass"])
        failed = len([r for r in self.validation_results if r.status == "fail"])
        warnings = len([r for r in self.validation_results if r.status == "warning"])

        critical_issues = len([r for r in self.validation_results if r.severity == ValidationSeverity.CRITICAL])

        logger.info("=" * 80)
        logger.info("🎯 DATABASE STATE VALIDATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Checks: {total}")
        logger.info(f"✅ Passed: {passed}")
        logger.info(f"❌ Failed: {failed}")
        logger.info(f"⚠️ Warnings: {warnings}")
        logger.info(f"🔴 Critical Issues: {critical_issues}")

        if critical_issues > 0:
            logger.error("🚨 CRITICAL ISSUES DETECTED - Database state is inconsistent!")
        elif failed > 0:
            logger.warning("⚠️ ERRORS DETECTED - Some pipelines may not work correctly")
        elif warnings > 0:
            logger.info("✨ VALIDATION COMPLETED with warnings")
        else:
            logger.info("🎉 VALIDATION PASSED - Database state is consistent!")

    def get_pipeline_readiness_report(self) -> Dict[str, Dict[str, Any]]:
        """Get a comprehensive pipeline readiness report."""
        pipeline_results = {}

        for pipeline_name in self.pipeline_requirements.keys():
            pipeline_checks = [
                r for r in self.validation_results if r.check_name.startswith(f"pipeline_readiness_{pipeline_name}")
            ]

            if pipeline_checks:
                result = pipeline_checks[0]
                pipeline_results[pipeline_name] = {
                    "ready": result.status == "pass",
                    "status": result.status,
                    "message": result.message,
                    "severity": result.severity.value,
                }
            else:
                pipeline_results[pipeline_name] = {
                    "ready": False,
                    "status": "unknown",
                    "message": "No validation performed",
                    "severity": "error",
                }

        return pipeline_results

    def requires_data_sync(self) -> Tuple[bool, List[str]]:
        """
        Check if database requires data synchronization.

        Returns:
            Tuple of (needs_sync, list_of_issues)
        """
        sync_issues = []
        needs_sync = False

        for result in self.validation_results:
            if result.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]:
                if "consistency" in result.check_name or "silent_failure" in result.check_name:
                    sync_issues.append(result.message)
                    needs_sync = True

        return needs_sync, sync_issues


def validate_database_state(connection_manager, schema_manager) -> DatabaseStateValidator:
    """
    Convenience function to run complete database state validation.

    Args:
        connection_manager: Database connection manager
        schema_manager: Schema manager instance

    Returns:
        DatabaseStateValidator with completed validation results
    """
    validator = DatabaseStateValidator(connection_manager, schema_manager)
    validator.validate_complete_state()
    return validator
