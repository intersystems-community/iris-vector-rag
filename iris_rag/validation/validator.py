"""
Pre-condition validator for RAG pipelines.

This module provides validation logic to ensure pipelines have all required
data and dependencies before execution.
"""

import logging
from typing import Dict, List, Any
from dataclasses import dataclass
from ..core.connection import ConnectionManager
from .requirements import PipelineRequirements, TableRequirement, EmbeddingRequirement

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""

    is_valid: bool
    message: str
    details: Dict[str, Any]
    suggestions: List[str]


@dataclass
class ValidationReport:
    """Complete validation report for a pipeline."""

    pipeline_name: str
    overall_valid: bool
    table_validations: Dict[str, ValidationResult]
    embedding_validations: Dict[str, ValidationResult]
    summary: str
    setup_suggestions: List[str]


class PreConditionValidator:
    """
    Validates that pipelines have all required data and dependencies.

    This validator checks:
    - Required tables exist and have data
    - Required embeddings are present and valid
    - Data integrity and completeness
    """

    def __init__(self, connection_manager: ConnectionManager):
        """
        Initialize the validator.

        Args:
            connection_manager: Database connection manager
        """
        self.connection_manager = connection_manager
        self.logger = logging.getLogger(__name__)

    def validate_pipeline_requirements(
        self, requirements: PipelineRequirements
    ) -> ValidationReport:
        """
        Validate all requirements for a pipeline.

        Args:
            requirements: Pipeline requirements to validate

        Returns:
            Complete validation report
        """
        self.logger.info(f"Validating requirements for {requirements.pipeline_name}")

        # Validate required tables
        table_validations = {}
        for table_req in requirements.required_tables:
            result = self._validate_table_requirement(table_req)
            table_validations[table_req.name] = result

        # Validate optional tables (for informational purposes)
        for table_req in requirements.optional_tables:
            result = self._validate_table_requirement(table_req)
            table_validations[f"{table_req.name}_optional"] = result

        # Validate required embeddings
        embedding_validations = {}
        for embedding_req in requirements.required_embeddings:
            result = self._validate_embedding_requirement(embedding_req)
            embedding_validations[embedding_req.name] = result

        # Validate optional embeddings (for informational purposes)
        for embedding_req in requirements.optional_embeddings:
            result = self._validate_embedding_requirement(embedding_req)
            embedding_validations[f"{embedding_req.name}_optional"] = result

        # Determine overall validity (only based on required components)
        required_table_valid = all(
            result.is_valid
            for name, result in table_validations.items()
            if not name.endswith("_optional")
        )
        required_embedding_valid = all(
            result.is_valid
            for name, result in embedding_validations.items()
            if not name.endswith("_optional")
        )
        overall_valid = required_table_valid and required_embedding_valid

        # Generate summary and suggestions
        summary = self._generate_summary(
            overall_valid, table_validations, embedding_validations
        )
        setup_suggestions = self._generate_setup_suggestions(
            requirements, table_validations, embedding_validations
        )

        return ValidationReport(
            pipeline_name=requirements.pipeline_name,
            overall_valid=overall_valid,
            table_validations=table_validations,
            embedding_validations=embedding_validations,
            summary=summary,
            setup_suggestions=setup_suggestions,
        )

    def _validate_table_requirement(
        self, table_req: TableRequirement
    ) -> ValidationResult:
        """
        Validate a single table requirement.

        Args:
            table_req: Table requirement to validate

        Returns:
            Validation result
        """
        try:
            connection = self.connection_manager.get_connection()
            cursor = connection.cursor()

            # Check if table exists
            table_name = f"{table_req.schema}.{table_req.name}"

            try:
                # Try to query the table
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]

                # Check minimum row requirement
                if row_count < table_req.min_rows:
                    return ValidationResult(
                        is_valid=False,
                        message=f"Table {table_name} has {row_count} rows, but requires at least {table_req.min_rows}",
                        details={
                            "row_count": row_count,
                            "min_required": table_req.min_rows,
                        },
                        suggestions=[f"Load more data into {table_name}"],
                    )

                return ValidationResult(
                    is_valid=True,
                    message=f"Table {table_name} exists with {row_count} rows",
                    details={"row_count": row_count},
                    suggestions=[],
                )

            except Exception as e:
                return ValidationResult(
                    is_valid=False,
                    message=f"Table {table_name} does not exist or is not accessible: {e}",
                    details={"error": str(e)},
                    suggestions=[
                        f"Create table {table_name}",
                        f"Check database permissions",
                    ],
                )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                message=f"Database connection error: {e}",
                details={"error": str(e)},
                suggestions=[
                    "Check database connection",
                    "Verify connection configuration",
                ],
            )
        finally:
            if "cursor" in locals():
                cursor.close()

    def _validate_embedding_requirement(
        self, embedding_req: EmbeddingRequirement
    ) -> ValidationResult:
        """
        Validate a single embedding requirement.

        Args:
            embedding_req: Embedding requirement to validate

        Returns:
            Validation result
        """
        try:
            connection = self.connection_manager.get_connection()
            cursor = connection.cursor()

            # Check if embedding column has data
            sql = f"""
                SELECT 
                    COUNT(*) as total_rows,
                    COUNT({embedding_req.column}) as rows_with_embeddings
                FROM {embedding_req.table}
            """

            cursor.execute(sql)
            result = cursor.fetchone()
            total_rows = result[0]
            rows_with_embeddings = result[1]

            if total_rows == 0:
                return ValidationResult(
                    is_valid=False,
                    message=f"No data in table {embedding_req.table}",
                    details={"total_rows": 0, "rows_with_embeddings": 0},
                    suggestions=[f"Load data into {embedding_req.table}"],
                )

            if rows_with_embeddings == 0:
                return ValidationResult(
                    is_valid=False,
                    message=f"No embeddings found in {embedding_req.table}.{embedding_req.column}",
                    details={"total_rows": total_rows, "rows_with_embeddings": 0},
                    suggestions=[f"Generate embeddings for {embedding_req.table}"],
                )

            # Check embedding completeness
            completeness_ratio = rows_with_embeddings / total_rows
            if completeness_ratio < 0.95:  # Less than 95% complete
                return ValidationResult(
                    is_valid=False,
                    message=f"Embeddings incomplete: {rows_with_embeddings}/{total_rows} ({completeness_ratio:.1%})",
                    details={
                        "total_rows": total_rows,
                        "rows_with_embeddings": rows_with_embeddings,
                        "completeness_ratio": completeness_ratio,
                    },
                    suggestions=[
                        f"Generate missing embeddings for {embedding_req.table}"
                    ],
                )

            # Validate embedding format (check for correct IRIS VECTOR format)
            sample_sql = f"""
                SELECT TOP 1 {embedding_req.column}
                FROM {embedding_req.table}
                WHERE {embedding_req.column} IS NOT NULL
            """

            cursor.execute(sample_sql)
            sample_result = cursor.fetchone()

            if sample_result and sample_result[0]:
                # Validate IRIS VECTOR format
                embedding_str = str(sample_result[0])

                # IRIS VECTOR columns return comma-separated values without brackets when retrieved
                # This is the correct format for VECTOR data in IRIS
                if "," in embedding_str and not embedding_str.startswith("["):
                    # This is the expected VECTOR format from IRIS - validation passes
                    pass
                elif embedding_str.startswith("[") and embedding_str.endswith("]"):
                    # This indicates string format, which should be converted to VECTOR
                    return ValidationResult(
                        is_valid=False,
                        message=f"Embeddings stored as strings instead of VECTOR type in {embedding_req.table}.{embedding_req.column}",
                        details={
                            "sample_embedding": embedding_str[:100],
                            "format_issue": "string_instead_of_vector",
                        },
                        suggestions=[
                            f"Convert embeddings to VECTOR type using TO_VECTOR() in {embedding_req.table}"
                        ],
                    )
                else:
                    # Invalid format
                    return ValidationResult(
                        is_valid=False,
                        message=f"Invalid embedding format in {embedding_req.table}.{embedding_req.column}",
                        details={
                            "sample_embedding": embedding_str[:100],
                            "format_issue": "unrecognized_format",
                        },
                        suggestions=[
                            f"Regenerate embeddings with correct VECTOR format for {embedding_req.table}"
                        ],
                    )

            return ValidationResult(
                is_valid=True,
                message=f"Embeddings valid: {rows_with_embeddings}/{total_rows} ({completeness_ratio:.1%})",
                details={
                    "total_rows": total_rows,
                    "rows_with_embeddings": rows_with_embeddings,
                    "completeness_ratio": completeness_ratio,
                },
                suggestions=[],
            )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                message=f"Error validating embeddings: {e}",
                details={"error": str(e)},
                suggestions=["Check database schema", "Verify embedding column exists"],
            )
        finally:
            if "cursor" in locals():
                cursor.close()

    def _generate_summary(
        self,
        overall_valid: bool,
        table_validations: Dict[str, ValidationResult],
        embedding_validations: Dict[str, ValidationResult],
    ) -> str:
        """Generate a summary of validation results."""
        if overall_valid:
            return "All requirements satisfied. Pipeline ready for execution."

        issues = []

        # Table issues
        failed_tables = [
            name for name, result in table_validations.items() if not result.is_valid
        ]
        if failed_tables:
            issues.append(f"Table issues: {', '.join(failed_tables)}")

        # Embedding issues
        failed_embeddings = [
            name
            for name, result in embedding_validations.items()
            if not result.is_valid
        ]
        if failed_embeddings:
            issues.append(f"Embedding issues: {', '.join(failed_embeddings)}")

        return f"Pipeline not ready. Issues: {'; '.join(issues)}"

    def _generate_setup_suggestions(
        self,
        requirements: PipelineRequirements,
        table_validations: Dict[str, ValidationResult],
        embedding_validations: Dict[str, ValidationResult],
    ) -> List[str]:
        """Generate setup suggestions based on validation results."""
        suggestions = []

        # Collect all suggestions from individual validations
        for result in table_validations.values():
            suggestions.extend(result.suggestions)

        for result in embedding_validations.values():
            suggestions.extend(result.suggestions)

        # Add pipeline-specific suggestions
        if not all(result.is_valid for result in table_validations.values()):
            suggestions.append(
                f"Run setup orchestrator for {requirements.pipeline_name}"
            )

        if not all(result.is_valid for result in embedding_validations.values()):
            suggestions.append("Use SetupOrchestrator.generate_missing_embeddings()")

        # Remove duplicates while preserving order
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion not in unique_suggestions:
                unique_suggestions.append(suggestion)

        return unique_suggestions

    def quick_validate(self, pipeline_type: str) -> bool:
        """
        Quick validation check for a pipeline type.

        Args:
            pipeline_type: Type of pipeline to validate

        Returns:
            True if pipeline is ready, False otherwise
        """
        from .requirements import get_pipeline_requirements

        try:
            requirements = get_pipeline_requirements(pipeline_type)
            report = self.validate_pipeline_requirements(requirements)
            return report.overall_valid
        except Exception as e:
            self.logger.error(f"Quick validation failed for {pipeline_type}: {e}")
            return False
