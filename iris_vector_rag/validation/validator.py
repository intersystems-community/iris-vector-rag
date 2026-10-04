"""
Pre-condition validator for RAG pipelines.

This module provides validation logic to ensure pipelines have all required
data and dependencies before execution.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from ..core.connection import ConnectionManager
from ..storage.catalog import CatalogReader, IRISCatalogReader
from .requirements import EmbeddingRequirement, PipelineRequirements, TableRequirement

logger = logging.getLogger(__name__)

# The share of rows that must carry an embedding for the requirement to be met.
MIN_EMBEDDING_COMPLETENESS = 0.95


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
    # What a setup run could not do, in the words of the error. Empty for a plain
    # validation; `SetupOrchestrator.setup_pipeline` fills it so a report that is still
    # invalid says why instead of leaving the reason in a log.
    setup_errors: List[str] = field(default_factory=list)
    # The keys of `table_validations` and `embedding_validations` that gate
    # `overall_valid`: in the requirements' required lists and flagged `required`.
    required_names: Set[str] = field(default_factory=set)


class PreConditionValidator:
    """
    Validates that pipelines have all required data and dependencies.

    This validator checks:
    - Required tables exist and have data
    - Required embeddings are present and valid
    - Data integrity and completeness
    """

    def __init__(
        self,
        connection_manager: ConnectionManager,
        catalog: Optional[CatalogReader] = None,
    ):
        """
        Initialize the validator.

        Args:
            connection_manager: Database connection manager
            catalog: Reader used to tell a missing table from a failed query. One is
                built over the connection when not supplied; pass a recorded one in a
                test.
        """
        self.connection_manager = connection_manager
        self._catalog = catalog
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

        table_validations: Dict[str, ValidationResult] = {}
        embedding_validations: Dict[str, ValidationResult] = {}
        # `.required` decides what gates the pipeline, not which list a requirement
        # sits in: GraphRAG lists `Communities` among its required tables with
        # `required=False`, and that flag is the truth.
        required_names: Set[str] = set()

        for table_req in requirements.required_tables:
            key = table_req.name if table_req.required else f"{table_req.name}_optional"
            table_validations[key] = self._validate_table_requirement(table_req)
            if table_req.required:
                required_names.add(key)

        for table_req in requirements.optional_tables:
            table_validations[f"{table_req.name}_optional"] = (
                self._validate_table_requirement(table_req)
            )

        for embedding_req in requirements.required_embeddings:
            key = (
                embedding_req.name
                if embedding_req.required
                else f"{embedding_req.name}_optional"
            )
            embedding_validations[key] = self._validate_embedding_requirement(
                embedding_req
            )
            if embedding_req.required:
                required_names.add(key)

        for embedding_req in requirements.optional_embeddings:
            embedding_validations[f"{embedding_req.name}_optional"] = (
                self._validate_embedding_requirement(embedding_req)
            )

        overall_valid = all(
            result.is_valid
            for validations in (table_validations, embedding_validations)
            for name, result in validations.items()
            if name in required_names
        )

        summary = self._generate_summary(
            overall_valid, table_validations, embedding_validations, required_names
        )
        setup_suggestions = self._generate_setup_suggestions(
            requirements, table_validations, embedding_validations, required_names
        )

        return ValidationReport(
            pipeline_name=requirements.pipeline_name,
            overall_valid=overall_valid,
            table_validations=table_validations,
            embedding_validations=embedding_validations,
            summary=summary,
            setup_suggestions=setup_suggestions,
            required_names=required_names,
        )

    def _catalog_reader(self, connection) -> CatalogReader:
        return self._catalog or IRISCatalogReader(connection)

    def _table_is_present(self, connection, schema: str, name: str) -> Optional[bool]:
        """Whether the catalog lists the table; None if the catalog cannot say."""
        try:
            return self._catalog_reader(connection).snapshot(schema, name) is not None
        except Exception as exc:  # noqa: BLE001 - "cannot say" is the answer
            self.logger.warning(
                f"Could not read the catalog for {schema}.{name}: {exc}"
            )
            return None

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
                            "error_kind": "below_min_rows",
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
                # The count failed. Whether the table is missing is a different
                # question, asked of the catalog: a dropped connection or a timeout
                # also fails a COUNT, and "create the table" is wrong advice for it.
                present = self._table_is_present(
                    connection, table_req.schema, table_req.name
                )
                if present is False:
                    return ValidationResult(
                        is_valid=False,
                        message=f"Table {table_name} does not exist: {e}",
                        details={"error": str(e), "error_kind": "missing_table"},
                        suggestions=[
                            f"Create table {table_name}",
                            "Check database permissions",
                        ],
                    )
                return ValidationResult(
                    is_valid=False,
                    message=(
                        f"Could not count the rows of {table_name}"
                        f"{'' if present else ' (and could not confirm whether it exists)'}"
                        f": {e}"
                    ),
                    details={
                        "error": str(e),
                        "error_kind": "count_failed",
                        "table_exists": present,
                    },
                    suggestions=[
                        f"Check that {table_name} is readable by this user",
                        "Check the database connection",
                    ],
                )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                message=f"Database connection error: {e}",
                details={"error": str(e), "error_kind": "connection_error"},
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
            if completeness_ratio < MIN_EMBEDDING_COMPLETENESS:
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
        required_names: Set[str],
    ) -> str:
        """Summarise the result; only what gates the pipeline is called an issue."""
        if overall_valid:
            return "All requirements satisfied. Pipeline ready for execution."

        issues = []

        failed_tables = [
            name
            for name, result in table_validations.items()
            if name in required_names and not result.is_valid
        ]
        if failed_tables:
            issues.append(f"Table issues: {', '.join(failed_tables)}")

        failed_embeddings = [
            name
            for name, result in embedding_validations.items()
            if name in required_names and not result.is_valid
        ]
        if failed_embeddings:
            issues.append(f"Embedding issues: {', '.join(failed_embeddings)}")

        return f"Pipeline not ready. Issues: {'; '.join(issues)}"

    def _generate_setup_suggestions(
        self,
        requirements: PipelineRequirements,
        table_validations: Dict[str, ValidationResult],
        embedding_validations: Dict[str, ValidationResult],
        required_names: Set[str],
    ) -> List[str]:
        """Generate setup suggestions based on validation results."""
        suggestions = []

        for result in table_validations.values():
            suggestions.extend(result.suggestions)

        for result in embedding_validations.values():
            suggestions.extend(result.suggestions)

        # Pipeline-level advice only when something that gates the pipeline failed.
        if not all(
            result.is_valid
            for name, result in table_validations.items()
            if name in required_names
        ):
            suggestions.append(
                f"Run setup orchestrator for {requirements.pipeline_name}"
            )

        if not all(
            result.is_valid
            for name, result in embedding_validations.items()
            if name in required_names
        ):
            suggestions.append(
                "Run SetupOrchestrator.setup_pipeline("
                f"{requirements.pipeline_name!r}) to generate missing embeddings"
            )

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
