"""
Setup orchestrator for automating pipeline data preparation.

This module provides automated setup workflows to generate missing embeddings
and prepare data for different pipeline types.

What it reports is what happened. A step that cannot do its job raises `SetupError`
or is recorded in `SetupOrchestrator.setup_errors`, which `setup_pipeline` returns on
the report; it never inserts a row so that a minimum is met, and it never treats a
verification it could not run as a pass.
"""

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from ..config.manager import ConfigurationManager
from ..core.connection import ConnectionManager
from ..embeddings.manager import EmbeddingManager
from ..exceptions import RAGException
from ..storage.catalog import CatalogReader, IRISCatalogReader
from ..storage.shape_resolver import TableShapeResolver
from ..storage.table_shape import ColumnRole, TableShape
from .requirements import (
    EmbeddingRequirement,
    PipelineRequirements,
    TableRequirement,
    get_pipeline_requirements,
)
from .validator import (
    MIN_EMBEDDING_COMPLETENESS,
    PreConditionValidator,
    ValidationReport,
)

logger = logging.getLogger(__name__)

# Entities linked to each other in a document: the most confident few, not all of them,
# so a document with a hundred entities does not write five thousand edges.
_MAX_COOCCURRENCE_ENTITIES = 10


class SetupError(RAGException):
    """A setup step could not do what it was asked, and the caller must know."""


class SetupProgress:
    """Tracks progress of setup operations."""

    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.step_times = []

    def next_step(self, step_name: str):
        """Move to next step and log progress."""
        self.current_step += 1
        step_time = time.time()
        self.step_times.append(step_time)

        elapsed = step_time - self.start_time
        progress_pct = (self.current_step / self.total_steps) * 100

        logger.info(
            f"Step {self.current_step}/{self.total_steps} ({progress_pct:.1f}%): {step_name} - {elapsed:.1f}s elapsed"
        )

    def complete(self):
        """Mark setup as complete."""
        total_time = time.time() - self.start_time
        logger.info(f"Setup completed in {total_time:.1f} seconds")


class SetupOrchestrator:
    """
    Orchestrates automated setup of pipeline requirements.

    This class handles:
    - Creating the standard tables through `SchemaManager`
    - Generating document and chunk embeddings
    - Extracting entities for the graph pipelines (needs an LLM)
    - Progress tracking and error reporting
    """

    # Tables whose DDL lives in SchemaManager (canonical layout, the configured
    # embedding width). Setup creates these and no others: it holds no DDL of its own.
    _SCHEMA_MANAGER_TABLES = (
        "SourceDocuments",
        "DocumentChunks",
        "Entities",
        "EntityRelationships",
        "Communities",
    )

    def __init__(
        self,
        connection_manager: ConnectionManager,
        config_manager: ConfigurationManager,
        shapes: Optional[TableShapeResolver] = None,
        llm_func: Optional[Callable[[str], str]] = None,
    ):
        """
        Initialize the setup orchestrator.

        Args:
            connection_manager: Database connection manager
            config_manager: Configuration manager
            shapes: Resolver for this session's table shapes (optional). One is built
                from this orchestrator's own connection on first use when not
                supplied; pass a resolver over a `RecordedCatalog` to describe a
                shape in a test.
            llm_func: The LLM entity extraction calls (optional). Without one the
                graph pipelines' Entities and EntityRelationships tables cannot be
                filled, and setup says so instead of inventing rows.
        """
        self.connection_manager = connection_manager
        self.config_manager = config_manager
        self.embedding_manager = EmbeddingManager(config_manager)
        self.validator = PreConditionValidator(connection_manager)
        self.logger = logging.getLogger(__name__)
        # The document table this orchestrator prepares, and the one description of
        # it every statement below is written from (feature 082).
        self.documents_table = (config_manager.get("storage:iris", {}) or {}).get(
            "table_name", "RAG.SourceDocuments"
        )
        self.shapes = shapes
        self.llm_func = llm_func
        # What the current `setup_pipeline` run could not do.
        self.setup_errors: List[str] = []
        self._entities_extracted = False

    def _shape_resolver(self) -> TableShapeResolver:
        if self.shapes is None:
            self.shapes = TableShapeResolver(self._catalog_for())
        return self.shapes

    def _catalog_for(self) -> CatalogReader:
        return IRISCatalogReader(self.connection_manager.get_connection())

    def _documents_shape(self) -> TableShape:
        """The document table's shape, read from the database once.

        A missing table raises `TableNotFound` naming it, which is the error the
        caller needs: the code this replaces went on to build SQL out of an empty
        column set and failed several statements later with a column-not-found.
        """
        if "." in self.documents_table:
            schema, _, table = self.documents_table.partition(".")
        else:
            schema, table = "RAG", self.documents_table
        return self._shape_resolver().resolve(schema, table)

    def _record(self, message: str) -> None:
        """Note a step that did not do its job: logged, and returned on the report."""
        self.logger.error(message)
        self.setup_errors.append(message)

    def setup_pipeline(
        self,
        pipeline_type: str,
        auto_fix: bool = True,
        llm_func: Optional[Callable[[str], str]] = None,
    ) -> ValidationReport:
        """
        Set up all requirements for a pipeline type.

        Args:
            pipeline_type: Type of pipeline to set up
            auto_fix: Whether to automatically fix issues
            llm_func: The LLM for entity extraction; replaces the one given at
                construction when set

        Returns:
            Final validation report. Its `setup_errors` lists every step that could
            not do its job; a report that is still invalid says why.
        """
        self.logger.info(f"Setting up pipeline: {pipeline_type}")
        if llm_func is not None:
            self.llm_func = llm_func
        self.setup_errors = []
        self._entities_extracted = False

        requirements = get_pipeline_requirements(pipeline_type)

        initial_report = self.validator.validate_pipeline_requirements(requirements)

        if initial_report.overall_valid:
            self.logger.info(f"Pipeline {pipeline_type} already ready")
            return initial_report

        if not auto_fix:
            self.logger.info("Auto-fix disabled, returning validation report")
            return initial_report

        self._fulfill_requirements(requirements)

        # Optional chunking enhancement
        self._setup_optional_chunking(requirements)

        final_report = self.validator.validate_pipeline_requirements(requirements)
        final_report.setup_errors = list(self.setup_errors)

        if final_report.overall_valid:
            self.logger.info(f"Pipeline {pipeline_type} setup completed successfully")
        else:
            self.logger.warning(f"Pipeline {pipeline_type} setup completed with issues")

        return final_report

    def _fulfill_requirements(self, requirements: PipelineRequirements):
        """
        Generic requirements fulfillment based on declared requirements.

        A step that cannot be done is recorded in `setup_errors` and the next one
        runs; the final validation then says which requirements are still unmet.

        Args:
            requirements: Pipeline requirements to fulfill
        """
        total_steps = len(requirements.required_tables) + len(
            requirements.required_embeddings
        )
        progress = SetupProgress(total_steps)

        for table_req in requirements.required_tables:
            progress.next_step(f"Setting up table: {table_req.name}")
            self._fulfill_table_requirement(table_req)

        for embedding_req in requirements.required_embeddings:
            progress.next_step(f"Setting up embeddings: {embedding_req.name}")
            self._fulfill_embedding_requirement(embedding_req)

        # Data requirements (minimum rows), e.g. entities for the graph pipelines
        for table_req in requirements.required_tables:
            if table_req.required and table_req.min_rows > 0:
                self._fulfill_data_requirement(table_req, requirements.pipeline_name)

        progress.complete()
        self.logger.info(
            f"Requirements fulfillment completed for {requirements.pipeline_name}"
        )

    def _fulfill_table_requirement(self, table_req: TableRequirement):
        """Create the table through SchemaManager if it does not exist.

        A requirement flagged `required=False` is not created: nothing needs it.
        """
        if not table_req.required:
            self.logger.info(f"Table {table_req.name} is optional, not creating it")
            return

        self.logger.info(f"Fulfilling table requirement: {table_req.name}")
        try:
            if self._table_exists(table_req.schema, table_req.name):
                self.logger.info(
                    f"Table {table_req.schema}.{table_req.name} already exists"
                )
                return
            self._create_table(table_req.schema, table_req.name)
            self.logger.info(f"Created table: {table_req.schema}.{table_req.name}")
        except SetupError as exc:
            self._record(str(exc))

    def _table_exists(self, schema: str, table: str) -> bool:
        """Whether the catalog lists the table.

        A catalog that cannot be read raises: "could not look" is not "absent", and
        the caller would go on to create a table that may be there.
        """
        try:
            return self._catalog_for().snapshot(schema, table) is not None
        except Exception as exc:
            raise SetupError(
                f"could not check whether {schema}.{table} exists: {exc}"
            ) from exc

    def _create_table(self, schema: str, table: str) -> None:
        """Create a standard table in its canonical layout, through SchemaManager.

        Raises:
            SetupError: the table is not one SchemaManager owns, or it could not be
                created.
        """
        if table not in self._SCHEMA_MANAGER_TABLES:
            raise SetupError(
                f"setup cannot create {schema}.{table}: it has no canonical DDL here "
                f"(only {', '.join(self._SCHEMA_MANAGER_TABLES)} are created by setup)"
            )

        from iris_vector_rag.storage.schema_manager import SchemaManager

        schema_manager = SchemaManager(
            self.connection_manager, self.config_manager, schema_prefix=schema
        )
        if not schema_manager.ensure_table_schema(table):
            raise SetupError(f"SchemaManager could not create {schema}.{table}")

    def _fulfill_embedding_requirement(self, embedding_req: EmbeddingRequirement):
        """Generate the embeddings a requirement asks for, where setup can."""
        if not embedding_req.required:
            return
        if (
            embedding_req.table == "RAG.SourceDocuments"
            and embedding_req.column == "embedding"
        ):
            try:
                self._ensure_document_embeddings()
            except SetupError as exc:
                self._record(str(exc))
        else:
            self._record(
                f"setup has no generator for {embedding_req.table}.{embedding_req.column}"
            )

    def _setup_optional_chunking(self, requirements: PipelineRequirements):
        """Set up the optional chunking enhancement if the pipeline lists it.

        Optional, so a failure here is recorded and does not stop setup; it is on
        the report either way.
        """
        has_chunk_table = any(
            table.name == "DocumentChunks" for table in requirements.optional_tables
        )
        has_chunk_embeddings = any(
            emb.name == "chunk_embeddings" for emb in requirements.optional_embeddings
        )

        if not (has_chunk_table or has_chunk_embeddings):
            return

        self.logger.info("Setting up optional chunking enhancement")
        progress = SetupProgress(3)
        try:
            progress.next_step("Creating chunks table")
            self._create_chunks_table()

            progress.next_step("Generating document chunks")
            chunk_count = self._generate_document_chunks()

            progress.next_step("Validating chunk embeddings")
            if chunk_count > 0:
                self._validate_embeddings_after_generation(
                    f"{self._documents_shape().schema}.DocumentChunks",
                    "chunk_embedding",
                    "chunk",
                )
            progress.complete()
        except SetupError as exc:
            self._record(f"optional chunking failed: {exc}")

    def _ensure_document_embeddings(self):
        """Ensure all documents have embeddings."""
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            shape = self._documents_shape()
            table = shape.qualified_name
            identifier = shape.column_for(ColumnRole.IDENTIFIER).name
            # Whether the embeddings live in this table or in a separate vector table
            # is a fact about the table, so the table is what gets asked.
            embedding = shape.optional_column_for(ColumnRole.EMBEDDING)
            vector_table = f"{shape.schema}.VectorEmbeddings"

            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            self.logger.info(f"Total documents in {table}: {cursor.fetchone()[0]}")

            if embedding is not None:
                cursor.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE {embedding.name} IS NULL"
                )
                missing_count = cursor.fetchone()[0]
            else:
                # Clean schema approach - check VectorEmbeddings table
                try:
                    cursor.execute(
                        f"SELECT COUNT(*) FROM {table} s "
                        f"LEFT JOIN {vector_table} v ON s.{identifier} = v.chunk_id "
                        "WHERE v.embedding IS NULL OR v.id IS NULL"
                    )
                    missing_count = cursor.fetchone()[0]
                except Exception as e:
                    self.logger.warning(f"VectorEmbeddings join failed: {e}")
                    # Fallback: count every document, since none of them can be shown
                    # to have an embedding.
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    missing_count = cursor.fetchone()[0]

            if missing_count > 0:
                self.logger.info(f"Generating embeddings for {missing_count} documents")
                self._generate_missing_document_embeddings()

                # Validate that embeddings were actually generated
                if embedding is not None:
                    self._validate_embeddings_after_generation(
                        table, embedding.name, "document"
                    )
                else:
                    self._validate_embeddings_after_generation(
                        vector_table, "embedding", "document"
                    )
            else:
                self.logger.info("All documents have embeddings")

        finally:
            cursor.close()

    def _generate_missing_document_embeddings(self):
        """Generate embeddings for documents that don't have them.

        Raises:
            SetupError: some documents still have no embedding afterwards, or the
                count could not be taken. "Could not verify" is not success.
        """
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            # One description of the table, and every statement below written from it:
            # the identifier selected here is the identifier the UPDATE matches on, and
            # the text embedded is the text the store reads back.
            shape = self._documents_shape()
            table = shape.qualified_name
            identifier = shape.column_for(ColumnRole.IDENTIFIER).name
            text_column = shape.column_for(ColumnRole.TEXT).name
            embedding_column = shape.optional_column_for(ColumnRole.EMBEDDING)
            vector_table = f"{shape.schema}.VectorEmbeddings"

            if embedding_column is not None:
                cursor.execute(
                    f"SELECT {identifier}, {text_column} FROM {table} "
                    f"WHERE {embedding_column.name} IS NULL"
                )
                documents = cursor.fetchall()
            else:
                # Clean schema approach - the embeddings live in their own table, so
                # every document without a row there needs one.
                try:
                    cursor.execute(
                        f"SELECT s.{identifier}, s.{text_column} FROM {table} s "
                        f"LEFT JOIN {vector_table} v ON s.{identifier} = v.chunk_id "
                        "WHERE v.embedding IS NULL OR v.id IS NULL"
                    )
                    documents = cursor.fetchall()
                except Exception as e:
                    self.logger.warning(
                        f"VectorEmbeddings join failed: {e}, using fallback"
                    )
                    # Fallback: every document, since none can be shown to have one.
                    cursor.execute(f"SELECT {identifier}, {text_column} FROM {table}")
                    documents = cursor.fetchall()
            if not documents:
                self.logger.info("No documents missing embeddings")
                return

            batch_size = 32
            total_processed = 0
            total_failed = 0

            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]

                # Generate embeddings for batch
                texts = [
                    doc[1] if doc[1] else "" for doc in batch
                ]  # Handle None content

                try:
                    embeddings = self.embedding_manager.embed_texts(texts)

                    # Update database with consistent vector format
                    for (doc_id, _), embedding in zip(batch, embeddings):
                        try:
                            # Ensure embedding is a list
                            if hasattr(embedding, "tolist"):
                                embedding = embedding.tolist()
                            elif not isinstance(embedding, list):
                                embedding = list(embedding)

                            # Use consistent vector format that validator expects
                            vector_str = f"[{','.join(map(str, embedding))}]"

                            # Written back through the column it was selected by: the
                            # `WHERE doc_id = ?` this replaces was fixed text, so on a
                            # table keyed by `id` it updated nothing and reported
                            # success.
                            if embedding_column is not None:
                                cursor.execute(
                                    f"UPDATE {table} "
                                    f"SET {embedding_column.name} = TO_VECTOR(?) "
                                    f"WHERE {identifier} = ?",
                                    [vector_str, doc_id],
                                )
                            else:
                                # Clean schema approach - insert into VectorEmbeddings table
                                cursor.execute(
                                    f"INSERT INTO {vector_table} "
                                    "(chunk_id, embedding_model, embedding) "
                                    "VALUES (?, 'validation-generated', TO_VECTOR(?))",
                                    [doc_id, vector_str],
                                )
                            total_processed += 1

                        except Exception as doc_error:
                            self.logger.warning(
                                f"Failed to update embedding for {doc_id}: {doc_error}"
                            )
                            total_failed += 1

                    connection.commit()
                    self.logger.info(
                        f"Generated embeddings for batch {i // batch_size + 1}/{(len(documents) + batch_size - 1) // batch_size}"
                    )

                except Exception as batch_error:
                    self.logger.error(
                        f"Failed to generate embeddings for batch {i // batch_size + 1}: {batch_error}"
                    )
                    total_failed += len(batch)
                    connection.rollback()

            # Final verification
            if embedding_column is not None:
                cursor.execute(
                    f"SELECT COUNT(*) FROM {table} "
                    f"WHERE {embedding_column.name} IS NULL"
                )
                remaining_missing = cursor.fetchone()[0]
            else:
                # Clean schema approach - check VectorEmbeddings table
                try:
                    cursor.execute(
                        f"SELECT COUNT(*) FROM {table} s "
                        f"LEFT JOIN {vector_table} v ON s.{identifier} = v.chunk_id "
                        "WHERE v.embedding IS NULL OR v.id IS NULL"
                    )
                    remaining_missing = cursor.fetchone()[0]
                except Exception as e:
                    self.logger.warning(
                        f"VectorEmbeddings verification failed: {e}, using fallback"
                    )
                    # Compare row counts; if even that cannot be read, the result is
                    # unknown and setup says so rather than assuming success.
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {vector_table}")
                        embedding_count = cursor.fetchone()[0]
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        doc_count = cursor.fetchone()[0]
                        remaining_missing = max(0, doc_count - embedding_count)
                    except Exception as count_error:
                        raise SetupError(
                            f"could not verify the embeddings written to {table}: "
                            f"{count_error}"
                        ) from count_error

            self.logger.info(
                f"Document embedding generation complete: {total_processed} processed, {total_failed} failed, {remaining_missing} still missing"
            )

            if remaining_missing > 0:
                raise SetupError(
                    f"{remaining_missing} documents in {table} still have no embedding "
                    f"after generation ({total_processed} written, {total_failed} failed)"
                )

        finally:
            cursor.close()

    def heal_token_embeddings(
        self, target_doc_count: int = 1000, force_regenerate: bool = False
    ) -> Dict[str, Any]:
        """Not implemented: this method wrote the wrong table's columns.

        It embedded each whitespace-separated word of a document as a "token" and
        inserted `(doc_id, token_index, token_text, token_embedding)` rows into
        `RAG.DocumentTokenEmbeddings`, whose columns are `(doc_id, tok_pos, tok_text,
        tok_vec, centroid_id)`. Every insert failed and the failure was logged at
        debug level. Token embeddings come from a ColBERT model, through
        `iris_vector_rag.pipelines.colbert_iris.ingest`.

        Raises:
            NotImplementedError: always.
        """
        raise NotImplementedError(
            "SetupOrchestrator.heal_token_embeddings() is not implemented: token "
            "embeddings for RAG.DocumentTokenEmbeddings are written by the ColBERT "
            "pipeline's ingest (iris_vector_rag.pipelines.colbert_iris.ingest), not "
            "by setup"
        )

    def _create_chunks_table(self):
        """Create the chunk table, in its canonical layout, if it does not exist."""
        schema = self._documents_shape().schema
        if self._table_exists(schema, "DocumentChunks"):
            self.logger.info("DocumentChunks table already exists")
            return
        self._create_table(schema, "DocumentChunks")
        self.logger.info("Created DocumentChunks table")

    def _generate_document_chunks(self) -> int:
        """Generate document chunks with embeddings.

        Returns:
            The number of chunks in the chunk table afterwards.

        Raises:
            SetupError: a document could not be chunked or a chunk could not be
                stored. Chunks that were stored stay.
        """
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            shape = self._documents_shape()
            chunk_table = f"{shape.schema}.DocumentChunks"

            cursor.execute(f"SELECT COUNT(*) FROM {chunk_table}")
            existing_count = cursor.fetchone()[0]

            if existing_count > 0:
                self.logger.info(
                    f"Document chunks already exist ({existing_count} chunks)"
                )
                return existing_count

            # The text of each document is the column the table's shape says holds it,
            # not another column that happens to be there: the fallback this replaces
            # chunked and embedded the file names.
            identifier = shape.column_for(ColumnRole.IDENTIFIER).name
            text_column = shape.column_for(ColumnRole.TEXT).name
            cursor.execute(
                f"SELECT {identifier}, {text_column} FROM {shape.qualified_name}"
            )
            documents = cursor.fetchall()

            if not documents:
                self.logger.warning("No documents found for chunk generation")
                return 0

            chunk_size = self.config_manager.get("chunking:chunk_size", 1000)
            chunk_overlap = self.config_manager.get("chunking:chunk_overlap", 200)

            total_chunks_processed = 0
            failures: List[str] = []
            documents_processed = 0

            for doc_id, content in documents:
                if not content:
                    continue

                chunks = self._split_text_into_chunks(
                    content, chunk_size, chunk_overlap
                )
                if not chunks:
                    continue

                try:
                    chunk_embeddings = self.embedding_manager.embed_texts(chunks)
                except Exception as embedding_error:
                    failures.append(
                        f"embedding the chunks of {doc_id}: {embedding_error}"
                    )
                    continue

                # The canonical chunk layout (CANONICAL_CHUNK_SHAPE): `doc_id` is the
                # chunk's own key, `document_id` the parent document.
                for chunk_index, (chunk_text, embedding) in enumerate(
                    zip(chunks, chunk_embeddings)
                ):
                    try:
                        if hasattr(embedding, "tolist"):
                            embedding = embedding.tolist()
                        elif not isinstance(embedding, list):
                            embedding = list(embedding)
                        vector_str = f"[{','.join(map(str, embedding))}]"

                        cursor.execute(
                            f"INSERT INTO {chunk_table} "
                            "(doc_id, document_id, chunk_text, chunk_embedding, metadata) "
                            "VALUES (?, ?, ?, TO_VECTOR(?), ?)",
                            [
                                f"{doc_id}_chunk_{chunk_index}",
                                str(doc_id),
                                chunk_text,
                                vector_str,
                                json.dumps(
                                    {
                                        "chunk_index": chunk_index,
                                        "chunk_size": len(chunk_text),
                                        "parent_doc": str(doc_id),
                                    }
                                ),
                            ],
                        )
                        total_chunks_processed += 1
                    except Exception as chunk_error:
                        failures.append(
                            f"storing chunk {chunk_index} of {doc_id}: {chunk_error}"
                        )

                connection.commit()
                documents_processed += 1

                if documents_processed % 50 == 0:
                    self.logger.info(
                        f"Processed {documents_processed}/{len(documents)} documents, {total_chunks_processed} chunks generated"
                    )

            cursor.execute(f"SELECT COUNT(*) FROM {chunk_table}")
            final_count = cursor.fetchone()[0]

            self.logger.info(
                f"Chunk generation complete: {final_count} total chunks, {documents_processed} documents processed, {len(failures)} failures"
            )

            if failures:
                raise SetupError(
                    f"chunk generation had {len(failures)} failures "
                    f"({final_count} chunks stored); first: {failures[0]}"
                )
            return final_count

        finally:
            cursor.close()

    def _split_text_into_chunks(
        self, text: str, chunk_size: int, overlap: int
    ) -> List[str]:
        """Split text into overlapping chunks."""
        if text is None:
            text = ""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                sentence_end = text.rfind(".", start, end)
                if sentence_end > start:
                    end = sentence_end + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap
            if start <= 0:
                start = end

        return chunks

    def _validate_embeddings_after_generation(
        self, table: str, column: str, embedding_type: str
    ):
        """
        Check that embeddings were generated and are in the right format.

        Args:
            table: Table name to validate
            column: Column name containing embeddings
            embedding_type: Type of embedding for messages (e.g., "document", "chunk")

        Raises:
            SetupError: the table is empty, holds no embeddings, holds fewer than the
                validator's completeness threshold, holds them in the wrong format,
                or the check itself could not run. Setup does not go on as if it
                had worked.
        """
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            try:
                cursor.execute(f"""
                    SELECT
                        COUNT(*) as total_rows,
                        COUNT({column}) as rows_with_embeddings
                    FROM {table}
                """)
                result = cursor.fetchone()
                total_rows = result[0]
                rows_with_embeddings = result[1]

                sample = None
                if rows_with_embeddings:
                    cursor.execute(f"""
                        SELECT TOP 1 {column}
                        FROM {table}
                        WHERE {column} IS NOT NULL
                    """)
                    sample_result = cursor.fetchone()
                    sample = sample_result[0] if sample_result else None
            except Exception as exc:
                raise SetupError(
                    f"could not verify {embedding_type} embeddings in {table}.{column}: {exc}"
                ) from exc
        finally:
            cursor.close()

        if total_rows == 0:
            raise SetupError(
                f"there are no rows in {table} to hold {embedding_type} embeddings"
            )

        if rows_with_embeddings == 0:
            raise SetupError(
                f"no {embedding_type} embeddings were generated in {table}.{column} "
                f"({total_rows} rows)"
            )

        completeness_ratio = rows_with_embeddings / total_rows
        if completeness_ratio < MIN_EMBEDDING_COMPLETENESS:
            raise SetupError(
                f"{embedding_type} embeddings incomplete in {table}.{column}: "
                f"{rows_with_embeddings}/{total_rows} ({completeness_ratio:.1%})"
            )

        if sample:
            embedding_str = str(sample)
            # IRIS VECTOR columns return comma-separated values without brackets when
            # retrieved; that is the correct format.
            if "," in embedding_str and not embedding_str.startswith("["):
                pass
            elif embedding_str.startswith("[") and embedding_str.endswith("]"):
                raise SetupError(
                    f"{embedding_type} embeddings in {table}.{column} are stored as a "
                    f"string, not a VECTOR: {embedding_str[:100]}"
                )
            else:
                raise SetupError(
                    f"unrecognized {embedding_type} embedding format in "
                    f"{table}.{column}: {embedding_str[:100]}"
                )

        self.logger.info(
            f"{embedding_type.capitalize()} embeddings validation passed: "
            f"{rows_with_embeddings}/{total_rows} ({completeness_ratio:.1%})"
        )

    def _fulfill_data_requirement(
        self, table_req: TableRequirement, pipeline_name: str
    ):
        """
        Fulfill a minimum-rows requirement where setup can produce the rows.

        The only rows setup can produce are entities and relationships for the graph
        pipelines, extracted from the documents already loaded. Documents it cannot
        invent, and it does not fill a table with placeholder rows to meet a minimum:
        the requirement stays unmet and is recorded.

        Args:
            table_req: Table requirement with min_rows constraint
            pipeline_name: Name of the pipeline requesting the data
        """
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table_req.schema}.{table_req.name}")
            current_rows = cursor.fetchone()[0]
        finally:
            cursor.close()

        if current_rows >= table_req.min_rows:
            self.logger.info(
                f"Table {table_req.name} has sufficient data: {current_rows} rows (need {table_req.min_rows})"
            )
            return

        self.logger.info(
            f"Table {table_req.name} needs data: {current_rows} rows (need {table_req.min_rows})"
        )

        if pipeline_name in ["graphrag", "hybrid_graphrag"] and table_req.name in [
            "Entities",
            "EntityRelationships",
        ]:
            if self._entities_extracted:
                self._record(
                    f"{table_req.schema}.{table_req.name} still has {current_rows} rows "
                    f"(needs {table_req.min_rows}) after entity extraction"
                )
                return
            self._entities_extracted = True
            try:
                self._perform_entity_extraction()
            except SetupError as exc:
                self._record(str(exc))
        else:
            self._record(
                f"{table_req.schema}.{table_req.name} has {current_rows} rows (needs "
                f"{table_req.min_rows}) and setup cannot produce them for the "
                f"{pipeline_name} pipeline: load data into it"
            )

    def _perform_entity_extraction(self):
        """
        Extract entities and co-occurrence relationships from the loaded documents.

        Reads each document's text through the document table's shape, runs it through
        `EntityExtractionService` with the LLM given to this orchestrator, and stores
        what comes back. Each document's entities are linked to each other (the most
        confident `_MAX_COOCCURRENCE_ENTITIES`) by a `co_occurrence` relationship.

        Raises:
            SetupError: there is no LLM, there are no documents, or extraction or
                storage failed for some documents (the others' entities stay).
        """
        if self.llm_func is None:
            raise SetupError(
                "entity extraction needs an LLM and none was given: pass llm_func= to "
                "SetupOrchestrator, setup_pipeline() or create_pipeline()"
            )

        from ..core.models import Document, Relationship
        from ..services.entity_extraction import (
            EntityExtractionError,
            EntityExtractionService,
        )

        self.logger.info("Starting entity extraction for GraphRAG pipeline")

        service = EntityExtractionService(
            self.config_manager,
            connection_manager=self.connection_manager,
            llm_func=self.llm_func,
        )

        shape = self._documents_shape()
        identifier = shape.column_for(ColumnRole.IDENTIFIER).name
        text_column = shape.column_for(ColumnRole.TEXT).name
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()
        try:
            cursor.execute(
                f"SELECT {identifier}, {text_column} FROM {shape.qualified_name} "
                f"ORDER BY {identifier}"
            )
            documents = cursor.fetchall()
        finally:
            cursor.close()

        if not documents:
            raise SetupError(
                f"there are no documents in {shape.qualified_name} to extract entities from"
            )

        self.logger.info(f"Extracting entities from {len(documents)} documents")

        entities_found = 0
        relationships_found = 0
        failures: List[str] = []

        for doc_id, text in documents:
            if not text:
                continue
            try:
                entities = service.extract_entities(
                    Document(id=str(doc_id), page_content=str(text))
                )
                if not entities:
                    continue

                linked = sorted(entities, key=lambda e: -e.confidence)[
                    :_MAX_COOCCURRENCE_ENTITIES
                ]
                relationships = [
                    Relationship(
                        source_entity_id=first.id,
                        target_entity_id=second.id,
                        relationship_type="co_occurrence",
                        confidence=0.5,
                        source_document_id=str(doc_id),
                    )
                    for position, first in enumerate(linked)
                    for second in linked[position + 1 :]
                    if first.text.strip().lower() != second.text.strip().lower()
                ]
                service.store_entities_and_relationships(entities, relationships)
                entities_found += len(entities)
                relationships_found += len(relationships)
            except EntityExtractionError as exc:
                failures.append(f"{doc_id}: {exc}")

        self.logger.info(
            f"Entity extraction completed: {entities_found} entities, "
            f"{relationships_found} relationships"
        )

        if failures:
            raise SetupError(
                f"entity extraction failed for {len(failures)} of {len(documents)} "
                f"documents; first: {failures[0]}"
            )
