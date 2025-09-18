"""
Setup orchestrator for automating pipeline data preparation.

This module provides automated setup workflows to generate missing embeddings
and prepare data for different pipeline types.
"""

import logging
import time
from typing import Dict, List, Any
from ..core.connection import ConnectionManager
from ..config.manager import ConfigurationManager
from ..embeddings.manager import EmbeddingManager
from .requirements import PipelineRequirements, get_pipeline_requirements
from .validator import PreConditionValidator, ValidationReport

logger = logging.getLogger(__name__)


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
    - Creating chunked embeddings for chunked pipelines
    - Dependency resolution and setup ordering
    - Progress tracking and error handling
    """

    def __init__(
        self,
        connection_manager: ConnectionManager,
        config_manager: ConfigurationManager,
    ):
        """
        Initialize the setup orchestrator.

        Args:
            connection_manager: Database connection manager
            config_manager: Configuration manager
        """
        self.connection_manager = connection_manager
        self.config_manager = config_manager
        self.embedding_manager = EmbeddingManager(config_manager)
        self.validator = PreConditionValidator(connection_manager)
        self.logger = logging.getLogger(__name__)

    def setup_pipeline(
        self, pipeline_type: str, auto_fix: bool = True
    ) -> ValidationReport:
        """
        Set up all requirements for a pipeline type.

        Args:
            pipeline_type: Type of pipeline to set up
            auto_fix: Whether to automatically fix issues

        Returns:
            Final validation report
        """
        self.logger.info(f"Setting up pipeline: {pipeline_type}")

        # Get requirements
        requirements = get_pipeline_requirements(pipeline_type)

        # Initial validation
        initial_report = self.validator.validate_pipeline_requirements(requirements)

        if initial_report.overall_valid:
            self.logger.info(f"Pipeline {pipeline_type} already ready")
            return initial_report

        if not auto_fix:
            self.logger.info("Auto-fix disabled, returning validation report")
            return initial_report

        # Perform setup based on pipeline type
        # NEW: Use generic requirements-driven approach for basic pipelines
        if pipeline_type in ["basic", "basic_rerank"]:
            self.logger.info(
                f"Using generic requirements fulfillment for {pipeline_type}"
            )
            self._fulfill_requirements(requirements)
        elif pipeline_type == "crag":
            self._setup_crag_pipeline(requirements)

        else:
            self.logger.warning(f"No specific setup logic for {pipeline_type}")
            # Fallback: Try generic approach for unknown pipelines
            self.logger.info(
                f"Attempting generic requirements fulfillment for {pipeline_type}"
            )
            self._fulfill_requirements(requirements)

        # Check for optional chunking enhancement
        self._setup_optional_chunking(requirements)

        # Final validation
        final_report = self.validator.validate_pipeline_requirements(requirements)

        if final_report.overall_valid:
            self.logger.info(f"Pipeline {pipeline_type} setup completed successfully")
        else:
            self.logger.warning(f"Pipeline {pipeline_type} setup completed with issues")

        return final_report

    def _fulfill_requirements(self, requirements: PipelineRequirements):
        """
        Generic requirements fulfillment based on declared requirements.

        This method replaces hardcoded pipeline-specific setup with a generic
        approach driven by the requirements registry system.

        Args:
            requirements: Pipeline requirements to fulfill
        """
        # Count total requirements for progress tracking
        total_steps = (
            len(requirements.required_tables)
            + len(requirements.required_embeddings)
            + len(getattr(requirements, "optional_tables", []))
        )

        progress = SetupProgress(total_steps)

        # Fulfill table requirements
        for table_req in requirements.required_tables:
            progress.next_step(f"Setting up table: {table_req.name}")
            self._fulfill_table_requirement(table_req)

        # Fulfill embedding requirements
        for embedding_req in requirements.required_embeddings:
            progress.next_step(f"Setting up embeddings: {embedding_req.name}")
            self._fulfill_embedding_requirement(embedding_req)

        # Fulfill optional requirements
        for optional_req in getattr(requirements, "optional_tables", []):
            progress.next_step(f"Setting up optional: {optional_req.name}")
            self._fulfill_optional_requirement(optional_req)

        progress.complete()
        self.logger.info(
            f"Generic requirements fulfillment completed for {requirements.pipeline_name}"
        )

    def _fulfill_table_requirement(self, table_req):
        """Fulfill a table requirement."""
        # For now, tables are created by schema manager automatically
        # This is a placeholder for future table-specific setup logic
        self.logger.debug(f"Table requirement handled: {table_req.name}")

    def _fulfill_embedding_requirement(self, embedding_req):
        """Fulfill an embedding requirement generically."""
        if (
            embedding_req.table == "RAG.SourceDocuments"
            and embedding_req.column == "embedding"
        ):
            self._ensure_document_embeddings()
        elif (
            embedding_req.table == "RAG.DocumentTokenEmbeddings"
            and embedding_req.column == "token_embedding"
        ):
            self._ensure_token_embeddings()
        else:
            self.logger.warning(
                f"Unknown embedding requirement: {embedding_req.table}.{embedding_req.column}"
            )

    def _fulfill_optional_requirement(self, optional_req):
        """Fulfill an optional requirement."""
        if optional_req.name == "DocumentChunks":
            self._setup_optional_chunking_for_requirement(optional_req)
        else:
            self.logger.debug(f"Optional requirement noted: {optional_req.name}")

    def _setup_optional_chunking_for_requirement(self, chunk_req):
        """Set up chunking for a specific requirement."""
        try:
            self._generate_document_chunks()
            self.logger.info("Document chunks generated successfully")
        except Exception as e:
            self.logger.warning(f"Failed to generate document chunks: {e}")

    def _setup_basic_pipeline(self, requirements: PipelineRequirements):
        """Set up basic RAG pipeline requirements."""
        progress = SetupProgress(2)

        progress.next_step("Checking document embeddings")
        self._ensure_document_embeddings()

        progress.next_step("Validating setup")
        progress.complete()

    def _setup_optional_chunking(self, requirements: PipelineRequirements):
        """Set up optional chunking enhancement if requested."""
        # Check if chunking tables/embeddings are in optional requirements
        has_chunk_table = any(
            table.name == "DocumentChunks" for table in requirements.optional_tables
        )
        has_chunk_embeddings = any(
            emb.name == "chunk_embeddings" for emb in requirements.optional_embeddings
        )

        if has_chunk_table or has_chunk_embeddings:
            self.logger.info("Setting up optional chunking enhancement")
            progress = SetupProgress(4)

            progress.next_step("Creating chunks table")
            self._create_chunks_table()

            progress.next_step("Generating document chunks")
            self._generate_document_chunks()

            progress.next_step("Validating chunk embeddings")
            self._validate_embeddings_after_generation(
                "RAG.DocumentChunks", "chunk_embedding", "chunk"
            )

            progress.next_step("Chunking setup complete")
            progress.complete()

    def _setup_crag_pipeline(self, requirements: PipelineRequirements):
        """Set up CRAG pipeline requirements."""
        progress = SetupProgress(2)

        progress.next_step("Checking document embeddings")
        self._ensure_document_embeddings()

        progress.next_step("Validating setup")
        progress.complete()

    def _ensure_document_embeddings(self):
        """Ensure all documents have embeddings."""
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()

        try:

            # Check total docs
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
            print("Total documents in RAG.SourceDocuments:", cursor.fetchone()[0])

            # Check for documents without embeddings
            cursor.execute(
                """
                SELECT COUNT(*) FROM RAG.SourceDocuments
                WHERE embedding IS NULL
            """
            )

            missing_count = cursor.fetchone()[0]

            if missing_count > 0:
                self.logger.info(f"Generating embeddings for {missing_count} documents")
                self._generate_missing_document_embeddings()

                # Validate that embeddings were actually generated
                self._validate_embeddings_after_generation(
                    "RAG.SourceDocuments", "embedding", "document"
                )
            else:
                self.logger.info("All documents have embeddings")

        finally:
            cursor.close()

    def _generate_missing_document_embeddings(self):
        """Generate embeddings for documents that don't have them."""
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            # Get documents without embeddings
            cursor.execute(
                """
                SELECT doc_id, text_content as content
                FROM RAG.SourceDocuments
                WHERE embedding IS NULL
            """
            )

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

                            cursor.execute(
                                """
                                UPDATE RAG.SourceDocuments
                                SET embedding = TO_VECTOR(?)
                                WHERE doc_id = ?
                            """,
                                [vector_str, doc_id],
                            )
                            total_processed += 1

                        except Exception as doc_error:
                            self.logger.warning(
                                f"Failed to update embedding for {doc_id}: {doc_error}"
                            )
                            total_failed += 1

                    connection.commit()
                    self.logger.info(
                        f"Generated embeddings for batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}"
                    )

                except Exception as batch_error:
                    self.logger.error(
                        f"Failed to generate embeddings for batch {i//batch_size + 1}: {batch_error}"
                    )
                    total_failed += len(batch)
                    connection.rollback()

            # Final verification
            cursor.execute(
                """
                SELECT COUNT(*) FROM RAG.SourceDocuments
                WHERE embedding IS NULL
            """
            )
            remaining_missing = cursor.fetchone()[0]

            self.logger.info(
                f"Document embedding generation complete: {total_processed} processed, {total_failed} failed, {remaining_missing} still missing"
            )

            if remaining_missing > 0:
                self.logger.warning(
                    f"Still have {remaining_missing} documents without embeddings after generation"
                )

        finally:
            cursor.close()

    def heal_token_embeddings(
        self, target_doc_count: int = 1000, force_regenerate: bool = False
    ) -> Dict[str, Any]:
        """
        Heal missing token embeddings for specified document count.

        Args:
            target_doc_count: Number of documents to ensure have token embeddings
            force_regenerate: If True, regenerate all embeddings for target documents

        Returns:
            Dictionary with healing results and statistics
        """
        self.logger.info(
            f"Healing token embeddings for {target_doc_count} documents (force_regenerate={force_regenerate})"
        )

        start_time = time.time()
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            # Get target document set (deterministic ordering) - now returns List[Dict[str, str]]
            target_doc_ids_with_content = self._get_target_document_set(
                cursor, target_doc_count
            )

            if not target_doc_ids_with_content:
                self.logger.warning(
                    "No documents returned by _get_target_document_set."
                )
                return {
                    "status": "no_documents",
                    "processed": 0,
                    "failed": 0,
                    "skipped_doc_ids_bad_content": [],
                }

            # This map is used later to get content for docs that need healing
            target_doc_ids_with_content_map = {
                item["doc_id"]: item["content"] for item in target_doc_ids_with_content
            }
            target_doc_ids_list = list(target_doc_ids_with_content_map.keys())

            # Initialize tracking for skipped documents due to bad content
            skipped_due_to_missing_content_ids = []

            # Identify documents needing token embeddings
            if force_regenerate:
                # Delete existing embeddings for target documents
                self._delete_token_embeddings_for_documents(cursor, target_doc_ids_list)
                connection.commit()
                docs_needing_embeddings = target_doc_ids_list
                self.logger.info(
                    f"Force regenerate: processing all {len(docs_needing_embeddings)} target documents"
                )
            else:
                doc_ids_missing_embeddings = self._identify_missing_token_embeddings(
                    cursor, target_doc_ids_list
                )
                self.logger.info(
                    f"Found {len(doc_ids_missing_embeddings)} documents missing token embeddings"
                )

                docs_to_process_data = {}
                skipped_due_to_missing_content_ids = []  # Keep this for the return

                if doc_ids_missing_embeddings:
                    self.logger.info(
                        f"Diagnostic: Building docs_to_process_data from {len(doc_ids_missing_embeddings)} missing IDs."
                    )
                    for doc_id in doc_ids_missing_embeddings:
                        content = target_doc_ids_with_content_map.get(doc_id)
                        if content and content.strip():
                            docs_to_process_data[doc_id] = content
                        else:
                            skipped_due_to_missing_content_ids.append(doc_id)

                    if skipped_due_to_missing_content_ids:
                        self.logger.warning(
                            f"Diagnostic: Skipped {len(skipped_due_to_missing_content_ids)} docs due to bad content: {skipped_due_to_missing_content_ids[:5]}"
                        )

                # docs_needing_embeddings is what's actually used for generation call
                # This variable is critical.
                docs_needing_embeddings = list(docs_to_process_data.keys())
                self.logger.info(
                    f"Diagnostic: docs_to_process_data contains {len(docs_to_process_data)} items."
                )
                self.logger.info(
                    f"Diagnostic: docs_needing_embeddings contains {len(docs_needing_embeddings)} items."
                )

                # This is the critical check that seems to be using old logic/state
                if not docs_needing_embeddings:  # Check based on the derived list
                    self.logger.warning(
                        "Diagnostic: Path taken - 'if not docs_needing_embeddings' is TRUE."
                    )
                    self.logger.warning(
                        "No valid documents to process after filtering for content."
                    )  # New log message
                    return {
                        "status": "no_valid_content",  # More specific status
                        "processed": 0,
                        "failed": 0,
                        "skipped_doc_ids_bad_content": skipped_due_to_missing_content_ids,
                        "still_missing": len(
                            doc_ids_missing_embeddings
                        ),  # All initially missing are still missing if none have valid content
                        "duration": time.time() - start_time,
                    }
                else:
                    self.logger.info(
                        "Diagnostic: Path taken - 'if not docs_needing_embeddings' is FALSE."
                    )

            # Check if we have any documents to process after all filtering
            if not docs_needing_embeddings:
                self.logger.info("All target documents already have token embeddings")
                return {
                    "status": "complete",
                    "processed": 0,
                    "failed": 0,
                    "skipped_doc_ids_bad_content": skipped_due_to_missing_content_ids,
                    "duration": time.time() - start_time,
                }

            # Process documents in batches
            self.logger.info(
                f"Diagnostic: Proceeding to call _generate_token_embeddings_for_documents with {len(docs_needing_embeddings)} documents."
            )
            batch_size = 16
            results = self._generate_token_embeddings_for_documents(
                connection, cursor, docs_needing_embeddings, batch_size
            )

            # Final verification
            final_missing = self._identify_missing_token_embeddings(
                cursor, target_doc_ids_list
            )

            duration = time.time() - start_time
            self.logger.info(
                f"Token embedding healing complete: {results['processed']} processed, "
                f"{results['failed']} failed, {len(final_missing)} still missing, "
                f"duration: {duration:.1f}s"
            )

            return {
                "status": "complete" if len(final_missing) == 0 else "partial",
                "target_doc_count": target_doc_count,
                "processed": results["processed"],
                "failed": results["failed"],
                "still_missing": len(final_missing),
                "skipped_doc_ids_bad_content": skipped_due_to_missing_content_ids,
                "duration": duration,
            }

        except Exception as e:
            self.logger.error(
                f"Error during token embedding healing: {e}", exc_info=True
            )
            connection.rollback()
            return {
                "status": "error",
                "error": str(e),
                "processed": 0,
                "failed": 0,
                "skipped_doc_ids_bad_content": [],
            }
        finally:
            cursor.close()

    def _get_target_document_set(self, cursor, doc_count: int) -> List[Dict[str, str]]:
        """
        Get the target set of documents with doc_id and content for the specified count.

        Handles CLOB content that might be returned as com.intersystems.jdbc.IRISInputStream
        objects by reading and decoding them into Python strings.
        """
        # Temporarily modify the SQL query for diagnostics: fetch all documents without TOP to see if any data is retrieved
        # query = f"""
        #     SELECT TOP ? doc_id, text_content
        #     FROM RAG.SourceDocuments
        #     ORDER BY doc_id
        # """
        # cursor.execute(query, (doc_count,))

        query = """
            SELECT doc_id, text_content
            FROM RAG.SourceDocuments
            ORDER BY doc_id
        """
        cursor.execute(query)  # No parameters needed now
        all_fetched_data = cursor.fetchall()
        self.logger.info(
            f"_get_target_document_set: Fetched {len(all_fetched_data)} total documents initially (before Python slicing)."
        )

        # Count and report documents with NULL or empty text_content
        null_or_empty_content_count = 0
        for doc_id, content in all_fetched_data:
            if not content:  # Checks for None or empty string
                null_or_empty_content_count += 1
        if null_or_empty_content_count > 0:
            self.logger.warning(
                f"_get_target_document_set: Found {null_or_empty_content_count} documents (out of {len(all_fetched_data)}) with NULL or empty text_content."
            )

        # Apply slicing in Python
        target_docs_data_tuples = all_fetched_data[:doc_count]

        # Process content to handle potential stream objects
        processed_docs = []
        for doc_id, content_val in target_docs_data_tuples:
            actual_content = None
            if hasattr(content_val, "read"):  # Check if it's a stream-like object
                try:
                    # Read the entire stream and decode
                    stream_data = content_val.read()
                    if isinstance(stream_data, bytes):
                        actual_content = stream_data.decode("utf-8")
                    elif isinstance(
                        stream_data, str
                    ):  # Should ideally be bytes from stream
                        actual_content = stream_data
                    else:  # Fallback if read() returns something unexpected
                        actual_content = (
                            str(stream_data) if stream_data is not None else None
                        )

                    if hasattr(content_val, "close"):  # Close the stream if possible
                        content_val.close()
                except Exception as e:
                    self.logger.error(
                        f"Error reading stream content for doc_id {doc_id}: {e}"
                    )
                    actual_content = None  # Or handle as appropriate
            elif isinstance(content_val, str):
                actual_content = content_val
            elif content_val is None:
                actual_content = None
            else:  # If it's some other type, try to convert to string
                self.logger.warning(
                    f"Unexpected content type for doc_id {doc_id}: {type(content_val)}. Attempting str()."
                )
                actual_content = str(content_val)

            processed_docs.append({"doc_id": doc_id, "content": actual_content})

        self.logger.info(
            f"_get_target_document_set: Processed and sliced to {len(processed_docs)} potential target documents."
        )
        if processed_docs:
            self.logger.debug(
                f"_get_target_document_set: Sample target doc_ids: {[d['doc_id'] for d in processed_docs[:3]]}"
            )

        return processed_docs

    def _identify_missing_token_embeddings(
        self, cursor, target_doc_ids: List[str]
    ) -> List[str]:
        """Identify documents missing token embeddings within target set."""
        self.logger.info(
            f"_identify_missing_token_embeddings: Received {len(target_doc_ids)} target_doc_ids to check."
        )
        if not target_doc_ids:
            return []

        # Create a temporary table or use IN clause for efficiency
        doc_ids_str = "', '".join(target_doc_ids)
        query = f"""
            SELECT doc_id
            FROM (
                SELECT DISTINCT '{doc_ids_str.split("', '")[0]}' as doc_id
                {' UNION ALL SELECT DISTINCT ' + "' UNION ALL SELECT DISTINCT '".join(f"'{doc_id}'" for doc_id in target_doc_ids[1:]) if len(target_doc_ids) > 1 else ''}
            ) target_docs
            WHERE doc_id NOT IN (
                SELECT DISTINCT doc_id
                FROM RAG.DocumentTokenEmbeddings
                WHERE doc_id IN ('{doc_ids_str}')
            )
        """

        # Simpler approach for IRIS SQL
        missing_docs = []
        docs_with_existing_embeddings = []
        for doc_id in target_doc_ids:
            cursor.execute(
                """
                SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings WHERE doc_id = ?
            """,
                [doc_id],
            )
            count = cursor.fetchone()[0]
            if count == 0:
                missing_docs.append(doc_id)
            else:
                docs_with_existing_embeddings.append(doc_id)

        self.logger.info(
            f"_identify_missing_token_embeddings: Found {len(docs_with_existing_embeddings)} documents from target set that already have some token embeddings."
        )

        return missing_docs

    def _delete_token_embeddings_for_documents(self, cursor, doc_ids: List[str]):
        """Delete existing token embeddings for specified documents."""
        if not doc_ids:
            return

        for doc_id in doc_ids:
            cursor.execute(
                "DELETE FROM RAG.DocumentTokenEmbeddings WHERE doc_id = ?", [doc_id]
            )

        self.logger.info(
            f"Deleted existing token embeddings for {len(doc_ids)} documents"
        )

    def _generate_token_embeddings_for_documents(
        self, connection, cursor, doc_ids: List[str], batch_size: int
    ) -> Dict[str, int]:
        """Generate token embeddings for specified documents."""
        # At the beginning of the method, log the number of doc_ids received and a sample of them
        self.logger.info(
            f"_generate_token_embeddings_for_documents: Received {len(doc_ids)} doc_ids to process. Sample: {doc_ids[:3]}"
        )

        if not doc_ids:
            return {"processed": 0, "failed": 0}

        # Get configuration parameters
        max_tokens = 512

        # Fetch document content for specified doc_ids
        doc_content_map = {}
        for doc_id in doc_ids:
            cursor.execute(
                "SELECT abstract FROM RAG.SourceDocuments WHERE doc_id = ? AND abstract IS NOT NULL",
                [doc_id],
            )
            result = cursor.fetchone()
            if result and result[0]:
                doc_content_map[doc_id] = result[0]

        if not doc_content_map:
            self.logger.warning(
                "No document content found for token embedding generation"
            )
            return {"processed": 0, "failed": 0}

        total_processed = 0
        total_failed = 0

        # Process documents in batches
        doc_items = list(doc_content_map.items())
        for i in range(0, len(doc_items), batch_size):
            batch = doc_items[i : i + batch_size]

            # Inside the loop, log the current batch of doc_ids
            current_batch_doc_ids = [doc_id for doc_id, _ in batch]
            self.logger.info(
                f"Processing batch {i//batch_size + 1} with doc_ids: {current_batch_doc_ids}"
            )

            batch_texts_to_embed = [content for _, content in batch]
            self.logger.debug(f"Batch texts to embed: {batch_texts_to_embed}")

            for doc_id, content in batch:
                self.logger.debug(
                    f"Processing doc_id: {doc_id} for token embedding. Content length: {len(content) if content else 0}"
                )
                if not content:
                    self.logger.warning(
                        f"Skipping doc_id: {doc_id} due to empty or None content."
                    )
                    total_failed += 1
                    continue
                try:
                    result = self._process_document_tokens(
                        cursor, doc_id, content, max_tokens
                    )
                    if result["success"]:
                        total_processed += 1
                    else:
                        total_failed += 1

                    # Commit after each document to ensure progress is saved
                    connection.commit()

                except Exception as e:
                    self.logger.warning(
                        f"Failed to generate token embeddings for {doc_id}: {e}"
                    )
                    total_failed += 1
                    connection.rollback()

            # Progress logging
            if (i // batch_size + 1) % 5 == 0:
                self.logger.info(
                    f"Processed batch {i//batch_size + 1}/{(len(doc_items) + batch_size - 1)//batch_size}, "
                    f"{total_processed} documents completed"
                )

        return {"processed": total_processed, "failed": total_failed}

    def _process_document_tokens(
        self, cursor, doc_id: str, content: str, max_tokens: int
    ) -> Dict[str, Any]:
        """Process tokens for a single document."""
        # At the beginning, log the doc_id and repr(content) received
        self.logger.info(
            f"_process_document_tokens: Processing doc_id: {doc_id}, Content repr: {repr(content)}"
        )

        try:
            # Simple tokenization approach for demonstration
            tokens = content.split()[:max_tokens]  # Limit tokens

            # Log the tokens data before iterating to insert them
            tokens_data_for_doc = [(i, token) for i, token in enumerate(tokens)]

            # Add logging for number of tokens generated, especially for documents with content '-1'
            self.logger.info(
                f"For doc_id: {doc_id} with content repr: {repr(content)}, generated {len(tokens_data_for_doc)} tokens/embeddings."
            )

            if (
                not tokens_data_for_doc and content and content.strip()
            ):  # If content was valid but no tokens generated
                self.logger.warning(
                    f"Doc_id: {doc_id} had valid content '{content}' but resulted in 0 tokens."
                )

            if not tokens:
                return {"success": False, "reason": "no_tokens"}

            tokens_processed = 0
            tokens_failed = 0

            self.logger.info(f"Tokens data for doc_id {doc_id}: {tokens_data_for_doc}")

            # Log the number of tokens being inserted for a given doc_id
            self.logger.info(
                f"Attempting to insert {len(tokens_data_for_doc)} tokens for doc_id {doc_id}"
            )

            # Generate embeddings for each token using the embedding manager
            for token_index, token_text in enumerate(tokens):
                try:
                    # Generate embedding for individual token
                    token_embedding = self.embedding_manager.embed_text(token_text)

                    # Ensure token_embedding is a list of floats
                    if hasattr(token_embedding, "tolist"):
                        token_embedding = token_embedding.tolist()
                    elif not isinstance(token_embedding, list):
                        if hasattr(token_embedding, "__iter__"):
                            token_embedding = list(token_embedding)
                        else:
                            # If it's a single float, skip this token
                            tokens_failed += 1
                            continue

                    # Use consistent vector format that validator expects
                    vector_str = f"[{','.join(map(str, token_embedding))}]"

                    cursor.execute(
                        """
                        INSERT INTO RAG.DocumentTokenEmbeddings
                        (doc_id, token_index, token_text, token_embedding)
                        VALUES (?, ?, ?, TO_VECTOR(?))
                    """,
                        [doc_id, token_index, token_text, vector_str],
                    )

                    tokens_processed += 1

                except Exception as token_error:
                    self.logger.debug(
                        f"Failed to generate embedding for token '{token_text}': {token_error}"
                    )
                    tokens_failed += 1
                    continue

            return {
                "success": tokens_processed > 0,
                "tokens_processed": tokens_processed,
                "tokens_failed": tokens_failed,
            }

        except Exception as e:
            self.logger.warning(f"Error processing tokens for document {doc_id}: {e}")
            return {"success": False, "reason": str(e)}

    def _create_chunks_table(self):
        """Create table for document chunks if it doesn't exist."""
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            # Check if table exists
            cursor.execute(
                """
                SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = 'DocumentChunks'
            """
            )

            if cursor.fetchone()[0] == 0:
                # Create table
                cursor.execute(
                    """
                    CREATE TABLE RAG.DocumentChunks (
                        id VARCHAR(255) PRIMARY KEY,
                        chunk_id VARCHAR(255),
                        doc_id VARCHAR(255),
                        chunk_text TEXT,
                        chunk_embedding VECTOR(FLOAT, 384),
                        chunk_index INTEGER,
                        chunk_type VARCHAR(100),
                        metadata TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (doc_id) REFERENCES RAG.SourceDocuments(id)
                    )
                """
                )
                connection.commit()
                self.logger.info("Created DocumentChunks table")
            else:
                self.logger.info("DocumentChunks table already exists")

        finally:
            cursor.close()

    def _generate_document_chunks(self):
        """Generate document chunks with embeddings."""
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            # Check if chunks already exist
            cursor.execute("SELECT COUNT(*) FROM RAG.DocumentChunks")
            existing_count = cursor.fetchone()[0]

            if existing_count > 0:
                self.logger.info(
                    f"Document chunks already exist ({existing_count} chunks)"
                )
                return

            # Get documents for chunking
            cursor.execute(
                "SELECT doc_id, text_content as content FROM RAG.SourceDocuments"
            )
            documents = cursor.fetchall()

            if not documents:
                self.logger.warning("No documents found for chunk generation")
                return

            chunk_size = self.config_manager.get("chunking:chunk_size", 1000)
            chunk_overlap = self.config_manager.get("chunking:chunk_overlap", 200)

            total_chunks_processed = 0
            total_chunks_failed = 0
            documents_processed = 0

            for doc_id, content in documents:
                if not content:
                    continue

                try:
                    # Simple chunking logic
                    chunks = self._split_text_into_chunks(
                        content, chunk_size, chunk_overlap
                    )

                    if not chunks:
                        continue

                    # Generate embeddings for chunks
                    try:
                        chunk_embeddings = self.embedding_manager.embed_texts(chunks)

                        # Store chunks with consistent vector format
                        for chunk_index, (chunk_text, embedding) in enumerate(
                            zip(chunks, chunk_embeddings)
                        ):
                            try:
                                # Ensure embedding is a list
                                if hasattr(embedding, "tolist"):
                                    embedding = embedding.tolist()
                                elif not isinstance(embedding, list):
                                    embedding = list(embedding)

                                # Use consistent vector format that validator expects
                                vector_str = f"[{','.join(map(str, embedding))}]"

                                chunk_id = f"{doc_id}_chunk_{chunk_index}"
                                cursor.execute(
                                    """
                                    INSERT INTO RAG.DocumentChunks
                                    (id, chunk_id, doc_id, chunk_index, chunk_text, chunk_embedding, metadata)
                                    VALUES (?, ?, ?, ?, ?, TO_VECTOR(?), ?)
                                """,
                                    [
                                        chunk_id,
                                        chunk_id,
                                        doc_id,
                                        chunk_index,
                                        chunk_text,
                                        vector_str,
                                        f'{{"chunk_size": {len(chunk_text)}, "parent_doc": "{doc_id}"}}',
                                    ],
                                )
                                total_chunks_processed += 1

                            except Exception as chunk_error:
                                self.logger.warning(
                                    f"Failed to store chunk {chunk_index} for {doc_id}: {chunk_error}"
                                )
                                total_chunks_failed += 1

                        connection.commit()
                        documents_processed += 1

                        if documents_processed % 50 == 0:
                            self.logger.info(
                                f"Processed {documents_processed}/{len(documents)} documents, {total_chunks_processed} chunks generated"
                            )

                    except Exception as embedding_error:
                        self.logger.warning(
                            f"Failed to generate embeddings for chunks in {doc_id}: {embedding_error}"
                        )
                        total_chunks_failed += len(chunks)
                        connection.rollback()

                except Exception as doc_error:
                    self.logger.warning(
                        f"Failed to process document {doc_id}: {doc_error}"
                    )
                    continue

            # Final verification
            cursor.execute("SELECT COUNT(*) FROM RAG.DocumentChunks")
            final_count = cursor.fetchone()[0]

            self.logger.info(
                f"Chunk generation complete: {final_count} total chunks, {documents_processed} documents processed, {total_chunks_failed} chunks failed"
            )

            if final_count == 0:
                self.logger.error("No chunks were generated successfully")
            elif (
                total_chunks_failed > total_chunks_processed * 0.1
            ):  # More than 10% failure rate
                self.logger.warning(
                    f"High failure rate in chunk generation: {total_chunks_failed}/{total_chunks_processed + total_chunks_failed}"
                )

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
        Validate that embeddings were successfully generated and are in the correct format.

        Args:
            table: Table name to validate
            column: Column name containing embeddings
            embedding_type: Type of embedding for logging (e.g., "document", "token", "chunk")
        """
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            # Check total count and count with embeddings
            cursor.execute(
                f"""
                SELECT
                    COUNT(*) as total_rows,
                    COUNT({column}) as rows_with_embeddings
                FROM {table}
            """
            )

            result = cursor.fetchone()
            total_rows = result[0]
            rows_with_embeddings = result[1]

            if total_rows == 0:
                self.logger.warning(
                    f"No data found in {table} for {embedding_type} embedding validation"
                )
                return

            if rows_with_embeddings == 0:
                self.logger.error(
                    f"No {embedding_type} embeddings were generated in {table}.{column}"
                )
                return

            # Check embedding format
            cursor.execute(
                f"""
                SELECT TOP 1 {column}
                FROM {table}
                WHERE {column} IS NOT NULL
            """
            )

            sample_result = cursor.fetchone()
            if sample_result and sample_result[0]:
                embedding_str = str(sample_result[0])
                # IRIS VECTOR columns return comma-separated values without brackets when retrieved
                # This is the correct format for VECTOR data in IRIS
                if "," in embedding_str and not embedding_str.startswith("["):
                    # This is the expected VECTOR format from IRIS
                    self.logger.info(
                        f"{embedding_type.capitalize()} embeddings validation passed: {rows_with_embeddings}/{total_rows} embeddings in correct VECTOR format"
                    )
                elif embedding_str.startswith("[") and embedding_str.endswith("]"):
                    # This might be string format, which should be converted to VECTOR
                    self.logger.warning(
                        f"Found string-formatted {embedding_type} embedding in {table}.{column}, should be VECTOR type: {embedding_str[:100]}"
                    )
                else:
                    self.logger.warning(
                        f"Invalid {embedding_type} embedding format detected in {table}.{column}: {embedding_str[:100]}"
                    )

            # Calculate completeness
            completeness_ratio = rows_with_embeddings / total_rows
            if completeness_ratio < 0.95:
                self.logger.warning(
                    f"{embedding_type.capitalize()} embeddings incomplete: {rows_with_embeddings}/{total_rows} ({completeness_ratio:.1%})"
                )
            else:
                self.logger.info(
                    f"{embedding_type.capitalize()} embeddings complete: {rows_with_embeddings}/{total_rows} ({completeness_ratio:.1%})"
                )

        except Exception as e:
            self.logger.error(
                f"Error validating {embedding_type} embeddings in {table}.{column}: {e}"
            )
        finally:
            cursor.close()
