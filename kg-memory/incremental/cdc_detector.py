#!/usr/bin/env python3
"""
Change Detection Component (CDC Detector) for LightRAG-inspired incremental indexing.

This module implements efficient document fingerprinting and change detection
for knowledge graph incremental updates without full rebuilds.
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional

from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.connection import ConnectionManager
from iris_rag.core.models import Document

logger = logging.getLogger(__name__)


@dataclass
class DocumentChangeInfo:
    """Information about a document change."""

    document_id: str
    change_type: str  # 'new', 'modified', 'deleted'
    current_fingerprint: Optional[str] = None
    previous_fingerprint: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ChangeBatch:
    """Batch of document changes detected during processing."""

    new_documents: List[DocumentChangeInfo] = field(default_factory=list)
    modified_documents: List[DocumentChangeInfo] = field(default_factory=list)
    deleted_documents: List[DocumentChangeInfo] = field(default_factory=list)
    batch_id: str = field(default_factory=lambda: str(int(time.time() * 1000)))
    processing_time_ms: float = 0.0
    total_documents_processed: int = 0

    @property
    def total_changes(self) -> int:
        """Total number of changes in this batch."""
        return (
            len(self.new_documents)
            + len(self.modified_documents)
            + len(self.deleted_documents)
        )

    @property
    def has_changes(self) -> bool:
        """Whether this batch contains any changes."""
        return self.total_changes > 0


class CDCDetector:
    """
    Change Detection Component for efficient document fingerprinting and change detection.

    Features:
    - SHA-256 content + metadata hashing for document fingerprinting
    - Incremental change detection (new, modified, deleted documents)
    - Batch processing optimization for large document sets (1K+ documents)
    - Integration with existing RAG.SourceDocuments table patterns
    - Performance target: <1ms per document fingerprinting
    """

    def __init__(
        self,
        connection_manager: ConnectionManager,
        config_manager: ConfigurationManager,
    ):
        """
        Initialize the CDC detector.

        Args:
            connection_manager: Database connection manager
            config_manager: Configuration manager for settings
        """
        self.connection_manager = connection_manager
        self.config_manager = config_manager

        # Load configuration
        self.cdc_config = self.config_manager.get("incremental_indexing:cdc", {})
        self.default_batch_size = self.cdc_config.get("default_batch_size", 1000)
        self.fingerprint_algorithm = self.cdc_config.get(
            "fingerprint_algorithm", "sha256"
        )
        self.include_metadata_in_fingerprint = self.cdc_config.get(
            "include_metadata_in_fingerprint", True
        )
        self.performance_tracking = self.cdc_config.get("performance_tracking", True)

        # Performance metrics
        self._fingerprint_times: List[float] = []
        self._batch_processing_times: List[float] = []

        # Ensure fingerprint table exists
        self._ensure_fingerprint_table()

        logger.info(
            f"CDC Detector initialized with batch_size={self.default_batch_size}, "
            f"algorithm={self.fingerprint_algorithm}"
        )

    def generate_fingerprint(self, document: Document) -> str:
        """
        Generate SHA-256 fingerprint for a document based on content + metadata.

        Performance target: <1ms per document

        Args:
            document: Document to fingerprint

        Returns:
            SHA-256 fingerprint string
        """
        start_time = time.perf_counter()

        try:
            # Create hash object
            hasher = hashlib.sha256()

            # Add document content
            hasher.update(document.page_content.encode("utf-8"))

            # Add document ID for uniqueness
            hasher.update(document.id.encode("utf-8"))

            # Add metadata if configured
            if self.include_metadata_in_fingerprint and document.metadata:
                # Sort metadata for consistent hashing
                metadata_str = str(sorted(document.metadata.items()))
                hasher.update(metadata_str.encode("utf-8"))

            fingerprint = hasher.hexdigest()

            # Track performance
            if self.performance_tracking:
                processing_time = (time.perf_counter() - start_time) * 1000  # ms
                self._fingerprint_times.append(processing_time)

                # Log performance warning if target exceeded
                if processing_time > 1.0:  # 1ms target
                    logger.warning(
                        f"Fingerprint generation took {processing_time:.2f}ms (target: <1ms)"
                    )

            return fingerprint

        except Exception as e:
            logger.error(
                f"Error generating fingerprint for document {document.id}: {e}"
            )
            raise

    def detect_changes(self, documents: List[Document]) -> ChangeBatch:
        """
        Detect changes in a list of documents compared to stored fingerprints.

        Args:
            documents: List of documents to check for changes

        Returns:
            ChangeBatch containing detected changes
        """
        start_time = time.perf_counter()

        try:
            # Get existing fingerprints from database
            existing_fingerprints = self._get_existing_fingerprints(
                [doc.id for doc in documents]
            )

            # Track document IDs we've seen
            processed_doc_ids = set()

            # Initialize change batch
            change_batch = ChangeBatch()

            # Process each document
            for document in documents:
                processed_doc_ids.add(document.id)

                # Generate current fingerprint
                current_fingerprint = self.generate_fingerprint(document)

                # Check for changes
                if document.id not in existing_fingerprints:
                    # New document
                    change_info = DocumentChangeInfo(
                        document_id=document.id,
                        change_type="new",
                        current_fingerprint=current_fingerprint,
                        metadata={
                            "document_title": document.metadata.get("title", "Untitled")
                        },
                    )
                    change_batch.new_documents.append(change_info)

                elif existing_fingerprints[document.id] != current_fingerprint:
                    # Modified document
                    change_info = DocumentChangeInfo(
                        document_id=document.id,
                        change_type="modified",
                        current_fingerprint=current_fingerprint,
                        previous_fingerprint=existing_fingerprints[document.id],
                        metadata={
                            "document_title": document.metadata.get("title", "Untitled")
                        },
                    )
                    change_batch.modified_documents.append(change_info)

            # Check for deleted documents (exist in fingerprints but not in current documents)
            for doc_id, fingerprint in existing_fingerprints.items():
                if doc_id not in processed_doc_ids:
                    change_info = DocumentChangeInfo(
                        document_id=doc_id,
                        change_type="deleted",
                        previous_fingerprint=fingerprint,
                    )
                    change_batch.deleted_documents.append(change_info)

            # Set batch metadata
            change_batch.total_documents_processed = len(documents)
            change_batch.processing_time_ms = (time.perf_counter() - start_time) * 1000

            # Track performance
            if self.performance_tracking:
                self._batch_processing_times.append(change_batch.processing_time_ms)

            logger.info(
                f"Change detection completed: {change_batch.total_changes} changes detected "
                f"from {len(documents)} documents in {change_batch.processing_time_ms:.2f}ms"
            )

            return change_batch

        except Exception as e:
            logger.error(f"Error detecting changes: {e}")
            raise

    def batch_process(
        self, document_source: Iterator[Document], batch_size: int = None
    ) -> List[ChangeBatch]:
        """
        Process documents in batches for efficient change detection.

        Performance target: <10s for 10K documents

        Args:
            document_source: Iterator of documents to process
            batch_size: Size of each batch (defaults to configured value)

        Returns:
            List of ChangeBatch objects for each processed batch
        """
        if batch_size is None:
            batch_size = self.default_batch_size

        start_time = time.perf_counter()
        change_batches = []
        total_documents = 0

        try:
            current_batch = []

            for document in document_source:
                current_batch.append(document)

                # Process batch when it reaches target size
                if len(current_batch) >= batch_size:
                    batch_result = self.detect_changes(current_batch)
                    change_batches.append(batch_result)
                    total_documents += len(current_batch)
                    current_batch = []

                    logger.debug(
                        f"Processed batch {len(change_batches)} with {batch_result.total_changes} changes"
                    )

            # Process remaining documents
            if current_batch:
                batch_result = self.detect_changes(current_batch)
                change_batches.append(batch_result)
                total_documents += len(current_batch)

            processing_time = (time.perf_counter() - start_time) * 1000

            # Performance validation
            if (
                total_documents >= 10000 and processing_time > 10000
            ):  # 10s target for 10K docs
                logger.warning(
                    f"Batch processing took {processing_time/1000:.2f}s for {total_documents} documents "
                    f"(target: <10s for 10K documents)"
                )

            logger.info(
                f"Batch processing completed: {len(change_batches)} batches, "
                f"{total_documents} documents, {processing_time:.2f}ms total"
            )

            return change_batches

        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            raise

    def update_fingerprints(self, change_batch: ChangeBatch) -> bool:
        """
        Update stored fingerprints based on detected changes.

        Args:
            change_batch: Batch of changes to apply

        Returns:
            True if update successful, False otherwise
        """
        if not change_batch.has_changes:
            logger.debug("No changes to update")
            return True

        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            # Update fingerprints for new documents
            for change_info in change_batch.new_documents:
                cursor.execute(
                    """
                    INSERT INTO RAG.DocumentFingerprints 
                    (document_id, fingerprint, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                """,
                    [
                        change_info.document_id,
                        change_info.current_fingerprint,
                        change_info.detected_at,
                        change_info.detected_at,
                    ],
                )

            # Update fingerprints for modified documents
            for change_info in change_batch.modified_documents:
                cursor.execute(
                    """
                    UPDATE RAG.DocumentFingerprints 
                    SET fingerprint = ?, updated_at = ?
                    WHERE document_id = ?
                """,
                    [
                        change_info.current_fingerprint,
                        change_info.detected_at,
                        change_info.document_id,
                    ],
                )

            # Remove fingerprints for deleted documents
            for change_info in change_batch.deleted_documents:
                cursor.execute(
                    """
                    DELETE FROM RAG.DocumentFingerprints 
                    WHERE document_id = ?
                """,
                    [change_info.document_id],
                )

            connection.commit()

            logger.info(
                f"Updated fingerprints: {len(change_batch.new_documents)} new, "
                f"{len(change_batch.modified_documents)} modified, "
                f"{len(change_batch.deleted_documents)} deleted"
            )

            return True

        except Exception as e:
            logger.error(f"Error updating fingerprints: {e}")
            connection.rollback()
            return False
        finally:
            cursor.close()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for fingerprinting and batch processing."""
        if not self._fingerprint_times:
            return {"status": "no_metrics_available"}

        return {
            "fingerprint_performance": {
                "avg_time_ms": sum(self._fingerprint_times)
                / len(self._fingerprint_times),
                "max_time_ms": max(self._fingerprint_times),
                "min_time_ms": min(self._fingerprint_times),
                "total_fingerprints": len(self._fingerprint_times),
                "target_met_percentage": len(
                    [t for t in self._fingerprint_times if t <= 1.0]
                )
                / len(self._fingerprint_times)
                * 100,
            },
            "batch_performance": {
                "avg_batch_time_ms": (
                    sum(self._batch_processing_times)
                    / len(self._batch_processing_times)
                    if self._batch_processing_times
                    else 0
                ),
                "total_batches": len(self._batch_processing_times),
            },
        }

    def _get_existing_fingerprints(self, document_ids: List[str]) -> Dict[str, str]:
        """Get existing fingerprints for a list of document IDs."""
        if not document_ids:
            return {}

        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            # Create placeholders for IN clause
            placeholders = ",".join(["?" for _ in document_ids])

            cursor.execute(
                f"""
                SELECT document_id, fingerprint 
                FROM RAG.DocumentFingerprints 
                WHERE document_id IN ({placeholders})
            """,
                document_ids,
            )

            results = cursor.fetchall()
            return {str(doc_id): str(fingerprint) for doc_id, fingerprint in results}

        except Exception as e:
            logger.error(f"Error getting existing fingerprints: {e}")
            return {}
        finally:
            cursor.close()

    def _ensure_fingerprint_table(self):
        """Ensure the DocumentFingerprints table exists."""
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            # Try to create the table - this will be handled by schema extensions later
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS RAG.DocumentFingerprints (
                    document_id VARCHAR(255) PRIMARY KEY,
                    fingerprint VARCHAR(64) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            connection.commit()
            logger.debug("DocumentFingerprints table ensured")

        except Exception as e:
            logger.warning(f"Could not ensure DocumentFingerprints table: {e}")
            # This will be handled properly by schema extensions
        finally:
            cursor.close()
