#!/usr/bin/env python3
"""
IRIS Stream Reader Utility
Handles reading IRIS stream objects properly from JDBC connections.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def read_iris_stream(stream_obj: Any) -> str:
    """
    Read content from an IRIS stream object returned by JDBC.

    Args:
        stream_obj: Stream object from IRIS JDBC connection

    Returns:
        String content of the stream
    """
    if stream_obj is None:
        return ""

    try:
        # Check if it's already a string
        if isinstance(stream_obj, str):
            return stream_obj

        # Handle different stream types from DBAPI vs JDBC

        # JDBC streams (Java objects) - use readAllBytes()
        if (
            hasattr(stream_obj, "readAllBytes")
            and str(type(stream_obj)).find("java") != -1
        ):
            # Use readAllBytes for IRIS JDBC streams - this gets the actual content
            try:
                content_bytes = stream_obj.readAllBytes()
                if hasattr(content_bytes, "__iter__"):
                    return bytes(content_bytes).decode("utf-8", errors="ignore")
                else:
                    return str(content_bytes)
            except Exception as e:
                logger.warning(f"Failed to read JDBC stream with readAllBytes: {e}")

        # DBAPI streams (Python objects) - use read()
        elif hasattr(stream_obj, "read") and str(type(stream_obj)).find("java") == -1:
            # For DBAPI streams, read() should work normally
            try:
                content_bytes = stream_obj.read()
                if isinstance(content_bytes, bytes):
                    return content_bytes.decode("utf-8", errors="ignore")
                elif isinstance(content_bytes, str):
                    return content_bytes
                else:
                    return str(content_bytes)
            except Exception as e:
                logger.warning(f"Failed to read DBAPI stream: {e}")

        # JDBC streams fallback - read() often returns length not content for JDBC
        elif hasattr(stream_obj, "read"):
            try:
                content_bytes = stream_obj.read()
                if isinstance(content_bytes, bytes):
                    return content_bytes.decode("utf-8", errors="ignore")
                elif isinstance(content_bytes, str):
                    return content_bytes
                else:
                    # For JDBC, this is likely a length, not content
                    if str(type(stream_obj)).find("java") != -1:
                        logger.debug(
                            f"JDBC stream.read() returned length ({content_bytes}), not content"
                        )
                        return ""
                    else:
                        return str(content_bytes)
            except Exception as e:
                logger.warning(f"Failed to read stream as fallback: {e}")

        # Handle Java String objects
        if hasattr(stream_obj, "toString"):
            return str(stream_obj.toString())

        # Handle numeric stream lengths (IRIS sometimes returns length instead of content)
        if hasattr(stream_obj, "__int__") or str(type(stream_obj)).find("JInt") != -1:
            logger.warning(
                f"Stream object appears to be length ({stream_obj}), not content"
            )
            return ""

        # Last resort - convert to string
        return str(stream_obj)

    except Exception as e:
        logger.error(f"Failed to read IRIS stream: {e}")
        return ""


def get_document_content_properly(cursor, doc_id: str) -> tuple[str, str, str]:
    """
    Get document content with proper stream handling.

    Returns:
        Tuple of (doc_id, title, content)
    """
    try:
        cursor.execute(
            """
            SELECT doc_id, title, text_content 
            FROM RAG.SourceDocuments 
            WHERE doc_id = ?
        """,
            (doc_id,),
        )

        result = cursor.fetchone()
        if not result:
            return doc_id, "", ""

        fetched_doc_id, title_stream, content_stream = result

        # Read streams properly
        title = read_iris_stream(title_stream)
        content = read_iris_stream(content_stream)

        return fetched_doc_id, title, content

    except Exception as e:
        logger.error(f"Failed to get document content for {doc_id}: {e}")
        return doc_id, "", ""


def test_stream_reading():
    """Test stream reading functionality."""
    from iris_connector import get_iris_connection

    conn = get_iris_connection()
    cursor = conn.cursor()

    try:
        # Get a sample document
        cursor.execute("SELECT TOP 1 doc_id FROM RAG.SourceDocuments")
        result = cursor.fetchone()

        if result:
            doc_id = result[0]
            doc_id, title, content = get_document_content_properly(cursor, doc_id)

            print(f"Document ID: {doc_id}")
            print(f"Title: {title[:100]}...")
            print(f"Content: {content[:200]}...")
            print(f"Content length: {len(content)}")

            return len(content) > 0
        else:
            print("No documents found")
            return False

    finally:
        cursor.close()


if __name__ == "__main__":
    test_stream_reading()
