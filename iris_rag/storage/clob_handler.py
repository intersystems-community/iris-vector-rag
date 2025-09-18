"""
CLOB (Character Large Object) handling utilities for IRIS database operations.

This module provides utilities for converting CLOB objects and streams
to Python strings, handling various edge cases and encoding issues.
"""

import logging
from typing import Any, Dict
from ..core.vector_store_exceptions import VectorStoreCLOBError

logger = logging.getLogger(__name__)


def convert_clob_to_string(value: Any) -> str:
    """
    Convert CLOB/IRISInputStream objects to strings.

    This function handles various types of CLOB-like objects that might
    be returned by IRIS database drivers, including stream objects and
    byte arrays.

    Args:
        value: The value to convert, potentially an IRISInputStream, bytes, or string

    Returns:
        String representation of the value

    Raises:
        VectorStoreCLOBError: If CLOB conversion fails critically
    """
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    # Handle stream-like objects (IRISInputStream, etc.)
    if hasattr(value, "read") and callable(getattr(value, "read")):
        try:
            stream_content = value.read()
            if isinstance(stream_content, bytes):
                return stream_content.decode("utf-8", errors="replace")
            else:
                return str(stream_content)
        except Exception as e:
            logger.warning(f"Could not read stream value: {e}")
            raise VectorStoreCLOBError(f"Failed to read CLOB stream: {e}")

    # Handle bytes directly
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception as e:
            logger.warning(f"Could not decode bytes value: {e}")
            raise VectorStoreCLOBError(f"Failed to decode bytes: {e}")

    # Handle other types by converting to string
    try:
        return str(value)
    except Exception as e:
        logger.warning(f"Could not convert value to string: {e}")
        raise VectorStoreCLOBError(f"Failed to convert value to string: {e}")


def process_document_row(row_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a database row dictionary to convert CLOB fields to strings.

    This function recursively processes a dictionary representing a database row,
    converting any CLOB-like values to strings. It's particularly useful for
    processing metadata dictionaries that might contain nested CLOB values.

    Args:
        row_dict: Dictionary representing a database row

    Returns:
        Processed dictionary with CLOB fields converted to strings

    Raises:
        VectorStoreCLOBError: If CLOB conversion fails critically
    """
    if not isinstance(row_dict, dict):
        return convert_clob_to_string(row_dict)

    processed_row = {}
    for key, value in row_dict.items():
        try:
            if isinstance(value, dict):
                # Recursively process nested dictionaries
                processed_row[key] = process_document_row(value)
            elif isinstance(value, (list, tuple)):
                # Process lists/tuples of values
                processed_row[key] = [
                    (
                        process_document_row(item)
                        if isinstance(item, dict)
                        else convert_clob_to_string(item)
                    )
                    for item in value
                ]
            else:
                # Convert individual values
                processed_row[key] = convert_clob_to_string(value)
        except VectorStoreCLOBError:
            # Re-raise CLOB errors
            raise
        except Exception as e:
            logger.warning(f"Error processing field '{key}': {e}")
            # For non-critical errors, use a fallback value
            processed_row[key] = f"[Error Processing Field: {e}]"

    return processed_row


def ensure_string_content(document_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure that document data has string content for page_content and metadata.

    This is a specialized function for processing document data retrieved from
    the database, ensuring that page_content and all metadata values are strings.

    Args:
        document_data: Dictionary containing document data with potential CLOBs

    Returns:
        Dictionary with all string content guaranteed

    Raises:
        VectorStoreCLOBError: If critical CLOB conversion fails
    """
    processed_data = {}

    # Process page_content specifically
    if "page_content" in document_data:
        processed_data["page_content"] = convert_clob_to_string(
            document_data["page_content"]
        )
    elif "text_content" in document_data:
        # Handle alternative column name
        processed_data["page_content"] = convert_clob_to_string(
            document_data["text_content"]
        )

    # Process metadata
    if "metadata" in document_data:
        metadata = document_data["metadata"]
        if isinstance(metadata, str):
            # If metadata is a JSON string, we keep it as string for later parsing
            processed_data["metadata"] = convert_clob_to_string(metadata)
        elif isinstance(metadata, dict):
            # If metadata is already a dict, process its values
            processed_data["metadata"] = process_document_row(metadata)
        else:
            # Convert other types to string
            processed_data["metadata"] = convert_clob_to_string(metadata)

    # Process other fields
    for key, value in document_data.items():
        if key not in ["page_content", "text_content", "metadata"]:
            processed_data[key] = convert_clob_to_string(value)

    return processed_data
