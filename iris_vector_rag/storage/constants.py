"""
Storage constants for iris-vector-rag.

This module defines default metadata filter keys and other storage-related
constants used across the RAG framework.

Constants:
    DEFAULT_FILTER_KEYS: List of metadata fields that are always allowed for filtering
"""

# Default metadata filter keys (always enabled for all users)
# These 17 standard fields can always be used in metadata_filter parameter
# Custom fields can be added via storage.iris.custom_filter_keys in config
DEFAULT_FILTER_KEYS = [
    "document_id",          # Unique document identifier
    "source",               # Source file or URL
    "title",                # Document title
    "author",               # Document author(s)
    "created_at",           # Creation timestamp
    "updated_at",           # Last update timestamp
    "content_type",         # MIME type (e.g., text/plain, application/pdf)
    "file_type",            # File extension (e.g., txt, pdf, md)
    "page_number",          # Page number for multi-page documents
    "section",              # Section or chapter identifier
    "chunk_id",             # Unique chunk identifier
    "chunk_index",          # Sequential chunk index within document
    "language",             # Document language (e.g., en, es, fr)
    "collection_id",        # Collection identifier for document grouping
    "metadata_hash",        # Hash of metadata for change detection
    "parent_document_id",   # Parent document ID for hierarchical documents
    "version",              # Document version number
]

# SQL injection prevention patterns
# These patterns are used to validate custom filter keys
INVALID_FIELD_NAME_PATTERNS = [
    r"[;'\"]",              # SQL injection characters
    r"--",                  # SQL comment
    r"/\*",                 # SQL block comment start
    r"\*/",                 # SQL block comment end
    r"\bDROP\b",            # DROP statement
    r"\bDELETE\b",          # DELETE statement
    r"\bUPDATE\b",          # UPDATE statement
    r"\bINSERT\b",          # INSERT statement
    r"\bEXEC\b",            # EXEC statement
    r"\bEXECUTE\b",         # EXECUTE statement
    r"\bSELECT\b",          # SELECT statement
    r"\bUNION\b",           # UNION statement
]

# Valid field name pattern
# Field names must be alphanumeric + underscores, starting with letter/underscore
VALID_FIELD_NAME_PATTERN = r"^[a-zA-Z_][a-zA-Z0-9_]*$"

# Maximum allowed field name length
MAX_FIELD_NAME_LENGTH = 128

# Collection ID validation
VALID_COLLECTION_ID_PATTERN = r"^[a-zA-Z0-9_-]+$"
MAX_COLLECTION_ID_LENGTH = 128

# Batch operation limits
DEFAULT_BATCH_SIZE = 1000
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 10000

# Schema discovery limits
DEFAULT_SAMPLE_SIZE = 100
MIN_SAMPLE_SIZE = 10
MAX_SAMPLE_SIZE = 200
