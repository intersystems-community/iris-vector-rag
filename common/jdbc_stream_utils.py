"""
Utilities for handling JDBC stream objects (BLOB/CLOB) from IRIS
"""

import logging

logger = logging.getLogger(__name__)

def read_iris_stream(stream_obj):
    """
    Read content from an IRISInputStream object
    
    Args:
        stream_obj: The IRISInputStream object from JDBC
        
    Returns:
        str: The decoded content or empty string if unable to read
    """
    if stream_obj is None:
        return ""
        
    try:
        # Check if it's already a string
        if isinstance(stream_obj, str):
            return stream_obj
            
        # Try to read from the stream
        if hasattr(stream_obj, 'read'):
            content = stream_obj.read()
            if isinstance(content, bytes):
                return content.decode('utf-8', errors='ignore')
            return str(content)
            
        # If it's an IRISInputStream, try to get its string representation
        # This is a workaround for JDBC stream handling
        stream_str = str(stream_obj)
        if stream_str.startswith('com.intersystems.jdbc.IRISInputStream@'):
            logger.warning(f"Unable to read IRISInputStream content directly: {stream_str}")
            return ""
            
        return stream_str
        
    except Exception as e:
        logger.error(f"Error reading IRIS stream: {e}")
        return ""

def process_document_with_streams(doc):
    """
    Process a document dictionary that may contain IRIS stream objects
    
    Args:
        doc: Document dictionary with potential stream objects
        
    Returns:
        dict: Document with streams converted to strings
    """
    if not isinstance(doc, dict):
        return doc
        
    processed = doc.copy()
    
    # Fields that might contain streams
    stream_fields = ['content', 'text', 'body', 'document_text']
    
    for field in stream_fields:
        if field in processed:
            processed[field] = read_iris_stream(processed[field])
            
    return processed