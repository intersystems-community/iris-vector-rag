"""
Utility to handle IRIS JDBC stream conversions
"""

import logging

logger = logging.getLogger(__name__)

def convert_iris_stream_to_string(stream_obj):
    """
    Convert IRIS JDBC stream objects to strings
    
    Args:
        stream_obj: IRISInputStream or similar JDBC stream object
        
    Returns:
        str: The string content of the stream
    """
    if stream_obj is None:
        return ""
    
    # Check if it's already a string
    if isinstance(stream_obj, str):
        return stream_obj
    
    # Check if it's an IRIS stream object
    if hasattr(stream_obj, '__class__') and 'IRISInputStream' in str(stream_obj.__class__):
        try:
            # Try to read the stream
            if hasattr(stream_obj, 'read'):
                # Read all bytes and decode
                content = stream_obj.read()
                if isinstance(content, bytes):
                    return content.decode('utf-8', errors='ignore')
                return str(content)
            elif hasattr(stream_obj, 'getBytes'):
                # Alternative method for JDBC streams
                bytes_data = stream_obj.getBytes(1, stream_obj.length())
                return bytes_data.decode('utf-8', errors='ignore')
            elif hasattr(stream_obj, 'getCharacterStream'):
                # Try character stream
                reader = stream_obj.getCharacterStream()
                chars = []
                while True:
                    char = reader.read()
                    if char == -1:
                        break
                    chars.append(chr(char))
                return ''.join(chars)
        except Exception as e:
            logger.warning(f"Failed to convert IRIS stream: {e}")
            return f"[Stream conversion error: {e}]"
    
    # Fallback to string representation
    return str(stream_obj)

def safe_get_document_content(row, content_index):
    """
    Safely extract document content from a database row
    
    Args:
        row: Database row tuple
        content_index: Index of the content field in the row
        
    Returns:
        str: The document content as a string
    """
    try:
        content = row[content_index]
        return convert_iris_stream_to_string(content)
    except Exception as e:
        logger.error(f"Error extracting document content: {e}")
        return ""