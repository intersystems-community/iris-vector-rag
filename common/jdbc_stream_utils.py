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
            logger.debug(f"Attempting to read from stream object: {type(stream_obj)}")
            byte_list = []
            # Java InputStream's read() returns a single byte as an int (0-255), or -1 for EOF.
            try:
                while True:
                    byte_val = stream_obj.read()
                    if byte_val == -1:  # End of stream
                        logger.debug("EOF reached while reading stream.")
                        break
                    if byte_val < 0 or byte_val > 255: # Should not happen for valid byte reads
                        logger.error(f"Invalid byte value read from stream: {byte_val}")
                        break
                    byte_list.append(byte_val)
                
                if not byte_list:
                    logger.warning("Stream was empty or read yielded no bytes.")
                    return ""

                content_bytes = bytes(byte_list)
                logger.debug(f"Successfully read {len(content_bytes)} bytes from stream.")
                decoded_content = content_bytes.decode('utf-8', errors='ignore')
                logger.debug(f"Decoded stream content snippet: '{decoded_content[:200]}'")
                return decoded_content
            except Exception as e_read:
                logger.error(f"Exception during stream_obj.read() loop: {e_read}", exc_info=True)
                # Fall through to string representation as a last resort if read loop fails
                pass # Fall through to attempt str(stream_obj)

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