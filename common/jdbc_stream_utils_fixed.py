"""
Fixed utilities for handling JDBC stream objects (BLOB/CLOB) from IRIS

This module provides improved stream reading capabilities that properly
handle IRISInputStream objects and convert them to usable strings.
"""

import logging

logger = logging.getLogger(__name__)

def read_iris_stream(stream_obj):
    """
    Improved IRISInputStream reader that handles various stream types.
    
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
            
        # Check if it's a numeric value that got converted incorrectly
        if isinstance(stream_obj, (int, float)):
            logger.warning(f"Found numeric value instead of text content: {stream_obj}")
            return str(stream_obj)
            
        # Try to read from the stream using different methods
        if hasattr(stream_obj, 'read'):
            logger.debug(f"Attempting to read from stream object: {type(stream_obj)}")
            
            # Method 1: Try reading all at once if available
            if hasattr(stream_obj, 'readAllBytes'):
                try:
                    byte_array = stream_obj.readAllBytes()
                    if byte_array:
                        content_bytes = bytes(byte_array)
                        decoded_content = content_bytes.decode('utf-8', errors='ignore')
                        logger.debug(f"Successfully read {len(content_bytes)} bytes using readAllBytes")
                        return decoded_content
                except Exception as e:
                    logger.debug(f"readAllBytes failed: {e}")
            
            # Method 2: Try reading with buffer
            if hasattr(stream_obj, 'read') and hasattr(stream_obj, 'available'):
                try:
                    available = stream_obj.available()
                    if available > 0:
                        # Create a buffer to read the available bytes
                        buffer = bytearray(available)
                        bytes_read = stream_obj.read(buffer)
                        if bytes_read > 0:
                            content_bytes = bytes(buffer[:bytes_read])
                            decoded_content = content_bytes.decode('utf-8', errors='ignore')
                            logger.debug(f"Successfully read {bytes_read} bytes using buffered read")
                            return decoded_content
                except Exception as e:
                    logger.debug(f"Buffered read failed: {e}")
            
            # Method 3: Byte-by-byte reading (fallback)
            try:
                byte_list = []
                max_bytes = 1000000  # 1MB limit to prevent infinite loops
                bytes_read = 0
                
                while bytes_read < max_bytes:
                    byte_val = stream_obj.read()
                    if byte_val == -1:  # End of stream
                        break
                    if byte_val < 0 or byte_val > 255:
                        logger.warning(f"Invalid byte value: {byte_val}")
                        break
                    byte_list.append(byte_val)
                    bytes_read += 1
                
                if byte_list:
                    content_bytes = bytes(byte_list)
                    decoded_content = content_bytes.decode('utf-8', errors='ignore')
                    logger.debug(f"Successfully read {len(content_bytes)} bytes using byte-by-byte")
                    return decoded_content
                    
            except Exception as e:
                logger.debug(f"Byte-by-byte read failed: {e}")
        
        # Method 4: Try to get string representation
        try:
            stream_str = str(stream_obj)
            if not stream_str.startswith('com.intersystems.jdbc.IRISInputStream@'):
                # If it's not just the object reference, it might be actual content
                return stream_str
            else:
                logger.warning(f"Got object reference instead of content: {stream_str}")
        except Exception as e:
            logger.debug(f"String conversion failed: {e}")
            
        # Method 5: Try to access underlying data if it's a wrapper
        if hasattr(stream_obj, 'toString'):
            try:
                content = stream_obj.toString()
                if content and not content.startswith('com.intersystems.jdbc.IRISInputStream@'):
                    return content
            except Exception as e:
                logger.debug(f"toString() failed: {e}")
                
        logger.warning(f"Unable to read content from stream object: {type(stream_obj)}")
        return ""
        
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
    stream_fields = ['content', 'text', 'body', 'document_text', 'text_content', 'title', 'abstract']
    
    for field in stream_fields:
        if field in processed:
            processed[field] = read_iris_stream(processed[field])
            
    return processed

def convert_stream_to_string(value):
    """
    Convert any value that might be a stream to a string.
    
    Args:
        value: Any value that might be an IRISInputStream
        
    Returns:
        str: String representation of the value
    """
    return read_iris_stream(value)