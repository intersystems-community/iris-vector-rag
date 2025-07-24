#!/usr/bin/env python3
"""
Fix for IRIS Stream Handling Issue

This script addresses the root cause of RAGAS evaluation failures:
IRISInputStream objects are not being properly converted to strings,
resulting in numeric placeholders instead of actual document content.

The fix involves:
1. Improving the IRISInputStream reading utility
2. Updating all RAG pipelines to use proper stream conversion
3. Testing the fix with actual document retrieval
4. Providing a data validation script
"""

import os
import sys
import logging

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required components
from iris_rag.core.connection import ConnectionManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_iris_stream_improved(stream_obj) -> str:
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

def test_stream_reading():
    """Test the improved stream reading with actual database content."""
    logger.info("=== TESTING IMPROVED STREAM READING ===")
    
    # Initialize connection
    connection_manager = ConnectionManager()
    connection = connection_manager.get_connection()
    cursor = connection.cursor()
    
    try:
        # Get a few sample documents
        sample_sql = """
            SELECT TOP 3 doc_id, text_content, title
            FROM RAG.SourceDocuments
            ORDER BY doc_id
        """
        cursor.execute(sample_sql)
        sample_results = cursor.fetchall()
        
        logger.info("Testing stream reading on sample documents:")
        for i, row in enumerate(sample_results):
            doc_id, text_content, title = row
            logger.info(f"  Document {i+1}: {doc_id}")
            
            # Test text_content reading
            content_str = read_iris_stream_improved(text_content)
            logger.info(f"    text_content length: {len(content_str)}")
            logger.info(f"    text_content preview: {content_str[:200]}...")
            
            # Test title reading
            title_str = read_iris_stream_improved(title)
            logger.info(f"    title: {title_str}")
            
            # Check if we're getting meaningful content
            if len(content_str) > 100 and not content_str.isdigit():
                logger.info(f"    ✅ Successfully read meaningful content")
            else:
                logger.warning(f"    ❌ Content appears to be corrupted or empty")
                
    except Exception as e:
        logger.error(f"Stream reading test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cursor.close()

def validate_document_content():
    """Validate that documents have proper content after stream reading."""
    logger.info("=== VALIDATING DOCUMENT CONTENT ===")
    
    connection_manager = ConnectionManager()
    connection = connection_manager.get_connection()
    cursor = connection.cursor()
    
    try:
        # Check total document count
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
        total_docs = cursor.fetchone()[0]
        logger.info(f"Total documents: {total_docs}")
        
        # Sample documents and check content quality
        cursor.execute("""
            SELECT TOP 10 doc_id, text_content, title
            FROM RAG.SourceDocuments
            ORDER BY doc_id
        """)
        sample_docs = cursor.fetchall()
        
        valid_content_count = 0
        empty_content_count = 0
        numeric_content_count = 0
        
        for doc_id, text_content, title in sample_docs:
            content_str = read_iris_stream_improved(text_content)
            title_str = read_iris_stream_improved(title)
            
            # Classify content quality
            if len(content_str) > 100 and not content_str.isdigit():
                valid_content_count += 1
                logger.info(f"✅ {doc_id}: Valid content ({len(content_str)} chars)")
            elif len(content_str) == 0:
                empty_content_count += 1
                logger.warning(f"⚠️ {doc_id}: Empty content")
            elif content_str.isdigit():
                numeric_content_count += 1
                logger.error(f"❌ {doc_id}: Numeric content: '{content_str}'")
            else:
                logger.warning(f"⚠️ {doc_id}: Short content ({len(content_str)} chars): '{content_str[:50]}...'")
        
        # Summary
        logger.info(f"\n=== CONTENT VALIDATION SUMMARY ===")
        logger.info(f"Valid content: {valid_content_count}/10")
        logger.info(f"Empty content: {empty_content_count}/10")
        logger.info(f"Numeric content: {numeric_content_count}/10")
        
        if numeric_content_count > 0:
            logger.error("❌ ISSUE CONFIRMED: Found numeric content instead of text")
            return False
        elif valid_content_count >= 7:
            logger.info("✅ Content appears to be properly stored and readable")
            return True
        else:
            logger.warning("⚠️ Content quality is questionable")
            return False
            
    except Exception as e:
        logger.error(f"Content validation failed: {e}")
        return False
    finally:
        cursor.close()

def main():
    """Main function to run the stream handling fix and validation."""
    logger.info("🔧 IRIS Stream Handling Fix and Validation")
    logger.info("=" * 60)
    
    # Test improved stream reading
    test_stream_reading()
    
    # Validate document content
    is_valid = validate_document_content()
    
    if is_valid:
        logger.info("\n✅ CONCLUSION: Stream reading is working correctly")
        logger.info("The issue may be in how pipelines handle the streams")
        logger.info("Next step: Update pipeline implementations to use improved stream reading")
    else:
        logger.error("\n❌ CONCLUSION: Stream reading issues confirmed")
        logger.error("The data corruption issue needs to be addressed at the database level")
        logger.error("Consider reloading PMC documents with proper content extraction")

if __name__ == "__main__":
    main()