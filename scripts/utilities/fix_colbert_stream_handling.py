#!/usr/bin/env python3
"""
Fix ColBERT Pipeline Stream Handling

This script fixes the ColBERT pipeline to properly handle IRISInputStream objects
and convert them to strings for RAGAS evaluation.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def fix_colbert_pipeline():
    """Apply the stream handling fix to the ColBERT pipeline."""
    
    # Read the current ColBERT pipeline
    colbert_pipeline_path = "colbert/pipeline.py"
    
    with open(colbert_pipeline_path, 'r') as f:
        content = f.read()
    
    # Check if the fix is already applied
    if "from common.jdbc_stream_utils_fixed import read_iris_stream" in content:
        print("✅ ColBERT pipeline already has stream handling fix applied")
        return True
    
    # Apply the fix by adding the import and modifying the document content handling
    lines = content.split('\n')
    
    # Find the import section and add our import
    import_added = False
    for i, line in enumerate(lines):
        if line.startswith("from common.utils import") and not import_added:
            lines.insert(i + 1, "from common.jdbc_stream_utils_fixed import read_iris_stream")
            import_added = True
            break
    
    if not import_added:
        # Add import after existing imports
        for i, line in enumerate(lines):
            if line.startswith("from common.") and i < 30:  # Within first 30 lines
                lines.insert(i + 1, "from common.jdbc_stream_utils_fixed import read_iris_stream")
                import_added = True
                break
    
    # Find the line where doc_contents is created and fix it
    for i, line in enumerate(lines):
        if "doc_contents = {doc_row[0]: doc_row[1] for doc_row in docs_data}" in line:
            # Replace with stream-aware version
            lines[i] = "            doc_contents = {doc_row[0]: read_iris_stream(doc_row[1]) for doc_row in docs_data}"
            print("✅ Fixed doc_contents creation to use stream reading")
            break
    
    # Write the fixed content back
    with open(colbert_pipeline_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print("✅ Applied stream handling fix to ColBERT pipeline")
    return True

def create_test_script():
    """Create a test script to verify the fix works."""
    
    test_script_content = '''#!/usr/bin/env python3
"""
Test script to verify ColBERT stream handling fix
"""

import os
import sys
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from colbert.pipeline import ColBERTRAGPipeline
from common.iris_connector import get_iris_connection
from common.utils import get_colbert_query_encoder_func, get_colbert_doc_encoder_func, get_llm_func

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_colbert_stream_handling():
    """Test that ColBERT pipeline properly handles streams."""
    
    logger.info("🧪 Testing ColBERT stream handling fix...")
    
    try:
        # Initialize pipeline components
        iris_connector = get_iris_connection()
        colbert_query_encoder = get_colbert_query_encoder_func()
        colbert_doc_encoder = get_colbert_doc_encoder_func()
        llm_func = get_llm_func()
        
        # Create pipeline
        pipeline = ColBERTRAGPipeline(
            iris_connector=iris_connector,
            colbert_query_encoder_func=colbert_query_encoder,
            colbert_doc_encoder_func=colbert_doc_encoder,
            llm_func=llm_func
        )
        
        # Test with a simple query
        test_query = "What is cancer treatment?"
        
        logger.info(f"Testing query: {test_query}")
        result = pipeline.query(test_query)
        
        # Check if we got meaningful results
        if result and "retrieved_documents" in result:
            docs = result["retrieved_documents"]
            logger.info(f"Retrieved {len(docs)} documents")
            
            # Check document content
            for i, doc in enumerate(docs[:3]):  # Check first 3 docs
                content = getattr(doc, 'content', '') or getattr(doc, 'page_content', '')
                logger.info(f"Doc {i+1} content length: {len(content)}")
                logger.info(f"Doc {i+1} content preview: {content[:100]}...")
                
                # Check if content is meaningful (not just numeric)
                if len(content) > 50 and not content.isdigit():
                    logger.info(f"✅ Doc {i+1}: Meaningful content found")
                else:
                    logger.warning(f"❌ Doc {i+1}: Content appears corrupted: '{content}'")
            
            logger.info("✅ ColBERT stream handling test completed")
            return True
        else:
            logger.error("❌ No documents retrieved")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_colbert_stream_handling()
    if success:
        print("\\n✅ ColBERT stream handling fix is working correctly")
    else:
        print("\\n❌ ColBERT stream handling fix needs more work")
'''
    
    with open("test_colbert_stream_fix.py", 'w') as f:
        f.write(test_script_content)
    
    print("✅ Created test script: test_colbert_stream_fix.py")

def main():
    """Main function to apply the ColBERT stream handling fix."""
    
    print("🔧 Fixing ColBERT Pipeline Stream Handling")
    print("=" * 50)
    
    # Apply the fix
    if fix_colbert_pipeline():
        print("✅ ColBERT pipeline fix applied successfully")
    else:
        print("❌ Failed to apply ColBERT pipeline fix")
        return False
    
    # Create test script
    create_test_script()
    
    print("\n📋 Next Steps:")
    print("1. Run the test script: python test_colbert_stream_fix.py")
    print("2. If successful, run RAGAS evaluation to verify the fix")
    print("3. Apply similar fixes to other RAG pipelines if needed")
    
    return True

if __name__ == "__main__":
    main()