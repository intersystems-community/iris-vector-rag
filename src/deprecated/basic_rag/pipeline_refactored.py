"""
Refactored BasicRAG Pipeline using centralized connection management
"""

import logging
from typing import List, Dict, Any, Optional
import os # Added for sys.path
import sys # Added for sys.path

# Add project root to sys.path if not already there
# This assumes the script is in src/deprecated/basic_rag/
# and common is in src/common/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import from common, assuming common is at the top level of src or project root
# If common is at the same level as src (i.e. project_root/common), then:
# from common.base_pipeline import BaseRAGPipeline
# from common.utils import Document

# If common is inside src (i.e. project_root/src/common), then:
from src.common.base_pipeline import BaseRAGPipeline
from src.common.utils import Document


logger = logging.getLogger(__name__)

class BasicRAGPipeline(BaseRAGPipeline):
    """Basic RAG pipeline with centralized connection management"""
    
    def retrieve_documents(
        self, 
        query: str, 
        top_k: int = 5,
        similarity_threshold: float = 0.1,
        **kwargs
    ) -> List[Document]:
        """
        Retrieve relevant documents using vector similarity search
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of retrieved documents
        """
        logger.info(f"BasicRAG: Retrieving documents for query: '{query[:50]}...'")
        
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        query_embedding_str = self.format_embedding_for_iris(query_embedding)
        
        # Build SQL query based on connection type
        if self.connection_manager.connection_type == "jdbc":
            # JDBC supports parameterized queries
            sql = """
                SELECT TOP ? 
                    doc_id, 
                    title, 
                    text_content,
                    VECTOR_COSINE(
                        TO_VECTOR(embedding, 'FLOAT', 384),
                        TO_VECTOR(?, 'FLOAT', 384)
                    ) as similarity_score
                FROM RAG.SourceDocuments
                WHERE embedding IS NOT NULL
                AND VECTOR_COSINE(
                    TO_VECTOR(embedding, 'FLOAT', 384),
                    TO_VECTOR(?, 'FLOAT', 384)
                ) > ?
                ORDER BY similarity_score DESC
            """
            params = [top_k, query_embedding_str, query_embedding_str, similarity_threshold]
        else:
            # ODBC needs the query formatted differently
            sql = f"""
                SELECT TOP {top_k} 
                    doc_id, 
                    title, 
                    text_content,
                    VECTOR_COSINE(
                        TO_VECTOR(embedding, 'FLOAT', 384),
                        TO_VECTOR('{query_embedding_str}', 'FLOAT', 384)
                    ) as similarity_score
                FROM RAG.SourceDocuments
                WHERE embedding IS NOT NULL
                AND VECTOR_COSINE(
                    TO_VECTOR(embedding, 'FLOAT', 384),
                    TO_VECTOR('{query_embedding_str}', 'FLOAT', 384)
                ) > {similarity_threshold}
                ORDER BY similarity_score DESC
            """
            params = None
        
        try:
            # Execute query using connection manager
            results = self.execute_query(sql, params)
            
            # Convert to Document objects
            documents = []
            for row in results:
                doc = Document(
                    doc_id=row[0], # Assuming doc_id is the first column
                    title=row[1],  # Assuming title is the second column
                    content=row[2],# Assuming content is the third column
                    score=float(row[3]) # Assuming score is the fourth column
                )
                documents.append(doc)
            
            logger.info(f"BasicRAG: Retrieved {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"BasicRAG: Error retrieving documents: {e}")
            raise
    
    def generate_answer(
        self, 
        query: str, 
        documents: List[Document],
        **kwargs
    ) -> str:
        """
        Generate answer using LLM based on retrieved documents
        
        Args:
            query: User query
            documents: Retrieved documents
            
        Returns:
            Generated answer
        """
        if not self.llm_func:
            raise ValueError("LLM function not provided")
        
        # Use base class prompt creation
        prompt = self.create_prompt(query, documents)
        
        # Generate answer
        answer = self.llm_func(prompt)
        return answer


def test_refactored_pipeline():
    """Test the refactored BasicRAG pipeline"""
    
    print("🔍 Testing Refactored BasicRAG Pipeline")
    print("=" * 50)
    
    # Adjust imports based on new structure
    # Assuming common.utils and common.connection_manager are in src/common/
    from src.common.utils import get_embedding_func, get_llm_func
    from src.common.connection_manager import get_connection_manager
    
    try:
        # Test with default connection (ODBC for stability)
        print("\n📊 Testing with default connection:")
        pipeline = BasicRAGPipeline(
            embedding_func=get_embedding_func(),
            llm_func=get_llm_func()
            # connection_manager will be initialized by BaseRAGPipeline
        )
        
        query = "What are the symptoms of diabetes?"
        result = pipeline.run(query, top_k=3)
        
        print(f"✅ Pipeline successful with {result['metadata']['connection_type']}!")
        print(f"   - Retrieved {result['metadata']['num_documents']} documents")
        print(f"   - Total time: {result['metadata']['total_time']:.3f}s")
        print(f"   - Answer preview: {result['answer'][:100]}...")
        
        # Show document scores
        print("\n📊 Document Scores:")
        for doc_dict in result['retrieved_documents']: # Iterate over dicts
            print(f"   - {doc_dict.get('doc_id', doc_dict.get('id'))}: {doc_dict.get('score'):.4f}") # Access score from dict
        
        # Try JDBC if available
        print("\n📊 Testing with JDBC connection:")
        try:
            jdbc_pipeline = BasicRAGPipeline(
                connection_manager=get_connection_manager("jdbc"),
                embedding_func=get_embedding_func(),
                llm_func=get_llm_func()
            )
            
            result = jdbc_pipeline.run(query, top_k=3)
            print(f"✅ JDBC Pipeline successful!")
            print(f"   - Retrieved {result['metadata']['num_documents']} documents")
            print(f"   - Total time: {result['metadata']['total_time']:.3f}s")
        except Exception as e:
            print(f"⚠️  JDBC Pipeline failed: {e}")
            print("   This is expected if JDBC is not properly configured")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Ensure correct sys.path for direct execution if common is not in PYTHONPATH
    # This is a bit redundant with the top-level sys.path modification but ensures
    # __main__ block works if common is structured as src/common
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming this file is in src/deprecated/basic_rag
    # and common is in src/common
    # So, go up three levels to project_root, then into src/
    path_to_src = os.path.abspath(os.path.join(current_dir, '../../..'))
    if path_to_src not in sys.path:
         sys.path.insert(0, path_to_src)

    test_refactored_pipeline()