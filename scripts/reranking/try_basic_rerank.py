import logging
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import iris_rag

# Optional: Dummy LLM function
def dummy_llm(prompt: str) -> str:
    print("\n--- Prompt to LLM ---\n")
    print(prompt)
    return "This is a dummy answer generated from the context."

def main():
    # Setup logging
    logging.basicConfig(level=logging.DEBUG)  # Set to INFO or WARNING to reduce verbosity
    logger = logging.getLogger()

    print("Creating RAG Reranking Pipeline with Auto-Setup")
    # Create pipeline using iris_rag factory with auto_setup=True
    # This ensures database schema is properly initialized
    reranking_rag_pipeline = iris_rag.create_pipeline(
        pipeline_type="basic_rerank",
        llm_func=dummy_llm,  # Replace with real LLM call if available
        auto_setup=True,     # Crucial: handles schema initialization automatically
        validate_requirements=True
    )
    print("✓ RAG Reranking Pipeline created successfully")

    print("Loading data")
    # Step 1: Load documents from a folder
    doc_path = "../../data/test_txt_docs"
    reranking_rag_pipeline.load_documents(doc_path)

    print("Running RAG + Reranking Pipeline")
    # Step 2: Run a sample query
    query = "What is InterSystems IRIS?"
    response = reranking_rag_pipeline.query(query, top_k=3)

    # Step 3: Print final answer
    print("\n========== RAG + Reranking Pipeline Output ==========")
    print(f"Query: {response['query']}")
    print(f"Answer: {response['answer']}")
    print(f"Execution Time: {response['execution_time']:.2f}s")

    # Step 4: Show retrieved sources
    print("\n--- Retrieved Sources ---")
    for source in response.get("sources", []):
        print(source)

    # Step 5: Show full context
    print("\n--- Retrieved Contexts ---")
    for i, ctx in enumerate(response['contexts'], 1):
        print(f"\n[Context {i}]\n{ctx[:300]}...")

    # Step 6: Clean up test data (as suggested by intern)
    print("\n--- Cleanup ---")
    try:
        # Get document count before cleanup
        connection = reranking_rag_pipeline.connection_manager.get_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
        count_before = cursor.fetchone()[0]
        
        # Clear all documents loaded during this test
        print(f"Documents in database before cleanup: {count_before}")
        
        # Clear documents from this test run (they should have the test data path in metadata)
        cursor.execute("""
            DELETE FROM RAG.SourceDocuments 
            WHERE metadata LIKE '%test_txt_docs%'
        """)
        
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
        count_after = cursor.fetchone()[0]
        documents_removed = count_before - count_after
        
        connection.commit()
        cursor.close()
        
        print(f"Documents removed: {documents_removed}")
        print(f"Documents remaining: {count_after}")
        print("✅ Cleanup completed successfully")
        
    except Exception as cleanup_error:
        print(f"⚠️ Cleanup failed (this is usually fine): {cleanup_error}")

if __name__ == "__main__":
    main()
