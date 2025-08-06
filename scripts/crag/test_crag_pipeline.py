"""
Test script for CRAG Pipeline.

This script tests the CRAGPipeline implementation by executing a sample query
and logging the results.
"""

import logging
import sys
import os
import openai
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import iris_rag

load_dotenv()

# === CONFIGURATION ===
USE_REAL_LLM = True  # Change to False to use dummy_llm
OPENAI_MODEL = "gpt-4.1-mini"  # GPT-4.1 Mini
client = openai.OpenAI()

# Optional: Dummy LLM function
def dummy_llm(prompt: str) -> str:
    print("\n--- Prompt to LLM ---\n")
    print(prompt)
    return "This is a dummy answer generated from the context."

# Real LLM function using OpenAI GPT-4.1 Mini
def openai_llm(prompt: str) -> str:
    print("\n--- Prompt to LLM (OpenAI) ---\n")
    print(prompt)

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant answering based on the context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()

def main():
    # Setup logging
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger()

    llm_func = openai_llm if USE_REAL_LLM else dummy_llm

    print("Creating CRAG Pipeline with Auto-Setup")
    # Create pipeline using iris_rag factory with auto_setup=True
    crag_pipeline = iris_rag.create_pipeline(
        pipeline_type="crag",
        llm_func=llm_func,  
        auto_setup=True,     # Crucial: handles schema initialization automatically
        validate_requirements=True
    )
    print("✓ CRAG Pipeline created successfully")

    print("Loading data")
    # Load documents from a folder (update the path as necessary)
    doc_path = "../../data/test_txt_docs"
    crag_pipeline.load_documents(doc_path)

    print("Running CRAG Pipeline")
    # Run a sample query
    query = "What are the benefits of using RAG pipelines?"
    response = crag_pipeline.query(query, top_k=3)

    # Print final answer
    print("\n========== CRAG Pipeline Output ==========")
    print(f"Query: {response['query']}")
    print(f"Answer: {response['answer']}")
    print(f"Execution Time: {response['execution_time']:.2f}s")

    # Show retrieved sources
    print("\n--- Retrieved Sources ---")
    for source in response.get("sources", []):
        print(source)

    # Show full context
    print("\n--- Retrieved Contexts ---")
    for i, ctx in enumerate(response['contexts'], 1):
        print(f"\n[Context {i}]\n{ctx[:300]}...")

    # Step 6: Clean up test data
    print("\n--- Cleanup ---")
    try:
        # Get document count before cleanup
        connection = crag_pipeline.connection_manager.get_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
        count_before = cursor.fetchone()[0]
        
        # Clear all documents loaded during this test
        print(f"Documents in database before cleanup: {count_before}")
        
        # Clear documents from this test run (they should have the test data path in metadata)
        import json

        # Step 1: Fetch all rows
        cursor.execute("SELECT doc_id, metadata FROM RAG.SourceDocuments")
        rows = cursor.fetchall()

        # Step 2: Identify doc_ids to delete
        doc_ids_to_delete = []

        for row in rows:
            doc_id = row[0]
            metadata_raw = row[1]

            # Decode metadata if it's a byte stream
            if isinstance(metadata_raw, (bytes, bytearray)):
                metadata_raw = metadata_raw.decode("utf-8")

            try:
                metadata_json = json.loads(metadata_raw)
                source = metadata_json.get("source", "")
                if "test_txt_docs" in source:
                    doc_ids_to_delete.append(doc_id)
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Skipping row {doc_id}: malformed metadata - {e}")

        # Step 3: Delete matching rows
        print(f"Found {len(doc_ids_to_delete)} documents to delete.")

        for doc_id in doc_ids_to_delete:
            cursor.execute("DELETE FROM RAG.SourceDocuments WHERE doc_id = ?", [doc_id])
        
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