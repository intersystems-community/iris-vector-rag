import logging
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from iris_rag.pipelines.basic_rerank import BasicRAGRerankingPipeline
from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager
from iris_rag.storage.vector_store_iris import IRISVectorStore

# Optional: Dummy LLM function
def dummy_llm(prompt: str) -> str:
    print("\n--- Prompt to LLM ---\n")
    print(prompt)
    return "This is a dummy answer generated from the context."

def main():
    # Setup logging
    logging.basicConfig(level=logging.DEBUG)  # Set to INFO or WARNING to reduce verbosity
    logger = logging.getLogger()

    print("Instantiating Managers")
    # Instantiate core components
    connection_manager = ConnectionManager()
    config_manager = ConfigurationManager()
    vector_store = IRISVectorStore(config_manager=config_manager)

    print("Instantiating RAG Reranking Pipeline")
    # Instantiate the RAG pipeline
    reranking_rag_pipeline = BasicRAGRerankingPipeline(
        connection_manager=connection_manager,
        config_manager=config_manager,
        vector_store=vector_store,
        llm_func=dummy_llm  # Replace with real LLM call if available
    )

    print("Loading data")
    # Step 1: Load documents from a folder
    doc_path = "./data/test_txt_docs"
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

if __name__ == "__main__":
    main()
