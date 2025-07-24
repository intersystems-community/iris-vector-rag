#!/usr/bin/env python3
"""
Simple script to run BasicRAG pipeline and inspect the exact response structure.
This helps understand why contexts aren't being extracted in RAGAs evaluation.
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import required components
from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager
from iris_rag.pipelines.basic import BasicRAGPipeline
from iris_rag.storage.vector_store_iris import IRISVectorStore
from iris_rag.core.models import Document

# LangChain imports for LLM
from langchain_openai import ChatOpenAI

def create_sample_documents():
    """Create some sample documents for testing."""
    documents = [
        Document(
            page_content="Diabetes is a chronic disease that occurs when the pancreas is no longer able to make insulin, or when the body cannot make good use of the insulin it produces. The main types are Type 1 and Type 2 diabetes.",
            metadata={"source": "medical_doc1.txt", "topic": "diabetes"}
        ),
        Document(
            page_content="Machine learning is a subset of artificial intelligence that involves training algorithms to learn patterns from data. It enables computers to make predictions or decisions without being explicitly programmed.",
            metadata={"source": "tech_doc1.txt", "topic": "ml"}
        ),
        Document(
            page_content="Mitochondria are membrane-bound cell organelles that generate most of the chemical energy needed to power the cell's biochemical reactions. They are often called the powerhouses of the cell.",
            metadata={"source": "bio_doc1.txt", "topic": "biology"}
        )
    ]
    return documents

def inspect_object_structure(obj, name="Object", max_depth=3, current_depth=0):
    """Recursively inspect object structure."""
    indent = "  " * current_depth
    print(f"{indent}{name}:")
    print(f"{indent}  Type: {type(obj).__name__}")
    
    if current_depth >= max_depth:
        print(f"{indent}  [Max depth reached]")
        return
    
    if isinstance(obj, dict):
        print(f"{indent}  Keys: {list(obj.keys())}")
        for key, value in obj.items():
            if isinstance(value, (dict, list)) and current_depth < max_depth - 1:
                inspect_object_structure(value, f"'{key}'", max_depth, current_depth + 1)
            else:
                value_type = type(value).__name__
                value_preview = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                print(f"{indent}  '{key}': {value_type} = {value_preview}")
    
    elif isinstance(obj, list):
        print(f"{indent}  Length: {len(obj)}")
        if obj:
            print(f"{indent}  First item type: {type(obj[0]).__name__}")
            if len(obj) > 0 and current_depth < max_depth - 1:
                inspect_object_structure(obj[0], f"First item", max_depth, current_depth + 1)
    
    elif hasattr(obj, '__dict__'):
        attrs = vars(obj)
        print(f"{indent}  Attributes: {list(attrs.keys())}")
        for attr, value in attrs.items():
            value_type = type(value).__name__
            value_preview = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
            print(f"{indent}  {attr}: {value_type} = {value_preview}")

def main():
    """Main function to run BasicRAG and inspect response."""
    print("="*80)
    print("BasicRAG Pipeline Response Structure Inspector")
    print("="*80)
    
    # Initialize configuration and connection
    config_manager = ConfigurationManager()
    connection_manager = ConnectionManager(config_manager)
    
    # Create LLM function
    if os.getenv("OPENAI_API_KEY"):
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=1024
        )
        llm_func = lambda prompt: llm.invoke(prompt).content
    else:
        print("Warning: OPENAI_API_KEY not found. Using mock LLM for testing.")
        # Mock LLM function that returns a simple response
        def mock_llm(prompt):
            return f"This is a mock response to the query. The provided context suggests relevant information about the topic."
        llm_func = mock_llm
    
    # Create vector store
    vector_store = IRISVectorStore(connection_manager, config_manager)
    
    # Initialize BasicRAG pipeline
    print("\n1. Initializing BasicRAG pipeline...")
    pipeline = BasicRAGPipeline(
        connection_manager=connection_manager,
        config_manager=config_manager,
        llm_func=llm_func,
        vector_store=vector_store
    )
    
    # Load sample documents
    print("\n2. Loading sample documents...")
    documents = create_sample_documents()
    pipeline.load_documents(
        documents_path="dummy_path",  # Not used when documents are provided directly
        documents=documents,
        chunk_documents=False,  # Don't chunk for this test
        generate_embeddings=True
    )
    print(f"   Loaded {len(documents)} documents")
    
    # Test queries
    test_queries = [
        "What are the main causes of diabetes?",
        "How does machine learning work?",
        "What is the role of mitochondria in cells?"
    ]
    
    print("\n3. Running test queries and inspecting responses...\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}: {query}")
        print(f"{'='*60}")
        
        # Execute pipeline using both methods
        print("\n--- Using execute() method ---")
        result_execute = pipeline.execute(query, top_k=2)
        
        print("\nResponse structure from execute():")
        inspect_object_structure(result_execute, "execute() result")
        
        print("\n--- Using run() method ---")
        result_run = pipeline.run(query, top_k=2)
        
        print("\nResponse structure from run():")
        inspect_object_structure(result_run, "run() result")
        
        # Compare the two results
        print("\n--- Comparison ---")
        print(f"execute() and run() return same object: {result_execute is result_run}")
        print(f"execute() keys: {set(result_execute.keys())}")
        print(f"run() keys: {set(result_run.keys())}")
        
        # Specifically check contexts field
        print("\n--- Contexts Field Analysis ---")
        if 'contexts' in result_execute:
            contexts = result_execute['contexts']
            print(f"Contexts type: {type(contexts)}")
            print(f"Contexts length: {len(contexts) if isinstance(contexts, list) else 'N/A'}")
            if isinstance(contexts, list) and contexts:
                print(f"First context type: {type(contexts[0])}")
                print(f"First context preview: {str(contexts[0])[:200]}...")
                
                # Check if all contexts are strings
                all_strings = all(isinstance(ctx, str) for ctx in contexts)
                print(f"All contexts are strings: {all_strings}")
        else:
            print("NO 'contexts' field found in response!")
        
        # Check retrieved_documents field
        print("\n--- Retrieved Documents Analysis ---")
        if 'retrieved_documents' in result_execute:
            docs = result_execute['retrieved_documents']
            print(f"Retrieved documents type: {type(docs)}")
            print(f"Retrieved documents length: {len(docs) if isinstance(docs, list) else 'N/A'}")
            if isinstance(docs, list) and docs:
                print(f"First document type: {type(docs[0])}")
                if hasattr(docs[0], '__dict__'):
                    print(f"First document attributes: {list(vars(docs[0]).keys())}")
                if hasattr(docs[0], 'page_content'):
                    print(f"First document page_content preview: {docs[0].page_content[:200]}...")
        
        # Save full response for detailed inspection
        output_file = f"basicrag_response_query_{i}.json"
        
        # Convert response to JSON-serializable format
        json_response = {}
        for key, value in result_execute.items():
            if key == 'retrieved_documents' and isinstance(value, list):
                # Convert Document objects to dicts
                json_response[key] = []
                for doc in value:
                    if hasattr(doc, 'to_dict'):
                        json_response[key].append(doc.to_dict())
                    elif hasattr(doc, '__dict__'):
                        doc_dict = {
                            'page_content': getattr(doc, 'page_content', ''),
                            'metadata': getattr(doc, 'metadata', {}),
                            'id': str(getattr(doc, 'id', ''))
                        }
                        json_response[key].append(doc_dict)
                    else:
                        json_response[key].append(str(doc))
            else:
                json_response[key] = value
        
        with open(output_file, 'w') as f:
            json.dump(json_response, f, indent=2, default=str)
        print(f"\nFull response saved to: {output_file}")
    
    print("\n" + "="*80)
    print("Inspection complete!")
    print("="*80)

if __name__ == "__main__":
    main()