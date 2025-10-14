#!/usr/bin/env python3
"""
Test script to verify the context extraction issue in RAGAs evaluation.
This demonstrates that BasicRAG already provides contexts as strings.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from iris_rag.core.models import Document


def mock_pipeline_response():
    """Create a mock BasicRAG response to demonstrate the structure"""
    return {
        "query": "What is RAG?",
        "answer": "RAG stands for Retrieval Augmented Generation...",
        "retrieved_documents": [
            Document(
                page_content="RAG is a technique that combines retrieval and generation.",
                metadata={"source": "doc1.txt"},
            ),
            Document(
                page_content="It retrieves relevant documents and uses them for generation.",
                metadata={"source": "doc2.txt"},
            ),
        ],
        "contexts": [
            "RAG is a technique that combines retrieval and generation.",
            "It retrieves relevant documents and uses them for generation.",
        ],
        "execution_time": 0.5,
        "sources": [
            {"source": "doc1.txt", "chunk_index": 0},
            {"source": "doc2.txt", "chunk_index": 1},
        ],
        "metadata": {
            "num_retrieved": 2,
            "processing_time": 0.5,
            "pipeline_type": "basic_rag",
        },
    }


def extract_contexts_wrong_way(documents):
    """The current way contexts are extracted (from documents)"""
    contexts = []
    for doc in documents:
        if isinstance(doc, dict):
            text = (
                doc.get("text", "")
                or doc.get("content", "")
                or doc.get("chunk_text", "")
            )
        elif hasattr(doc, "page_content"):
            text = doc.page_content
        elif hasattr(doc, "text"):
            text = doc.text
        elif hasattr(doc, "content"):
            text = doc.content
        else:
            text = str(doc)
        if text:
            contexts.append(text)
    return contexts


def extract_contexts_correct_way(result):
    """The correct way - use the contexts key directly"""
    # First check if contexts are already provided
    if "contexts" in result and isinstance(result["contexts"], list):
        return result["contexts"]

    # Fall back to extracting from documents if contexts not provided
    documents = result.get("retrieved_documents", [])
    return extract_contexts_wrong_way(documents)


def main():
    print("Testing RAGAs Context Extraction\n")

    # Get mock response
    response = mock_pipeline_response()

    print("1. BasicRAG Response Structure:")
    print(f"   - Contains 'contexts' key: {'contexts' in response}")
    print(f"   - Contexts type: {type(response.get('contexts'))}")
    print(f"   - Number of contexts: {len(response.get('contexts', []))}")
    print(f"   - First context: {response['contexts'][0][:50]}...")

    print("\n2. Current Extraction Method (from documents):")
    contexts_wrong = extract_contexts_wrong_way(response["retrieved_documents"])
    print(f"   - Number of contexts extracted: {len(contexts_wrong)}")
    print(
        f"   - First context: {contexts_wrong[0][:50] if contexts_wrong else 'None'}..."
    )

    print("\n3. Correct Extraction Method (use contexts key):")
    contexts_correct = extract_contexts_correct_way(response)
    print(f"   - Number of contexts extracted: {len(contexts_correct)}")
    print(
        f"   - First context: {contexts_correct[0][:50] if contexts_correct else 'None'}..."
    )

    print("\n4. Comparison:")
    print(
        f"   - Both methods produce same result: {contexts_wrong == contexts_correct}"
    )
    print(
        f"   - Contexts are already strings: {all(isinstance(c, str) for c in response['contexts'])}"
    )

    print("\n5. Recommendation:")
    print(
        "   The RAGAs evaluation should be updated to use the 'contexts' key directly"
    )
    print("   when it's available in the response, as BasicRAG already provides it.")

    # Show the fix
    print("\n6. Suggested Fix in unified_ragas_evaluation_framework.py:")
    print(
        """
    # Replace line 518:
    # contexts = self._extract_contexts(documents)
    
    # With:
    contexts = result.get('contexts')
    if contexts is None:
        # Fall back to extracting from documents if contexts not provided
        contexts = self._extract_contexts(documents)
    """
    )


if __name__ == "__main__":
    main()
