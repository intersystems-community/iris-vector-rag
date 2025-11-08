"""
Basic example: IRIS vector search with retrieve-dspy.

Prerequisites:
1. IRIS database running with vector data
2. Environment variables set (IRIS_HOST, IRIS_PORT, etc.)
3. Table RAG.Documents with text_content and text_content_embedding columns

Setup:
    export IRIS_HOST="localhost"
    export IRIS_PORT="21972"
    export IRIS_NAMESPACE="USER"
    export IRIS_USERNAME="_SYSTEM"
    export IRIS_PASSWORD="SYS"

Usage:
    python basic_example.py
"""

import os
import sys

# Add retrieve_dspy to path if running from examples directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from retrieve_dspy.database.iris_database import iris_search_tool


def check_environment():
    """Verify required environment variables are set."""
    required_vars = ['IRIS_HOST', 'IRIS_PORT', 'IRIS_NAMESPACE',
                     'IRIS_USERNAME', 'IRIS_PASSWORD']

    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print("ERROR: Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nSet them with:")
        print("  export IRIS_HOST='localhost'")
        print("  export IRIS_PORT='21972'")
        print("  export IRIS_NAMESPACE='USER'")
        print("  export IRIS_USERNAME='_SYSTEM'")
        print("  export IRIS_PASSWORD='SYS'")
        return False

    return True


def main():
    """Run basic IRIS vector search example."""
    print("=" * 60)
    print("IRIS Vector Search with retrieve-dspy")
    print("=" * 60)
    print()

    # Check environment
    if not check_environment():
        sys.exit(1)

    # Display configuration
    print("Configuration:")
    print(f"  Host: {os.getenv('IRIS_HOST')}")
    print(f"  Port: {os.getenv('IRIS_PORT')}")
    print(f"  Namespace: {os.getenv('IRIS_NAMESPACE')}")
    print(f"  Username: {os.getenv('IRIS_USERNAME')}")
    print()

    # Execute search
    print("Executing search: 'What are the symptoms of diabetes?'")
    print()

    try:
        results = iris_search_tool(
            query="What are the symptoms of diabetes?",
            collection_name="RAG.Documents",
            target_property_name="text_content",
            retrieved_k=5,
            return_vector=False
        )

        # Display results
        print(f"Found {len(results)} results:")
        print()

        for obj in results:
            print(f"[{obj.relevance_rank}] Score: {obj.relevance_score:.4f}")
            print(f"    ID: {obj.object_id}")
            print(f"    Content: {obj.content[:100]}...")
            print()

        print("=" * 60)
        print("SUCCESS: IRIS adapter working correctly!")
        print("=" * 60)

    except Exception as e:
        print(f"ERROR: Search failed: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Verify IRIS is running")
        print("  2. Check table RAG.Documents exists")
        print("  3. Verify table has text_content and text_content_embedding columns")
        print("  4. Ensure sample data is loaded")
        sys.exit(1)


if __name__ == "__main__":
    main()
