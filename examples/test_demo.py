#!/usr/bin/env python3
"""
Simple test script to verify CRAG hallucination demo functionality.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crag_hallucination_demo import CRAGHallucinationDemo


def test_basic_functionality():
    """Test basic demo functionality."""
    print("Testing CRAG Hallucination Demo...")

    # Initialize demo
    demo = CRAGHallucinationDemo({"use_colors": False})

    # Test a simple query
    query = "What is the mortality rate of COVID-19?"
    print(f"\nTesting query: {query}")

    result = demo.demonstrate_correction(query)

    print("\nBasic RAG Response:")
    print(result["basic_answer"])

    print("\nCRAG Corrected Response:")
    print(result["crag_answer"])

    print(f"\nDetected {len(result['detections'])} hallucinations:")
    for i, detection in enumerate(result["detections"], 1):
        print(f"  {i}. {detection['type']}: {detection['explanation']}")

    print(f"\nProcessing times:")
    print(f"  Basic RAG: {result['basic_time']:.3f}s")
    print(f"  CRAG: {result['crag_time']:.3f}s")

    print("\n✅ Demo test completed successfully!")
    return True


if __name__ == "__main__":
    test_basic_functionality()
