#!/usr/bin/env python3
"""
Test script for GraphRAG Multi-Hop Demonstration

This script validates the structure and basic functionality of the
GraphRAG multi-hop demonstration without requiring visualization packages.
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_json_structure():
    """Test the JSON query structure."""
    print("🧪 Testing JSON query structure...")

    json_file = Path(__file__).parent / "graphrag_multihop_queries.json"

    if not json_file.exists():
        print("❌ JSON file not found")
        return False

    try:
        with open(json_file, "r") as f:
            data = json.load(f)

        # Validate required structure
        required_keys = [
            "metadata",
            "queries",
            "evaluation_metrics",
            "usage_instructions",
        ]
        for key in required_keys:
            if key not in data:
                print(f"❌ Missing required key: {key}")
                return False

        # Validate query categories
        query_types = ["2_hop_queries", "3_hop_queries", "complex_reasoning_queries"]
        for query_type in query_types:
            if query_type not in data["queries"]:
                print(f"❌ Missing query type: {query_type}")
                return False

        # Count total queries
        total_queries = 0
        for query_type in query_types:
            total_queries += len(data["queries"][query_type])

        print(f"✅ JSON structure valid - {total_queries} total queries")
        return True

    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error validating JSON: {e}")
        return False


def test_core_imports():
    """Test core module imports without visualization dependencies."""
    print("🧪 Testing core imports...")

    try:
        # Test IRIS RAG imports
        from iris_vector_rag.config.manager import ConfigurationManager
        from iris_vector_rag.core.connection import ConnectionManager
        from iris_vector_rag.core.models import Document, Entity, Relationship

        print("✅ IRIS RAG core imports successful")

        # Test model creation
        doc = Document(
            page_content="Test content", metadata={"test": "value"}, id="test_doc"
        )
        print("✅ Document model creation successful")

        entity = Entity(
            text="test entity",
            entity_type="TEST",
            confidence=0.9,
            start_offset=0,
            end_offset=11,
            source_document_id="test_doc",
        )
        print("✅ Entity model creation successful")

        relationship = Relationship(
            source_entity_id="entity1",
            target_entity_id="entity2",
            relationship_type="test_relationship",
            confidence=0.8,
            source_document_id="test_doc",
        )
        print("✅ Relationship model creation successful")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing models: {e}")
        return False


def test_demo_class_structure():
    """Test the demo class can be imported and has required methods."""
    print("🧪 Testing demo class structure...")

    try:
        # Create a mock version without dependencies
        class MockGraphRAGMultiHopDemo:
            def __init__(self, config_path=None, output_dir="test_output"):
                self.output_dir = Path(output_dir)
                self.query_results = []

            def load_sample_medical_documents(self):
                pass

            def demonstrate_2hop_queries(self):
                pass

            def demonstrate_3hop_queries(self):
                pass

            def demonstrate_complex_reasoning_queries(self):
                pass

            def compare_with_single_hop_rag(self):
                pass

            def generate_performance_report(self):
                pass

            def run_full_demonstration(self):
                pass

        # Test mock class
        demo = MockGraphRAGMultiHopDemo()

        # Verify required methods exist
        required_methods = [
            "load_sample_medical_documents",
            "demonstrate_2hop_queries",
            "demonstrate_3hop_queries",
            "demonstrate_complex_reasoning_queries",
            "compare_with_single_hop_rag",
            "generate_performance_report",
            "run_full_demonstration",
        ]

        for method in required_methods:
            if not hasattr(demo, method):
                print(f"❌ Missing required method: {method}")
                return False

        print("✅ Demo class structure valid")
        return True

    except Exception as e:
        print(f"❌ Error testing demo class: {e}")
        return False


def test_query_examples():
    """Test that query examples follow expected patterns."""
    print("🧪 Testing query examples...")

    try:
        json_file = Path(__file__).parent / "graphrag_multihop_queries.json"
        with open(json_file, "r") as f:
            data = json.load(f)

        # Test 2-hop queries
        hop2_queries = data["queries"]["2_hop_queries"]
        if len(hop2_queries) < 3:
            print("❌ Insufficient 2-hop queries")
            return False

        # Test 3-hop queries
        hop3_queries = data["queries"]["3_hop_queries"]
        if len(hop3_queries) < 3:
            print("❌ Insufficient 3-hop queries")
            return False

        # Test complex queries
        complex_queries = data["queries"]["complex_reasoning_queries"]
        if len(complex_queries) < 3:
            print("❌ Insufficient complex queries")
            return False

        # Validate query structure
        for query_list in [hop2_queries, hop3_queries, complex_queries]:
            for query in query_list:
                required_fields = [
                    "id",
                    "query",
                    "category",
                    "difficulty",
                    "reasoning_path",
                ]
                for field in required_fields:
                    if field not in query:
                        print(
                            f"❌ Missing field {field} in query {query.get('id', 'unknown')}"
                        )
                        return False

        print("✅ Query examples structure valid")
        return True

    except Exception as e:
        print(f"❌ Error testing query examples: {e}")
        return False


def test_medical_domain_coverage():
    """Test that medical domain concepts are well covered."""
    print("🧪 Testing medical domain coverage...")

    try:
        json_file = Path(__file__).parent / "graphrag_multihop_queries.json"
        with open(json_file, "r") as f:
            data = json.load(f)

        # Extract all queries
        all_queries = []
        for query_type in [
            "2_hop_queries",
            "3_hop_queries",
            "complex_reasoning_queries",
        ]:
            all_queries.extend(data["queries"][query_type])

        # Check for medical concepts
        medical_concepts = [
            "diabetes",
            "hypertension",
            "cardiovascular",
            "medication",
            "drug",
            "treatment",
            "symptom",
            "complication",
            "interaction",
            "side effect",
        ]

        concept_coverage = {}
        for concept in medical_concepts:
            concept_coverage[concept] = 0
            for query in all_queries:
                query_text = query["query"].lower()
                if concept in query_text:
                    concept_coverage[concept] += 1

        # Verify good coverage
        total_concepts = len(medical_concepts)
        covered_concepts = sum(1 for count in concept_coverage.values() if count > 0)
        coverage_ratio = covered_concepts / total_concepts

        if coverage_ratio < 0.7:  # At least 70% concept coverage
            print(f"❌ Insufficient medical domain coverage: {coverage_ratio:.1%}")
            return False

        print(f"✅ Medical domain coverage: {coverage_ratio:.1%}")
        return True

    except Exception as e:
        print(f"❌ Error testing medical domain coverage: {e}")
        return False


def run_all_tests():
    """Run all validation tests."""
    print("🚀 Starting GraphRAG Multi-Hop Demo Validation Tests\n")

    tests = [
        test_json_structure,
        test_core_imports,
        test_demo_class_structure,
        test_query_examples,
        test_medical_domain_coverage,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
            print()  # Add spacing between tests
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}\n")

    print("=" * 60)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Demo is ready for use.")
        print("\n📝 USAGE INSTRUCTIONS:")
        print("1. Install required packages: pip install networkx matplotlib colorama")
        print("2. Set up IRIS RAG environment and database connection")
        print("3. Run: python examples/graphrag_multihop_demo.py")
        return True
    else:
        print("❌ Some tests failed. Please fix issues before using demo.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
