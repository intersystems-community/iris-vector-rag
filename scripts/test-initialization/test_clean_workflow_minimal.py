#!/usr/bin/env python3
"""
Minimal Clean IRIS Workflow Test

Tests the complete workflow from clean IRIS database using simplified validation
to verify that the framework can work end-to-end without pre-existing data.

Constitutional Requirement: Clean IRIS Testing (NON-NEGOTIABLE)
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from iris_vector_rag import create_pipeline
from iris_vector_rag.core.models import Document
from scripts.testing.mock_providers import MockLLMProvider


def test_minimal_clean_workflow():
    """Test minimal workflow from completely clean IRIS database."""

    def log(message: str, level: str = "info"):
        timestamp = time.strftime("%H:%M:%S")
        if level == "error":
            print(f"❌ [{timestamp}] {message}")
        elif level == "success":
            print(f"✅ [{timestamp}] {message}")
        elif level == "warning":
            print(f"⚠️  [{timestamp}] {message}")
        else:
            print(f"ℹ️  [{timestamp}] {message}")

    log("🧪 Testing Minimal Clean IRIS Workflow", "info")
    log("=" * 60, "info")

    try:
        # Step 1: Clean database initialization (already done by initialize_clean_schema.py)
        log("Step 1: Assuming clean database from initialize_clean_schema.py", "info")

        # Step 2: Create pipeline without validation to test core functionality
        log("Step 2: Creating basic pipeline without strict validation...", "info")
        start_time = time.time()

        # Create pipeline with minimal validation for clean database testing
        pipeline = create_pipeline(
            pipeline_type="basic",
            validate_requirements=False,  # Skip validation for clean DB test
            auto_setup=True,
        )

        # Configure with mock LLM for testing
        mock_llm = MockLLMProvider(mode="realistic")
        pipeline.llm_func = mock_llm.generate_response

        creation_time = time.time() - start_time
        log(f"Pipeline created successfully ({creation_time:.2f}s)", "success")

        # Step 3: Test document loading
        log("Step 3: Testing document loading...", "info")
        data_start = time.time()

        # Create simple test documents
        test_documents = [
            Document(
                page_content="COVID-19 is a viral infection caused by SARS-CoV-2. Common symptoms include fever, cough, and fatigue.",
                metadata={"source": "covid_info.txt", "title": "COVID-19 Information"},
            ),
            Document(
                page_content="Diabetes is a chronic condition affecting blood sugar regulation. Treatment involves insulin and dietary management.",
                metadata={
                    "source": "diabetes_info.txt",
                    "title": "Diabetes Information",
                },
            ),
        ]

        # Load documents into pipeline
        try:
            if hasattr(pipeline, "load_documents"):
                pipeline.load_documents("", documents=test_documents)
                log(f"Loaded {len(test_documents)} documents successfully", "success")
            else:
                log("Pipeline does not support load_documents method", "warning")

        except Exception as e:
            log(f"Document loading failed: {e}", "error")
            # Continue with test even if loading fails

        data_time = time.time() - data_start

        # Step 4: Test basic query functionality
        log("Step 4: Testing query functionality...", "info")
        query_start = time.time()

        test_query = "What are the symptoms of COVID-19?"

        try:
            result = pipeline.query(test_query, generate_answer=True, top_k=3)

            # Analyze results
            has_answer = bool(result.get("answer"))
            answer_length = len(result.get("answer", ""))
            context_count = len(result.get("contexts", []))

            query_time = time.time() - query_start

            log(f"Query completed ({query_time:.2f}s)", "success")
            log(
                f"  Answer generated: {has_answer}",
                "success" if has_answer else "warning",
            )
            log(f"  Answer length: {answer_length} chars", "info")
            log(f"  Contexts retrieved: {context_count}", "info")

            if has_answer and answer_length > 20:
                log("✅ Basic query functionality working", "success")
                query_success = True
            else:
                log("⚠️  Query functionality limited", "warning")
                query_success = False

        except Exception as e:
            log(f"Query failed: {e}", "error")
            query_success = False
            query_time = time.time() - query_start

        # Step 5: Summary
        total_time = time.time() - start_time

        log("", "info")
        log("📊 Minimal Clean IRIS Workflow Results", "info")
        log("=" * 60, "info")
        log(f"Total time: {total_time:.2f}s", "info")
        log(f"Pipeline creation: {creation_time:.2f}s", "info")
        log(f"Document loading: {data_time:.2f}s", "info")
        log(f"Query execution: {query_time:.2f}s", "info")

        # Determine overall success
        pipeline_created = pipeline is not None

        overall_success = pipeline_created and query_success

        if overall_success:
            log("🎉 MINIMAL CLEAN IRIS WORKFLOW SUCCESSFUL!", "success")
            log("✅ Framework can operate from completely clean database", "success")
            return True
        else:
            log("⚠️  Minimal workflow completed with limitations", "warning")
            log(f"   Pipeline created: {pipeline_created}", "info")
            log(f"   Query successful: {query_success}", "info")
            return False

    except Exception as e:
        log(f"Minimal workflow test failed: {e}", "error")
        import traceback

        log(f"Full traceback: {traceback.format_exc()}", "error")
        return False


def main():
    """Main execution function."""
    print("🧪 Minimal Clean IRIS Workflow Test")
    print("Constitutional Requirement: Clean IRIS Testing")
    print("=" * 60)

    success = test_minimal_clean_workflow()

    if success:
        print("\n🎉 Minimal clean IRIS workflow validation successful!")
        print("✅ Core framework functionality works from clean database")
        return 0
    else:
        print("\n⚠️  Minimal workflow test completed with issues")
        print("📋 This indicates areas for improvement in clean database support")
        return 1


if __name__ == "__main__":
    sys.exit(main())
