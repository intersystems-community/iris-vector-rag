import logging
import time
import sys
import os
import re
import io
from typing import Optional
import contextlib # For redirecting stdout

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Assuming scripts is in project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.deprecated.basic_rag.pipeline import BasicRAGPipeline # Updated import
from src.experimental.hybrid_ifind_rag.pipeline import HybridiFindRAGPipeline # Updated import
from src.common.utils import get_embedding_func, get_llm_func # Updated import
from src.common.iris_connector_jdbc import get_iris_connection # Updated import

# Configure logging to capture specific messages for timing
log_capture_string_io = io.StringIO()
# Get root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO) # Set root logger level

# Remove existing handlers to avoid duplicate outputs if script is re-run
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Handler for stdout
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.WARNING) # Only show warnings and above on stdout
logger.addHandler(stdout_handler)

# Handler for capturing specific logs for timing
string_io_handler = logging.StreamHandler(log_capture_string_io)
string_io_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
string_io_handler.setFormatter(formatter)
logging.getLogger('hybrid_ifind_rag.pipeline').addHandler(string_io_handler)
logging.getLogger('hybrid_ifind_rag.pipeline').propagate = False # Prevent duplication to root logger for these specific logs


# Regex to extract vector_time from HybridIFindRAG logs
HYBRID_VECTOR_TIME_REGEX = re.compile(r"Vector: \d+ results in (\d+\.\d+)s")
# Regex to extract BasicRAG retrieve_documents time from stdout
BASIC_RETRIEVE_DOCS_TIME_REGEX = re.compile(r"Function retrieve_documents executed in (\d+\.\d+) seconds")

def extract_hybrid_vector_search_time(log_output: str) -> Optional[float]:
    matches = HYBRID_VECTOR_TIME_REGEX.findall(log_output)
    if matches:
        return float(matches[-1]) # Get the last match, assuming it's the relevant one
    return None

def extract_basic_retrieve_docs_time(stdout_output: str) -> Optional[float]:
    match = BASIC_RETRIEVE_DOCS_TIME_REGEX.search(stdout_output)
    if match:
        return float(match.group(1))
    return None

def run_quick_performance_test():
    print("Starting Quick Performance Test...")

    queries = [
        "What is diabetes?",
        "How do neurons work?",
        "Tell me about machine learning."
    ]

    results_summary = []

    try:
        db_conn = get_iris_connection()
        embed_fn = get_embedding_func()
        # Use a stub LLM to focus on retrieval performance
        llm_fn_stub = get_llm_func(provider="stub")

        # Initialize BasicRAG Pipeline
        basic_rag_pipeline = BasicRAGPipeline(
            iris_connector=db_conn,
            embedding_func=embed_fn,
            llm_func=llm_fn_stub
        )

        # Initialize HybridIFindRAG Pipeline
        hybrid_ifind_rag_pipeline = HybridiFindRAGPipeline(
            iris_connector=db_conn,
            embedding_func=embed_fn,
            llm_func=llm_fn_stub
        )
        # Ensure hybrid pipeline logger is set up to capture its specific logs
        hybrid_pipeline_logger = logging.getLogger('hybrid_ifind_rag.pipeline')
        hybrid_pipeline_logger.handlers.clear() # Clear any previous handlers if re-running
        hybrid_pipeline_logger.addHandler(string_io_handler)
        hybrid_pipeline_logger.setLevel(logging.INFO)
        hybrid_pipeline_logger.propagate = False


        for query_text in queries:
            print(f"\n--- Testing Query: '{query_text}' ---")
            query_results = {"query": query_text}

            # --- BasicRAG Test ---
            print("Testing BasicRAG...")
            # Get total time and document count
            basic_run_result = basic_rag_pipeline.run(query_text, top_k=5)
            basic_total_time_ms = basic_run_result.get('latency_ms', 0)
            basic_doc_count = basic_run_result.get('document_count', 0)

            # Get vector search time
            # The retrieve_documents method is timed by @timing_decorator
            # We need to access the 'latency_ms' from its execution.
            # To do this cleanly, we can inspect the __wrapped__ method if it stores results,
            # or re-run it if it's cheap enough. Given it's a DB call, let's assume
            # the timing decorator stores it on the result or we can call it.
            # The `run` method calls `retrieve_documents` internally.
            # The `timing_decorator` prints the execution time of `retrieve_documents` to stdout.
            # Capture stdout to get this timing.
            stdout_capture = io.StringIO()
            with contextlib.redirect_stdout(stdout_capture):
                _ = basic_rag_pipeline.retrieve_documents(query_text, top_k=5) # Call the timed function
            
            stdout_output = stdout_capture.getvalue()
            basic_vector_search_time_s = extract_basic_retrieve_docs_time(stdout_output)
            
            if basic_vector_search_time_s is None:
                print("  Warning: Could not extract BasicRAG retrieve_documents time from stdout.")
                basic_vector_search_time_ms = 0.0
            else:
                basic_vector_search_time_ms = basic_vector_search_time_s * 1000
            
            query_results["basic_rag"] = {
                "total_time_ms": basic_total_time_ms,
                "vector_search_time_ms": basic_vector_search_time_ms,
                "docs_retrieved": basic_doc_count
            }
            print(f"  BasicRAG: Total Time: {basic_total_time_ms:.2f} ms, Vector Search: {basic_vector_search_time_ms:.2f} ms, Docs: {basic_doc_count}")

            # --- HybridIFindRAG Test ---
            print("Testing HybridIFindRAG...")
            log_capture_string_io.truncate(0) # Clear previous logs
            log_capture_string_io.seek(0)

            # Time the call to hybrid_ifind_rag_pipeline.query() externally
            hybrid_external_start_time = time.perf_counter()
            hybrid_run_result = hybrid_ifind_rag_pipeline.query(query_text)
            hybrid_external_total_time_s = time.perf_counter() - hybrid_external_start_time
            
            # Still attempt to get internal time for comparison, but use external for summary
            internal_hybrid_total_s = hybrid_run_result.get("metadata", {}).get("timings", {}).get("total_time_seconds", 0)
            hybrid_doc_count = len(hybrid_run_result.get("retrieved_documents", []))
            
            # Extract vector search time from logs
            log_content = log_capture_string_io.getvalue()
            hybrid_vector_search_time_s = extract_hybrid_vector_search_time(log_content)
            if hybrid_vector_search_time_s is None:
                print("  Warning: Could not extract HybridIFindRAG vector search time from logs.")
                hybrid_vector_search_time_s = 0 # Default if not found

            query_results["hybrid_ifind_rag"] = {
                "total_time_ms": hybrid_external_total_time_s * 1000, # Use externally measured time
                "vector_search_time_ms": hybrid_vector_search_time_s * 1000 if hybrid_vector_search_time_s else 0.0,
                "docs_retrieved": hybrid_doc_count
            }
            print(f"  HybridIFindRAG: Total Time (script timed): {hybrid_external_total_time_s*1000:.2f} ms, Vector Search: {(hybrid_vector_search_time_s*1000 if hybrid_vector_search_time_s else 0.0):.2f} ms, Docs: {hybrid_doc_count}")
            print(f"    (Internal total_time_seconds from pipeline: {internal_hybrid_total_s:.3f}s)")
            
            results_summary.append(query_results)

    except Exception as e:
        print(f"An error occurred during the test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'db_conn' in locals() and db_conn:
            db_conn.close()
            print("\nDatabase connection closed.")

    print("\n\n--- Quick Performance Test Summary ---")
    for res in results_summary:
        print(f"\nQuery: {res['query']}")
        br = res['basic_rag']
        hr = res['hybrid_ifind_rag']
        print(f"  BasicRAG:      Total: {br['total_time_ms']:.2f}ms, Vector Search: {br['vector_search_time_ms']:.2f}ms, Docs: {br['docs_retrieved']}")
        print(f"  HybridIFindRAG: Total: {hr['total_time_ms']:.2f}ms, Vector Search: {hr['vector_search_time_ms']:.2f}ms, Docs: {hr['docs_retrieved']}")

        # Validation checks
        if br['docs_retrieved'] > 0:
            print("    ✅ BasicRAG: Retrieved documents (>0)")
        else:
            print("    ❌ BasicRAG: Did NOT retrieve documents (should be >0)")

        if hr['vector_search_time_ms'] / 1000 < 8.0:
            print(f"    ✅ HybridIFindRAG: Vector search is fast ({hr['vector_search_time_ms']/1000:.3f}s < 8s)")
        else:
            print(f"    ❌ HybridIFindRAG: Vector search is SLOW ({hr['vector_search_time_ms']/1000:.3f}s >= 8s)")
        
        if hr['docs_retrieved'] > 0:
             print("    ✅ HybridIFindRAG: Retrieved documents (>0)")
        else:
            print("    ❌ HybridIFindRAG: Did NOT retrieve documents (should be >0)")


    print("\nQuick Performance Test Finished.")

if __name__ == "__main__":
    # Optional is now imported globally
    run_quick_performance_test()