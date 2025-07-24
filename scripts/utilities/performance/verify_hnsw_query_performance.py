#!/usr/bin/env python3
"""
Verify HNSW Index Query Performance

This script runs direct SQL queries to test the performance of HNSW-indexed
vector searches against the RAG.SourceDocuments table. It compares queries
targeting the HNSW-indexed column versus explicit TO_VECTOR conversion on
another embedding column.
"""

import logging
import time
import sys
import os
from typing import Dict, Any, Optional
import statistics

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from common.iris_connector import get_iris_connection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TABLE_NAME = "RAG.SourceDocuments"
INDEXED_VECTOR_COLUMN = "document_embedding_vector" # Has HNSW index, type VARCHAR
RAW_EMBEDDING_COLUMN = "embedding" # Type VARCHAR

def get_sample_query_vector(conn) -> Optional[str]:
    """Fetches a sample embedding string from the table to use as a query vector."""
    try:
        with conn.cursor() as cursor:
            # Preferentially get from the raw embedding column if it's populated
            sql = f"SELECT TOP 1 {RAW_EMBEDDING_COLUMN} FROM {TABLE_NAME} WHERE {RAW_EMBEDDING_COLUMN} IS NOT NULL AND {RAW_EMBEDDING_COLUMN} != ''"
            cursor.execute(sql)
            row = cursor.fetchone()
            if row and row[0]:
                logger.info(f"Using sample vector from '{RAW_EMBEDDING_COLUMN}'.")
                return row[0]

            # Fallback to the indexed column if raw is empty
            sql_fallback = f"SELECT TOP 1 {INDEXED_VECTOR_COLUMN} FROM {TABLE_NAME} WHERE {INDEXED_VECTOR_COLUMN} IS NOT NULL AND {INDEXED_VECTOR_COLUMN} != ''"
            logger.info(f"'{RAW_EMBEDDING_COLUMN}' is empty or not found, trying '{INDEXED_VECTOR_COLUMN}'.")
            cursor.execute(sql_fallback)
            row_fallback = cursor.fetchone()
            if row_fallback and row_fallback[0]:
                logger.info(f"Using sample vector from '{INDEXED_VECTOR_COLUMN}'.")
                return row_fallback[0]
            
            logger.error("Could not fetch a sample query vector from the database.")
            return None
    except Exception as e:
        logger.error(f"Error fetching sample query vector: {e}")
        return None

def execute_query_and_time(conn, query_name: str, sql: str, params: tuple, num_runs: int = 5) -> Dict[str, Any]:
    """Executes a given SQL query multiple times and returns timing statistics."""
    timings_ms = []
    results_count = 0
    
    logger.info(f"Executing query '{query_name}' ({num_runs} runs)...")
    
    for i in range(num_runs):
        try:
            with conn.cursor() as cursor:
                start_time = time.perf_counter()
                cursor.execute(sql, params)
                results = cursor.fetchall()
                end_time = time.perf_counter()
                
                timings_ms.append((end_time - start_time) * 1000)
                if i == 0: # Get results count from the first run
                    results_count = len(results)
                time.sleep(0.1) # Small delay between runs
        except Exception as e:
            logger.error(f"Error executing query '{query_name}' on run {i+1}: {e}")
            timings_ms.append(float('inf')) # Indicate failure

    if not timings_ms:
        return {"avg_time_ms": float('inf'), "min_time_ms": float('inf'), "max_time_ms": float('inf'), "std_dev_ms": 0, "results_count": 0, "runs": num_runs, "successful_runs": 0}

    successful_timings = [t for t in timings_ms if t != float('inf')]
    
    return {
        "avg_time_ms": statistics.mean(successful_timings) if successful_timings else float('inf'),
        "min_time_ms": min(successful_timings) if successful_timings else float('inf'),
        "max_time_ms": max(successful_timings) if successful_timings else float('inf'),
        "std_dev_ms": statistics.stdev(successful_timings) if len(successful_timings) > 1 else 0,
        "results_count": results_count,
        "runs": num_runs,
        "successful_runs": len(successful_timings)
    }

def get_query_plan(conn, sql: str, params: tuple) -> Optional[str]:
    """Attempts to get the query plan. Note: May require specific permissions or syntax."""
    try:
        with conn.cursor() as cursor:
            # Common way to get plan, might need adjustment for IRIS exact syntax / permissions
            # For IRIS, often done via Management Portal or specific system procs
            # This is a placeholder; direct EXPLAIN might not work via standard ODBC/JDBC for all DBs
            # or might return a format not easily parsable.
            # cursor.execute(f"EXPLAIN {sql}", params) # Example, likely needs IRIS specific syntax
            # plan = cursor.fetchall()
            # return "\n".join([str(row) for row in plan])
            logger.warning("Query plan retrieval is not fully implemented for IRIS in this script. Check Management Portal.")
            return "Query plan retrieval not implemented in script."
    except Exception as e:
        logger.error(f"Error getting query plan: {e}")
        return f"Error getting query plan: {e}"

def main():
    logger.info("🚀 Starting HNSW Query Performance Verification")
    
    conn = None
    try:
        conn = get_iris_connection()
        if not conn:
            logger.error("❌ Failed to connect to the database.")
            return

        query_vector_str = get_sample_query_vector(conn)
        if not query_vector_str:
            return

        logger.info(f"Sample query vector (first 50 chars): {query_vector_str[:50]}...")

        top_k_values = [5, 10, 20]
        num_test_runs = 5 # Number of times to run each query for averaging

        all_results = []

        # --- Test Query 1: Using HNSW-indexed column (document_embedding_vector) ---
        # This column is VARCHAR but has HNSW index. VECTOR_COSINE will do implicit conversion.
        query1_sql_template = f"""
            SELECT TOP ? doc_id, VECTOR_COSINE(TO_VECTOR({INDEXED_VECTOR_COLUMN}), TO_VECTOR(?)) AS similarity
            FROM {TABLE_NAME}
            WHERE {INDEXED_VECTOR_COLUMN} IS NOT NULL AND {INDEXED_VECTOR_COLUMN} != ''
            ORDER BY similarity DESC
        """
        logger.info(f"\n--- Testing Query on HNSW-indexed '{INDEXED_VECTOR_COLUMN}' (VARCHAR) ---")
        for top_k in top_k_values:
            params = (top_k, query_vector_str)
            stats = execute_query_and_time(conn, f"HNSW Indexed (Top {top_k})", query1_sql_template, params, num_test_runs)
            stats["query_type"] = "HNSW Indexed Column"
            stats["top_k"] = top_k
            all_results.append(stats)
            # plan = get_query_plan(conn, query1_sql_template, params)
            # logger.info(f"Query Plan for HNSW Indexed (Top {top_k}):\n{plan}")


        # --- Test Query 2: Using raw embedding column with explicit TO_VECTOR ---
        # This column (embedding) is VARCHAR and may or may not have a suitable index for this operation.
        query2_sql_template = f"""
            SELECT TOP ? doc_id, VECTOR_COSINE(TO_VECTOR({RAW_EMBEDDING_COLUMN}), TO_VECTOR(?)) AS similarity
            FROM {TABLE_NAME}
            WHERE {RAW_EMBEDDING_COLUMN} IS NOT NULL AND {RAW_EMBEDDING_COLUMN} != ''
            ORDER BY similarity DESC
        """
        logger.info(f"\n--- Testing Query on '{RAW_EMBEDDING_COLUMN}' (VARCHAR) with explicit TO_VECTOR ---")
        for top_k in top_k_values:
            params = (top_k, query_vector_str)
            stats = execute_query_and_time(conn, f"Explicit TO_VECTOR (Top {top_k})", query2_sql_template, params, num_test_runs)
            stats["query_type"] = "Explicit TO_VECTOR on Raw Column"
            stats["top_k"] = top_k
            all_results.append(stats)
            # plan = get_query_plan(conn, query2_sql_template, params)
            # logger.info(f"Query Plan for Explicit TO_VECTOR (Top {top_k}):\n{plan}")

        # --- Print Summary ---
        logger.info("\n\n--- HNSW Query Performance Summary ---")
        logger.info(f"{'Query Type':<35} | {'Top K':>5} | {'Avg Time (ms)':>15} | {'Min Time (ms)':>15} | {'Max Time (ms)':>15} | {'StdDev (ms)':>12} | {'Docs':>5} | {'Runs':>5}")
        logger.info("-" * 130)
        for res in all_results:
            logger.info(
                f"{res['query_type']:<35} | {res['top_k']:>5} | "
                f"{res['avg_time_ms']:>15.2f} | {res['min_time_ms']:>15.2f} | {res['max_time_ms']:>15.2f} | "
                f"{res['std_dev_ms']:>12.2f} | {res['results_count']:>5} | {res['successful_runs']:>2}/{res['runs']:<2}"
            )
        
        logger.info("\n🎯 Verification Steps & Expectations:")
        logger.info("1. HNSW Index Status: Checked earlier - RAG.SourceDocuments.document_embedding_vector (VARCHAR) has HNSW indexes.")
        logger.info("2. Direct SQL Performance:")
        logger.info("   - 'HNSW Indexed Column' queries should be fast (ideally sub-100ms, check consistency).")
        logger.info("   - Compare with 'Explicit TO_VECTOR' queries. If HNSW is effective, the indexed path should be significantly faster.")
        logger.info("3. Previous Results Comparison:")
        logger.info("   - Aim for sub-100ms for HNSW path. HybridIFindRAG's 34-42ms might include other overheads or different query patterns.")
        logger.info("4. Query Plans: (Manual Check Recommended) Use IRIS Management Portal to verify HNSW index usage for the first query type.")
        logger.info("5. Index Effectiveness:")
        logger.info("   - Observe performance across different TOP K values. Should remain consistently fast.")
        logger.info("   - (Future extension: Test different similarity thresholds if query patterns allow direct filtering).")

    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed.")
    
    logger.info("\n✅ HNSW Query Performance Verification Script Finished.")

if __name__ == "__main__":
    main()