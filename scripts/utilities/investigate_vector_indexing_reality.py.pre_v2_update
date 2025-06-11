#!/usr/bin/env python3
"""
Vector Indexing Reality Investigation Script

This script investigates the actual implementation of vector indexing in our IRIS system
to understand the truth about HNSW performance vs. what's actually implemented.

Objectives:
1. Test actual vector search performance with different dataset sizes
2. Verify if HNSW indexing is actually being used
3. Compare performance with and without claimed "HNSW" optimization
4. Document the real vector architecture vs. claimed architecture
"""

import time
import json
import logging
import numpy as np
from typing import List, Dict, Any, Tuple
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.iris_connector import get_iris_connection
from common.db_vector_search import search_source_documents_dynamically
from common.embedding_utils import get_embedding_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def investigate_database_schema(iris_connector) -> Dict[str, Any]:
    """Investigate the actual database schema to understand vector storage."""
    logger.info("🔍 Investigating actual database schema...")
    
    schema_info = {
        "tables": {},
        "indexes": {},
        "vector_columns": {},
        "actual_storage_types": {}
    }
    
    cursor = iris_connector.cursor()
    
    try:
        # Check what tables actually exist
        cursor.execute("""
            SELECT TABLE_SCHEMA, TABLE_NAME, TABLE_TYPE 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_SCHEMA IN ('RAG', 'RAG_HNSW', 'RAG_CHUNKS')
            ORDER BY TABLE_SCHEMA, TABLE_NAME
        """)
        
        tables = cursor.fetchall()
        for table in tables:
            schema_name, table_name, table_type = table
            full_name = f"{schema_name}.{table_name}"
            schema_info["tables"][full_name] = {
                "schema": schema_name,
                "name": table_name,
                "type": table_type
            }
            
        # Check column definitions for vector-related columns
        for table_name in schema_info["tables"]:
            try:
                cursor.execute(f"""
                    SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, IS_NULLABLE
                    FROM INFORMATION_SCHEMA.COLUMNS 
                    WHERE TABLE_NAME = '{table_name.split('.')[1]}' 
                    AND TABLE_SCHEMA = '{table_name.split('.')[0]}'
                    AND (COLUMN_NAME LIKE '%embedding%' OR COLUMN_NAME LIKE '%vector%')
                    ORDER BY ORDINAL_POSITION
                """)
                
                columns = cursor.fetchall()
                if columns:
                    schema_info["vector_columns"][table_name] = []
                    for col in columns:
                        col_name, data_type, max_length, nullable = col
                        schema_info["vector_columns"][table_name].append({
                            "name": col_name,
                            "data_type": data_type,
                            "max_length": max_length,
                            "nullable": nullable
                        })
                        
            except Exception as e:
                logger.warning(f"Could not get column info for {table_name}: {e}")
        
        # Check for indexes on vector columns
        try:
            cursor.execute("""
                SELECT 
                    i.INDEX_SCHEMA,
                    i.INDEX_NAME,
                    i.TABLE_NAME,
                    i.COLUMN_NAME,
                    i.INDEX_TYPE
                FROM INFORMATION_SCHEMA.STATISTICS i
                WHERE i.TABLE_SCHEMA IN ('RAG', 'RAG_HNSW', 'RAG_CHUNKS')
                AND (i.COLUMN_NAME LIKE '%embedding%' OR i.COLUMN_NAME LIKE '%vector%')
                ORDER BY i.INDEX_SCHEMA, i.TABLE_NAME, i.INDEX_NAME
            """)
            
            indexes = cursor.fetchall()
            for idx in indexes:
                schema, idx_name, table, column, idx_type = idx
                key = f"{schema}.{table}.{column}"
                schema_info["indexes"][key] = {
                    "index_name": idx_name,
                    "index_type": idx_type,
                    "table": f"{schema}.{table}",
                    "column": column
                }
                
        except Exception as e:
            logger.warning(f"Could not get index information: {e}")
            
    except Exception as e:
        logger.error(f"Error investigating schema: {e}")
    finally:
        cursor.close()
    
    return schema_info

def test_vector_search_performance(iris_connector, test_sizes: List[int]) -> Dict[str, Any]:
    """Test vector search performance with different approaches."""
    logger.info("⚡ Testing vector search performance...")
    
    # Get embedding function
    embed_func = get_embedding_model(mock=True)
    
    # Create test query
    test_query = "What are the effects of COVID-19 on cardiovascular health?"
    query_embedding = embed_func(test_query)
    query_vector_str = f"[{','.join(map(str, query_embedding))}]"
    
    performance_results = {
        "test_query": test_query,
        "query_vector_dimension": len(query_embedding),
        "tests": {}
    }
    
    for test_size in test_sizes:
        logger.info(f"Testing with top_k={test_size}")
        
        # Test multiple runs to get average performance
        times = []
        results_count = []
        
        for run in range(3):  # 3 runs for averaging
            start_time = time.time()
            
            try:
                results = search_source_documents_dynamically(
                    iris_connector=iris_connector,
                    top_k=test_size,
                    vector_string=query_vector_str
                )
                
                end_time = time.time()
                query_time = end_time - start_time
                
                times.append(query_time)
                results_count.append(len(results))
                
                logger.info(f"  Run {run+1}: {query_time:.4f}s, {len(results)} results")
                
            except Exception as e:
                logger.error(f"  Run {run+1} failed: {e}")
                times.append(float('inf'))
                results_count.append(0)
        
        # Calculate statistics
        valid_times = [t for t in times if t != float('inf')]
        if valid_times:
            avg_time = sum(valid_times) / len(valid_times)
            min_time = min(valid_times)
            max_time = max(valid_times)
        else:
            avg_time = min_time = max_time = float('inf')
            
        performance_results["tests"][f"top_k_{test_size}"] = {
            "top_k": test_size,
            "avg_time_seconds": avg_time,
            "min_time_seconds": min_time,
            "max_time_seconds": max_time,
            "avg_results_count": sum(results_count) / len(results_count) if results_count else 0,
            "success_rate": len(valid_times) / len(times),
            "raw_times": times,
            "raw_results_counts": results_count
        }
    
    return performance_results

def analyze_vector_storage_reality(iris_connector) -> Dict[str, Any]:
    """Analyze how vectors are actually stored and retrieved."""
    logger.info("🔬 Analyzing vector storage reality...")
    
    cursor = iris_connector.cursor()
    storage_analysis = {
        "sample_embeddings": [],
        "storage_format": "unknown",
        "actual_vector_operations": "unknown",
        "document_count": 0
    }
    
    try:
        # Get document count
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NOT NULL")
        doc_count = cursor.fetchone()[0]
        storage_analysis["document_count"] = doc_count
        
        # Sample some embeddings to understand storage format
        cursor.execute("""
            SELECT TOP 3 doc_id, 
                   SUBSTRING(embedding, 1, 100) as embedding_sample,
                   LENGTH(embedding) as embedding_length
            FROM RAG.SourceDocuments 
            WHERE embedding IS NOT NULL AND embedding <> ''
        """)
        
        samples = cursor.fetchall()
        for sample in samples:
            doc_id, embedding_sample, embedding_length = sample
            storage_analysis["sample_embeddings"].append({
                "doc_id": doc_id,
                "sample": embedding_sample,
                "total_length": embedding_length
            })
        
        # Determine storage format
        if samples and samples[0][1]:
            sample_text = samples[0][1]
            if sample_text.startswith('[') and ',' in sample_text:
                storage_analysis["storage_format"] = "comma_separated_array"
            elif sample_text.replace('.', '').replace(',', '').replace('-', '').isdigit():
                storage_analysis["storage_format"] = "numeric_string"
            else:
                storage_analysis["storage_format"] = "unknown_format"
        
        # Test if VECTOR_COSINE actually works
        try:
            test_vector = "[0.1,0.2,0.3]" + ",0.0" * 765  # 768-dimensional test vector
            cursor.execute(f"""
                SELECT TOP 1 doc_id,
                       VECTOR_COSINE(
                           TO_VECTOR(embedding, 'DOUBLE', 768),
                           TO_VECTOR('{test_vector}', 'DOUBLE', 768)
                       ) AS score
                FROM RAG.SourceDocuments
                WHERE embedding IS NOT NULL AND embedding <> ''
            """)
            
            result = cursor.fetchone()
            if result:
                storage_analysis["actual_vector_operations"] = "VECTOR_COSINE_working"
                storage_analysis["test_score"] = float(result[1])
            else:
                storage_analysis["actual_vector_operations"] = "VECTOR_COSINE_no_results"
                
        except Exception as e:
            storage_analysis["actual_vector_operations"] = f"VECTOR_COSINE_failed: {str(e)}"
            
    except Exception as e:
        logger.error(f"Error analyzing vector storage: {e}")
        storage_analysis["error"] = str(e)
    finally:
        cursor.close()
    
    return storage_analysis

def check_hnsw_index_reality(iris_connector) -> Dict[str, Any]:
    """Check if HNSW indexes actually exist and are being used."""
    logger.info("🏗️ Checking HNSW index reality...")
    
    cursor = iris_connector.cursor()
    hnsw_analysis = {
        "hnsw_indexes_found": [],
        "vector_type_columns": [],
        "index_usage_evidence": "none",
        "performance_characteristics": "unknown"
    }
    
    try:
        # Look for HNSW indexes specifically
        cursor.execute("""
            SELECT 
                INDEX_SCHEMA,
                INDEX_NAME,
                TABLE_NAME,
                COLUMN_NAME,
                INDEX_TYPE
            FROM INFORMATION_SCHEMA.STATISTICS
            WHERE INDEX_TYPE LIKE '%HNSW%' OR INDEX_NAME LIKE '%HNSW%'
            OR INDEX_TYPE LIKE '%VECTOR%'
        """)
        
        hnsw_indexes = cursor.fetchall()
        for idx in hnsw_indexes:
            hnsw_analysis["hnsw_indexes_found"].append({
                "schema": idx[0],
                "index_name": idx[1],
                "table": idx[2],
                "column": idx[3],
                "type": idx[4]
            })
        
        # Look for VECTOR type columns
        cursor.execute("""
            SELECT 
                TABLE_SCHEMA,
                TABLE_NAME,
                COLUMN_NAME,
                DATA_TYPE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE DATA_TYPE LIKE '%VECTOR%'
        """)
        
        vector_columns = cursor.fetchall()
        for col in vector_columns:
            hnsw_analysis["vector_type_columns"].append({
                "schema": col[0],
                "table": col[1],
                "column": col[2],
                "data_type": col[3]
            })
        
        # Test performance characteristics to infer index usage
        # If HNSW is working, performance should scale logarithmically, not linearly
        test_sizes = [5, 10, 50, 100]
        performance_scaling = []
        
        embed_func = get_embedding_model(mock=True)
        test_query = "test query for performance scaling"
        query_embedding = embed_func(test_query)
        query_vector_str = f"[{','.join(map(str, query_embedding))}]"
        
        for size in test_sizes:
            start_time = time.time()
            try:
                results = search_source_documents_dynamically(
                    iris_connector=iris_connector,
                    top_k=size,
                    vector_string=query_vector_str
                )
                end_time = time.time()
                query_time = end_time - start_time
                performance_scaling.append({
                    "top_k": size,
                    "time": query_time,
                    "results_count": len(results)
                })
            except Exception as e:
                performance_scaling.append({
                    "top_k": size,
                    "time": float('inf'),
                    "error": str(e)
                })
        
        hnsw_analysis["performance_scaling"] = performance_scaling
        
        # Analyze scaling pattern
        valid_times = [p["time"] for p in performance_scaling if p["time"] != float('inf')]
        if len(valid_times) >= 2:
            # If times are roughly constant, likely using an index
            # If times scale linearly with top_k, likely brute force
            time_ratios = []
            for i in range(1, len(valid_times)):
                if valid_times[i-1] > 0:
                    ratio = valid_times[i] / valid_times[i-1]
                    time_ratios.append(ratio)
            
            if time_ratios:
                avg_ratio = sum(time_ratios) / len(time_ratios)
                if avg_ratio < 1.5:  # Times don't scale much with size
                    hnsw_analysis["index_usage_evidence"] = "likely_indexed"
                elif avg_ratio > 2.0:  # Times scale significantly
                    hnsw_analysis["index_usage_evidence"] = "likely_brute_force"
                else:
                    hnsw_analysis["index_usage_evidence"] = "unclear"
                
                hnsw_analysis["performance_characteristics"] = {
                    "avg_scaling_ratio": avg_ratio,
                    "scaling_ratios": time_ratios
                }
        
    except Exception as e:
        logger.error(f"Error checking HNSW reality: {e}")
        hnsw_analysis["error"] = str(e)
    finally:
        cursor.close()
    
    return hnsw_analysis

def main():
    """Main investigation function."""
    logger.info("🚀 Starting Vector Indexing Reality Investigation")
    
    investigation_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "investigation_type": "vector_indexing_reality_check",
        "findings": {}
    }
    
    try:
        # Get IRIS connection
        iris_connector = get_iris_connection()
        logger.info("✅ Connected to IRIS database")
        
        # 1. Investigate database schema
        logger.info("\n" + "="*60)
        schema_info = investigate_database_schema(iris_connector)
        investigation_results["findings"]["database_schema"] = schema_info
        
        # 2. Analyze vector storage reality
        logger.info("\n" + "="*60)
        storage_analysis = analyze_vector_storage_reality(iris_connector)
        investigation_results["findings"]["vector_storage"] = storage_analysis
        
        # 3. Check HNSW index reality
        logger.info("\n" + "="*60)
        hnsw_analysis = check_hnsw_index_reality(iris_connector)
        investigation_results["findings"]["hnsw_indexing"] = hnsw_analysis
        
        # 4. Test vector search performance
        logger.info("\n" + "="*60)
        test_sizes = [5, 10, 20, 50, 100]
        performance_results = test_vector_search_performance(iris_connector, test_sizes)
        investigation_results["findings"]["performance_testing"] = performance_results
        
        # Close connection
        iris_connector.close()
        
    except Exception as e:
        logger.error(f"Investigation failed: {e}")
        investigation_results["error"] = str(e)
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"vector_indexing_investigation_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(investigation_results, f, indent=2, default=str)
    
    logger.info(f"\n📊 Investigation complete! Results saved to: {results_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("🔍 VECTOR INDEXING REALITY INVESTIGATION SUMMARY")
    print("="*80)
    
    if "database_schema" in investigation_results["findings"]:
        schema = investigation_results["findings"]["database_schema"]
        print(f"\n📋 Database Schema:")
        print(f"   Tables found: {len(schema['tables'])}")
        print(f"   Vector columns: {len(schema['vector_columns'])}")
        print(f"   Vector indexes: {len(schema['indexes'])}")
        
        for table, columns in schema["vector_columns"].items():
            for col in columns:
                print(f"   - {table}.{col['name']}: {col['data_type']} ({col['max_length']} chars)")
    
    if "vector_storage" in investigation_results["findings"]:
        storage = investigation_results["findings"]["vector_storage"]
        print(f"\n💾 Vector Storage:")
        print(f"   Document count: {storage['document_count']}")
        print(f"   Storage format: {storage['storage_format']}")
        print(f"   Vector operations: {storage['actual_vector_operations']}")
    
    if "hnsw_indexing" in investigation_results["findings"]:
        hnsw = investigation_results["findings"]["hnsw_indexing"]
        print(f"\n🏗️ HNSW Indexing:")
        print(f"   HNSW indexes found: {len(hnsw['hnsw_indexes_found'])}")
        print(f"   VECTOR type columns: {len(hnsw['vector_type_columns'])}")
        print(f"   Index usage evidence: {hnsw['index_usage_evidence']}")
        
        if "performance_characteristics" in hnsw and isinstance(hnsw["performance_characteristics"], dict):
            perf = hnsw["performance_characteristics"]
            print(f"   Performance scaling: {perf.get('avg_scaling_ratio', 'unknown'):.2f}x average")
    
    if "performance_testing" in investigation_results["findings"]:
        perf = investigation_results["findings"]["performance_testing"]
        print(f"\n⚡ Performance Testing:")
        print(f"   Query dimension: {perf['query_vector_dimension']}")
        
        for test_name, test_data in perf["tests"].items():
            if test_data["success_rate"] > 0:
                print(f"   {test_name}: {test_data['avg_time_seconds']:.4f}s avg, "
                      f"{test_data['avg_results_count']:.1f} results")
    
    print("\n" + "="*80)
    print("🎯 CONCLUSION: Check the detailed JSON report for complete findings!")
    print("="*80)

if __name__ == "__main__":
    main()