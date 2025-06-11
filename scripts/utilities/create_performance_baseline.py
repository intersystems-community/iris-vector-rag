import sys
import logging
import os
import json
import time
from datetime import datetime

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.iris_connector import get_iris_connection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_performance_baseline():
    """Create initial performance baseline for the system"""
    logging.info("Creating performance baseline...")
    
    baseline = {
        "timestamp": datetime.now().isoformat(),
        "system_info": {},
        "database_info": {},
        "performance_metrics": {}
    }
    
    try:
        conn = get_iris_connection()
        
        with conn.cursor() as cursor:
            # Database version and configuration
            cursor.execute("SELECT $SYSTEM.Version.GetVersion() AS version")
            version_result = cursor.fetchone()
            baseline["database_info"]["iris_version"] = version_result[0] if version_result else "Unknown"
            
            # Check vector search capabilities
            try:
                cursor.execute("SELECT TO_VECTOR('[0.1, 0.2, 0.3]') AS test")
                baseline["database_info"]["vector_search_enabled"] = True
            except:
                baseline["database_info"]["vector_search_enabled"] = False
            
            # Schema information
            cursor.execute("SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'RAG'")
            baseline["database_info"]["rag_tables_count"] = cursor.fetchone()[0]
            
            # Initial data counts
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
            baseline["database_info"]["initial_document_count"] = cursor.fetchone()[0]
            
            # Test basic vector operations performance
            test_vector = "[" + ",".join(["0.1"] * 384) + "]"
            
            # Test TO_VECTOR performance
            start_time = time.time()
            cursor.execute("SELECT TO_VECTOR(?) AS test_vector", (test_vector,))
            cursor.fetchone()
            to_vector_time = (time.time() - start_time) * 1000
            
            baseline["performance_metrics"]["to_vector_time_ms"] = to_vector_time
            
            # Test vector similarity performance (if data exists)
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NOT NULL")
            embedded_count = cursor.fetchone()[0]
            
            if embedded_count > 0:
                start_time = time.time()
                cursor.execute(f"""
                    SELECT TOP 5 doc_id, VECTOR_COSINE(embedding, TO_VECTOR(?)) AS similarity
                    FROM RAG.SourceDocuments 
                    WHERE embedding IS NOT NULL
                    ORDER BY similarity DESC
                """, (test_vector,))
                cursor.fetchall()
                similarity_time = (time.time() - start_time) * 1000
                baseline["performance_metrics"]["vector_similarity_time_ms"] = similarity_time
            else:
                baseline["performance_metrics"]["vector_similarity_time_ms"] = None
            
        conn.close()
        
        # Save baseline to file
        baseline_file = "logs/performance_baseline.json"
        os.makedirs(os.path.dirname(baseline_file), exist_ok=True)
        
        with open(baseline_file, 'w') as f:
            json.dump(baseline, f, indent=2)
        
        logging.info(f"Performance baseline saved to {baseline_file}")
        logging.info(f"TO_VECTOR performance: {to_vector_time:.2f}ms")
        
        if baseline["performance_metrics"]["vector_similarity_time_ms"]:
            logging.info(f"Vector similarity performance: {baseline['performance_metrics']['vector_similarity_time_ms']:.2f}ms")
        
        return True
        
    except Exception as e:
        logging.error(f"Error creating performance baseline: {e}")
        return False

if __name__ == "__main__":
    success = create_performance_baseline()
    sys.exit(0 if success else 1)