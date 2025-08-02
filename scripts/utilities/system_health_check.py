import sys
import logging
import os
import time
import psutil
import docker

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.iris_connector import get_iris_connection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_system_resources():
    """Check system resources"""
    logging.info("=== System Resources ===")
    
    # Memory
    memory = psutil.virtual_memory()
    logging.info(f"Memory: {memory.used / (1024**3):.1f}GB used / {memory.total / (1024**3):.1f}GB total ({memory.percent:.1f}%)")
    
    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    logging.info(f"CPU: {cpu_percent:.1f}% usage")
    
    # Disk
    disk = psutil.disk_usage('/')
    logging.info(f"Disk: {disk.used / (1024**3):.1f}GB used / {disk.total / (1024**3):.1f}GB total ({disk.percent:.1f}%)")
    
    return memory.percent < 90 and cpu_percent < 90 and disk.percent < 90

def check_docker_containers():
    """Check Docker container status"""
    logging.info("=== Docker Containers ===")
    
    try:
        client = docker.from_env()
        containers = client.containers.list(all=True)
        
        iris_container = None
        for container in containers:
            if 'iris' in container.name.lower():
                iris_container = container
                break
        
        if iris_container:
            logging.info(f"IRIS Container: {iris_container.name}")
            logging.info(f"Status: {iris_container.status}")
            
            if iris_container.status == 'running':
                # Get container stats
                stats = iris_container.stats(stream=False)
                memory_usage = stats['memory_stats']['usage'] / (1024**3)
                memory_limit = stats['memory_stats']['limit'] / (1024**3)
                logging.info(f"Container Memory: {memory_usage:.1f}GB / {memory_limit:.1f}GB")
                return True
            else:
                logging.error("IRIS container is not running!")
                return False
        else:
            logging.error("IRIS container not found!")
            return False
            
    except Exception as e:
        logging.error(f"Error checking Docker containers: {e}")
        return False

def check_database_connection():
    """Check database connectivity and basic operations"""
    logging.info("=== Database Connection ===")
    
    try:
        conn = get_iris_connection()
        
        with conn.cursor() as cursor:
            # Test basic connectivity
            cursor.execute("SELECT 1 AS test")
            result = cursor.fetchone()
            if result and result[0] == 1:
                logging.info("✅ Database connection successful")
            else:
                logging.error("❌ Database connection test failed")
                return False
            
            # Check schema
            cursor.execute("SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'RAG'")
            table_count = cursor.fetchone()[0]
            logging.info(f"✅ RAG schema has {table_count} tables")
            
            # Check if we can perform vector operations
            cursor.execute("SELECT TO_VECTOR('[0.1, 0.2, 0.3]') AS test_vector")
            vector_result = cursor.fetchone()
            if vector_result:
                logging.info("✅ Vector operations working")
            else:
                logging.error("❌ Vector operations failed")
                return False
            
        conn.close()
        return True
        
    except Exception as e:
        logging.error(f"❌ Database connection failed: {e}")
        return False

def check_data_ingestion_status():
    """Check current data ingestion status"""
    logging.info("=== Data Ingestion Status ===")
    
    try:
        conn = get_iris_connection()
        
        with conn.cursor() as cursor:
            # Check SourceDocuments count
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
            doc_count = cursor.fetchone()[0]
            logging.info(f"Documents in SourceDocuments: {doc_count:,}")
            
            # Check DocumentChunks count
            try:
                cursor.execute("SELECT COUNT(*) FROM RAG.DocumentChunks")
                chunk_count = cursor.fetchone()[0]
                logging.info(f"Chunks in DocumentChunks: {chunk_count:,}")
            except:
                logging.info("DocumentChunks table not available or empty")
                chunk_count = 0
            
            # Check for embeddings
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NOT NULL")
            embedded_count = cursor.fetchone()[0]
            logging.info(f"Documents with embeddings: {embedded_count:,}")
            
            if doc_count > 0:
                embedding_percentage = (embedded_count / doc_count) * 100
                logging.info(f"Embedding completion: {embedding_percentage:.1f}%")
            
        conn.close()
        return doc_count > 0
        
    except Exception as e:
        logging.error(f"Error checking data status: {e}")
        return False

def check_hnsw_performance():
    """Check HNSW index performance"""
    logging.info("=== HNSW Performance ===")
    
    try:
        conn = get_iris_connection()
        
        with conn.cursor() as cursor:
            # Check if we have enough data for performance testing
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NOT NULL")
            embedded_count = cursor.fetchone()[0]
            
            if embedded_count < 100:
                logging.info(f"Only {embedded_count} documents with embeddings - skipping performance test")
                return True
            
            # Test query performance
            test_vector = "[" + ",".join(["0.1"] * 384) + "]"  # 384-dimensional test vector
            
            start_time = time.time()
            cursor.execute(f"""
                SELECT TOP 10 doc_id, VECTOR_COSINE(embedding, TO_VECTOR(?)) AS similarity
                FROM RAG.SourceDocuments 
                WHERE embedding IS NOT NULL
                ORDER BY similarity DESC
            """, (test_vector,))
            
            results = cursor.fetchall()
            end_time = time.time()
            
            query_time_ms = (end_time - start_time) * 1000
            logging.info(f"Vector similarity query time: {query_time_ms:.1f}ms")
            logging.info(f"Results returned: {len(results)}")
            
            if query_time_ms < 100:
                logging.info("✅ Query performance is excellent (<100ms)")
                return True
            elif query_time_ms < 500:
                logging.info("⚠️  Query performance is acceptable (<500ms)")
                return True
            else:
                logging.warning(f"⚠️  Query performance is slow (>{query_time_ms:.1f}ms)")
                return False
        
        conn.close()
        
    except Exception as e:
        logging.error(f"Error checking HNSW performance: {e}")
        return False

def main():
    """Run comprehensive system health check"""
    logging.info("🏥 Starting RAG System Health Check...")
    
    checks = [
        ("System Resources", check_system_resources),
        ("Docker Containers", check_docker_containers),
        ("Database Connection", check_database_connection),
        ("Data Ingestion Status", check_data_ingestion_status),
        ("HNSW Performance", check_hnsw_performance)
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        logging.info(f"\n--- {check_name} ---")
        try:
            results[check_name] = check_func()
        except Exception as e:
            logging.error(f"Check failed with exception: {e}")
            results[check_name] = False
    
    # Summary
    logging.info("\n=== Health Check Summary ===")
    all_passed = True
    
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logging.info(f"{check_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logging.info("🎉 All health checks passed! System is ready for operation.")
        return 0
    else:
        logging.error("⚠️  Some health checks failed. Please review the issues above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)