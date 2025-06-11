import sys
import time
sys.path.append('.')
from common.iris_connector import get_iris_connection

def monitor_ingestion():
    iris = get_iris_connection()
    cursor = iris.cursor()
    
    print("=== GraphRAG Ingestion Monitor ===")
    print("Press Ctrl+C to stop monitoring\n")
    
    prev_entities = 0
    prev_relationships = 0
    
    try:
        while True:
            # Get current counts
            cursor.execute('SELECT COUNT(*) FROM RAG.Entities')
            entities = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM RAG.Relationships')
            relationships = cursor.fetchone()[0]
            
            # Calculate rates
            entity_rate = entities - prev_entities
            rel_rate = relationships - prev_relationships
            
            # Display status
            timestamp = time.strftime("%H:%M:%S")
            print(f"\r[{timestamp}] Entities: {entities:,} (+{entity_rate}) | "
                  f"Relationships: {relationships:,} (+{rel_rate})    ", end='', flush=True)
            
            prev_entities = entities
            prev_relationships = relationships
            
            time.sleep(5)  # Update every 5 seconds
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
    finally:
        cursor.close()
        iris.close()

if __name__ == "__main__":
    monitor_ingestion()