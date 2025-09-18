#!/usr/bin/env python3
"""
Knowledge Graph Monitoring Tool
Tracks entities and relationships per document during ingestion
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

# Add project paths
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeGraphMonitor:
    """Monitor knowledge graph population during ingestion"""
    
    def __init__(self):
        self.last_counts = {}
        self.start_time = datetime.now()
        
    def get_database_connection(self):
        """Get database connection"""
        try:
            from common.config import get_iris_config
            from common.iris_client import IRISClient
            
            config = get_iris_config()
            return IRISClient(config)
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return None
    
    def query_entity_counts(self, client):
        """Query entity counts per document"""
        try:
            # Query entities table using correct column names
            entity_query = """
            SELECT
                source_doc_id,
                COUNT(*) as entity_count,
                COUNT(DISTINCT entity_type) as unique_types
            FROM RAG.Entities
            GROUP BY source_doc_id
            ORDER BY source_doc_id
            """
            
            entities = client.query(entity_query)
            return entities
            
        except Exception as e:
            logger.warning(f"Entity query failed: {e}")
            return []
    
    def query_relationship_counts(self, client):
        """Query relationship counts per document"""
        try:
            # Query relationships table using correct column names
            relationship_query = """
            SELECT
                source_doc_id,
                COUNT(*) as relationship_count,
                COUNT(DISTINCT relationship_type) as unique_rel_types
            FROM RAG.Relationships
            GROUP BY source_doc_id
            ORDER BY source_doc_id
            """
            
            relationships = client.query(relationship_query)
            return relationships
            
        except Exception as e:
            logger.warning(f"Relationship query failed: {e}")
            return []
    
    def query_total_counts(self, client):
        """Query total entity and relationship counts"""
        try:
            total_stats = {}
            
            # Total entities - avoid using COUNT as column alias
            entity_total = client.query("SELECT COUNT(*) as total_count FROM RAG.Entities")
            if entity_total:
                total_stats['total_entities'] = entity_total[0]['total_count']
            
            # Total relationships
            rel_total = client.query("SELECT COUNT(*) as total_count FROM RAG.Relationships")
            if rel_total:
                total_stats['total_relationships'] = rel_total[0]['total_count']
            
            # Unique documents with entities
            doc_count = client.query("""
                SELECT COUNT(DISTINCT source_doc_id) as docs_count
                FROM RAG.Entities
            """)
            if doc_count:
                total_stats['documents_with_entities'] = doc_count[0]['docs_count']
            
            # Entity types
            entity_types = client.query("""
                SELECT entity_type, COUNT(*) as type_count
                FROM RAG.Entities
                GROUP BY entity_type
                ORDER BY type_count DESC
            """)
            total_stats['entity_types'] = entity_types
            
            # Relationship types
            rel_types = client.query("""
                SELECT relationship_type, COUNT(*) as type_count
                FROM RAG.Relationships
                GROUP BY relationship_type
                ORDER BY type_count DESC
            """)
            total_stats['relationship_types'] = rel_types
            
            return total_stats
            
        except Exception as e:
            logger.warning(f"Total counts query failed: {e}")
            return {}
    
    def display_monitoring_report(self, entity_data, relationship_data, total_stats):
        """Display comprehensive monitoring report"""
        
        print("\n" + "="*80)
        print("🕸️  KNOWLEDGE GRAPH MONITORING REPORT")
        print("="*80)
        print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🚀 Monitoring Duration: {datetime.now() - self.start_time}")
        
        # Total Statistics
        print("\n📊 OVERALL STATISTICS")
        print("-" * 40)
        total_entities = total_stats.get('total_entities', 0)
        total_relationships = total_stats.get('total_relationships', 0)
        docs_with_entities = total_stats.get('documents_with_entities', 0)
        
        print(f"🎯 Total Entities: {total_entities:,}")
        print(f"🔗 Total Relationships: {total_relationships:,}")
        print(f"📄 Documents with Entities: {docs_with_entities}")
        
        if docs_with_entities > 0:
            avg_entities = total_entities / docs_with_entities
            avg_relationships = total_relationships / docs_with_entities if total_relationships > 0 else 0
            print(f"📈 Avg Entities/Doc: {avg_entities:.1f}")
            print(f"📈 Avg Relationships/Doc: {avg_relationships:.1f}")
        
        # Entity Types Distribution
        entity_types = total_stats.get('entity_types', [])
        if entity_types:
            print("\n🏷️  TOP ENTITY TYPES")
            print("-" * 40)
            for i, etype in enumerate(entity_types[:10]):
                print(f"  {i+1:2d}. {etype['entity_type']:<20} {etype['type_count']:>8,}")
        
        # Relationship Types Distribution
        rel_types = total_stats.get('relationship_types', [])
        if rel_types:
            print("\n🔗 TOP RELATIONSHIP TYPES")
            print("-" * 40)
            for i, rtype in enumerate(rel_types[:10]):
                print(f"  {i+1:2d}. {rtype['relationship_type']:<20} {rtype['type_count']:>8,}")
        
        # Per-Document Details (top documents)
        if entity_data and relationship_data:
            print("\n📋 TOP DOCUMENTS BY ENTITY COUNT")
            print("-" * 40)
            
            # Combine entity and relationship data
            doc_stats = {}
            for ent in entity_data:
                doc = ent['source_doc_id']
                doc_stats[doc] = {
                    'entities': ent['entity_count'],
                    'entity_types': ent['unique_types'],
                    'relationships': 0,
                    'rel_types': 0
                }
            
            for rel in relationship_data:
                doc = rel['source_doc_id']
                if doc in doc_stats:
                    doc_stats[doc]['relationships'] = rel['relationship_count']
                    doc_stats[doc]['rel_types'] = rel['unique_rel_types']
            
            # Sort by entity count
            sorted_docs = sorted(doc_stats.items(), 
                               key=lambda x: x[1]['entities'], 
                               reverse=True)
            
            for i, (doc, stats) in enumerate(sorted_docs[:15]):
                doc_name = doc.split('/')[-1] if '/' in doc else doc
                print(f"  {i+1:2d}. {doc_name:<25} "
                      f"E:{stats['entities']:>4} "
                      f"R:{stats['relationships']:>4} "
                      f"ET:{stats['entity_types']:>2} "
                      f"RT:{stats['rel_types']:>2}")
        
        print("\n" + "="*80)
    
    def run_monitoring_loop(self, interval_seconds=30):
        """Run continuous monitoring loop"""
        
        print("🚀 Starting Knowledge Graph Monitoring...")
        print(f"📊 Update interval: {interval_seconds} seconds")
        print("⏹️  Press Ctrl+C to stop monitoring")
        
        try:
            while True:
                client = self.get_database_connection()
                if not client:
                    print("❌ Database connection failed, retrying in 30 seconds...")
                    time.sleep(30)
                    continue
                
                try:
                    with client:
                        # Query current state
                        entity_data = self.query_entity_counts(client)
                        relationship_data = self.query_relationship_counts(client)
                        total_stats = self.query_total_counts(client)
                        
                        # Display report
                        self.display_monitoring_report(entity_data, relationship_data, total_stats)
                        
                        # Save snapshot
                        self.save_monitoring_snapshot({
                            'timestamp': datetime.now().isoformat(),
                            'entity_data': entity_data,
                            'relationship_data': relationship_data,
                            'total_stats': total_stats
                        })
                        
                except Exception as e:
                    logger.error(f"Monitoring iteration failed: {e}")
                
                # Wait for next iteration
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
    
    def save_monitoring_snapshot(self, data):
        """Save monitoring snapshot to file"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            snapshot_file = f"logs/kg_monitoring_snapshot_{timestamp}.json"
            
            # Ensure logs directory exists
            Path("logs").mkdir(exist_ok=True)
            
            with open(snapshot_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save snapshot: {e}")
    
    def run_single_check(self):
        """Run a single monitoring check"""
        
        client = self.get_database_connection()
        if not client:
            print("❌ Database connection failed")
            return
        
        try:
            with client:
                entity_data = self.query_entity_counts(client)
                relationship_data = self.query_relationship_counts(client)
                total_stats = self.query_total_counts(client)
                
                self.display_monitoring_report(entity_data, relationship_data, total_stats)
                
        except Exception as e:
            logger.error(f"Single check failed: {e}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor Knowledge Graph Population')
    parser.add_argument('--continuous', '-c', action='store_true',
                      help='Run continuous monitoring')
    parser.add_argument('--interval', '-i', type=int, default=30,
                      help='Update interval in seconds (default: 30)')
    
    args = parser.parse_args()
    
    monitor = KnowledgeGraphMonitor()
    
    if args.continuous:
        monitor.run_monitoring_loop(args.interval)
    else:
        monitor.run_single_check()

if __name__ == "__main__":
    main()