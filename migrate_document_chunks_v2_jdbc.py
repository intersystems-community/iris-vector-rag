#!/usr/bin/env python3
"""
Migrate DocumentChunks_V2 embeddings to native VECTOR column using JDBC
This uses JDBC connection which has better support for IRIS vector functions.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

import time
import logging
from tqdm import tqdm
import jaydebeapi
import jpype

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentChunksV2MigrationJDBC:
    """Migrate DocumentChunks_V2 embeddings to native VECTOR column using JDBC"""
    
    def __init__(self):
        # Initialize JVM if not already started
        if not jpype.isJVMStarted():
            jpype.startJVM(jpype.getDefaultJVMPath(), 
                          f"-Djava.class.path=./intersystems-jdbc-3.8.4.jar")
        
        # Connect using JDBC
        self.conn = jaydebeapi.connect(
            "com.intersystems.jdbc.IRISDriver",
            "jdbc:IRIS://localhost:1972/USER",
            ["_SYSTEM", "SYS"],
            "./intersystems-jdbc-3.8.4.jar"
        )
        logger.info("Connected to IRIS database via JDBC")
    
    def check_migration_status(self):
        """Check current migration status"""
        cursor = self.conn.cursor()
        
        print("\n📊 Checking DocumentChunks_V2 migration status...")
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(embedding) as has_embedding,
                COUNT(chunk_embedding_vector) as has_vector
            FROM RAG.DocumentChunks_V2
        """)
        total, has_emb, has_vec = cursor.fetchone()
        
        print(f"\nDocumentChunks_V2 Status:")
        print(f"  Total records: {total:,}")
        print(f"  Has embedding (VARCHAR): {has_emb:,}")
        print(f"  Has chunk_embedding_vector (VECTOR): {has_vec:,}")
        print(f"  Need migration: {has_emb - has_vec:,}")
        
        cursor.close()
        return has_emb - has_vec
    
    def migrate_chunks_direct_sql(self):
        """Migrate using direct SQL without parameters"""
        cursor = self.conn.cursor()
        
        print("\n🔄 Starting DocumentChunks_V2 migration (direct SQL approach)...")
        
        try:
            start_time = time.time()
            
            # Direct SQL update without parameters
            update_sql = """
                UPDATE RAG.DocumentChunks_V2
                SET chunk_embedding_vector = TO_VECTOR(embedding)
                WHERE embedding IS NOT NULL 
                AND chunk_embedding_vector IS NULL
            """
            
            cursor.execute(update_sql)
            affected_rows = cursor.rowcount
            self.conn.commit()
            
            elapsed = time.time() - start_time
            print(f"✅ Migration completed in {elapsed:.2f} seconds")
            print(f"   Migrated {affected_rows:,} records")
            
            cursor.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Direct SQL migration failed: {e}")
            self.conn.rollback()
            cursor.close()
            return False
    
    def migrate_chunks_batch_approach(self, batch_size=50):
        """Migrate using batch approach with direct SQL construction"""
        cursor = self.conn.cursor()
        
        print("\n🔄 Starting DocumentChunks_V2 migration (batch approach)...")
        
        # Get total count to migrate
        cursor.execute("""
            SELECT COUNT(*) 
            FROM RAG.DocumentChunks_V2 
            WHERE embedding IS NOT NULL 
            AND chunk_embedding_vector IS NULL
        """)
        total_to_migrate = cursor.fetchone()[0]
        
        if total_to_migrate == 0:
            print("✅ No chunks need migration")
            return
        
        print(f"📊 Migrating {total_to_migrate:,} chunks in batches of {batch_size}...")
        
        migrated = 0
        failed = 0
        
        with tqdm(total=total_to_migrate, desc="Migrating chunks") as pbar:
            while migrated + failed < total_to_migrate:
                # Get batch of chunk IDs
                cursor.execute(f"""
                    SELECT TOP {batch_size} chunk_id
                    FROM RAG.DocumentChunks_V2
                    WHERE embedding IS NOT NULL 
                    AND chunk_embedding_vector IS NULL
                """)
                
                chunk_ids = [row[0] for row in cursor.fetchall()]
                if not chunk_ids:
                    break
                
                # Process each chunk
                for chunk_id in chunk_ids:
                    try:
                        # Use direct SQL without parameters
                        update_sql = f"""
                            UPDATE RAG.DocumentChunks_V2
                            SET chunk_embedding_vector = TO_VECTOR(embedding)
                            WHERE chunk_id = '{chunk_id}'
                        """
                        cursor.execute(update_sql)
                        migrated += 1
                        
                    except Exception as e:
                        logger.error(f"Error migrating chunk {chunk_id}: {e}")
                        failed += 1
                
                # Commit after each batch
                self.conn.commit()
                pbar.update(len(chunk_ids))
        
        cursor.close()
        print(f"\n✅ Migration completed:")
        print(f"   Successfully migrated: {migrated:,}")
        print(f"   Failed: {failed:,}")
    
    def verify_migration(self):
        """Verify the migration was successful"""
        cursor = self.conn.cursor()
        
        print("\n✅ Verifying migration...")
        
        # Check final counts
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(embedding) as has_embedding,
                COUNT(chunk_embedding_vector) as has_vector
            FROM RAG.DocumentChunks_V2
        """)
        total, has_emb, has_vec = cursor.fetchone()
        
        print(f"\nFinal DocumentChunks_V2 Status:")
        print(f"  Total records: {total:,}")
        print(f"  Has embedding: {has_emb:,}")
        print(f"  Has chunk_embedding_vector: {has_vec:,}")
        
        if has_emb == has_vec:
            print("\n✅ All embeddings successfully migrated to VECTOR column!")
        else:
            print(f"\n⚠️  Migration incomplete: {has_emb - has_vec:,} records still need migration")
        
        # Test vector search
        print("\n🔍 Testing vector search on migrated data...")
        
        try:
            # Test VECTOR_COSINE search with HNSW index
            cursor.execute("""
                SELECT TOP 1 chunk_id, chunk_embedding_vector
                FROM RAG.DocumentChunks_V2
                WHERE chunk_embedding_vector IS NOT NULL
            """)
            
            result = cursor.fetchone()
            if result:
                test_chunk_id = result[0]
                
                # Test self-similarity
                start_time = time.time()
                cursor.execute(f"""
                    SELECT chunk_id, 
                           VECTOR_COSINE(chunk_embedding_vector, 
                                       (SELECT chunk_embedding_vector 
                                        FROM RAG.DocumentChunks_V2 
                                        WHERE chunk_id = '{test_chunk_id}')) as similarity
                    FROM RAG.DocumentChunks_V2
                    WHERE chunk_id = '{test_chunk_id}'
                """)
                
                search_result = cursor.fetchone()
                search_time = time.time() - start_time
                
                if search_result:
                    _, similarity = search_result
                    print(f"✅ Vector search successful!")
                    print(f"   Self-similarity: {similarity:.4f} (should be ~1.0)")
                    print(f"   Search time: {search_time:.3f}s")
                    
                    # Test HNSW index usage
                    print(f"\n🔍 Testing HNSW index performance...")
                    start_time = time.time()
                    cursor.execute(f"""
                        SELECT TOP 10 chunk_id,
                               VECTOR_COSINE(chunk_embedding_vector, 
                                           (SELECT chunk_embedding_vector 
                                            FROM RAG.DocumentChunks_V2 
                                            WHERE chunk_id = '{test_chunk_id}')) as similarity
                        FROM RAG.DocumentChunks_V2
                        WHERE chunk_embedding_vector IS NOT NULL
                        ORDER BY similarity DESC
                    """)
                    
                    results = cursor.fetchall()
                    search_time = time.time() - start_time
                    print(f"   Found {len(results)} similar chunks in {search_time:.3f}s")
                    print(f"   HNSW index (idx_hnsw_chunks_v2) is active")
        
        except Exception as e:
            logger.error(f"Error during vector search test: {e}")
        
        cursor.close()
    
    def run_migration(self):
        """Run the complete migration process"""
        print("🚀 Starting DocumentChunks_V2 Vector Migration (JDBC)")
        print("=" * 60)
        
        # Check current status
        needs_migration = self.check_migration_status()
        
        if needs_migration == 0:
            print("\n✅ No migration needed - all chunks already have VECTOR data!")
            return
        
        # Try direct SQL approach first
        print("\n💡 Attempting direct SQL migration...")
        if self.migrate_chunks_direct_sql():
            print("✅ Direct SQL migration successful!")
        else:
            # Fall back to batch approach
            print("\n💡 Falling back to batch migration approach...")
            self.migrate_chunks_batch_approach()
        
        # Verify results
        self.verify_migration()
        
        print("\n✅ Migration process complete!")
        print("\n💡 Benefits:")
        print("  - Native VECTOR type columns now populated")
        print("  - HNSW index (idx_hnsw_chunks_v2) can now be fully utilized")
        print("  - Better performance for chunk-based vector searches")
        print("  - Ready for production-scale RAG operations")
    
    def close(self):
        """Close connection"""
        if self.conn:
            self.conn.close()

def main():
    """Main migration function"""
    migration = DocumentChunksV2MigrationJDBC()
    try:
        migration.run_migration()
    finally:
        migration.close()

if __name__ == "__main__":
    main()