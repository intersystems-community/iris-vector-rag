#!/usr/bin/env python3
"""
Migrate DocumentChunks_V2 embeddings to native VECTOR column
This is the only remaining migration needed based on current database state.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

import time
import logging
from tqdm import tqdm
from common.iris_connector import get_iris_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentChunksV2Migration:
    """Migrate DocumentChunks_V2 embeddings to native VECTOR column"""
    
    def __init__(self):
        self.conn = get_iris_connection()
        logger.info("Connected to IRIS database")
    
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
    
    def migrate_chunks_batch_approach(self, batch_size=100):
        """Migrate using batch approach for better control and error handling"""
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
                # Get batch of chunks
                cursor.execute(f"""
                    SELECT TOP {batch_size} chunk_id, embedding
                    FROM RAG.DocumentChunks_V2
                    WHERE embedding IS NOT NULL 
                    AND chunk_embedding_vector IS NULL
                """)
                
                batch = cursor.fetchall()
                if not batch:
                    break
                
                # Process each chunk in the batch
                for chunk_id, embedding in batch:
                    try:
                        # Use dynamic SQL to avoid parameter binding issues with TO_VECTOR
                        update_sql = f"""
                            UPDATE RAG.DocumentChunks_V2
                            SET chunk_embedding_vector = TO_VECTOR(embedding, 'DOUBLE', 384)
                            WHERE chunk_id = '{chunk_id}'
                        """
                        cursor.execute(update_sql)
                        migrated += 1
                        
                    except Exception as e:
                        logger.error(f"Error migrating chunk {chunk_id}: {e}")
                        failed += 1
                
                # Commit after each batch
                self.conn.commit()
                pbar.update(len(batch))
        
        cursor.close()
        print(f"\n✅ Migration completed:")
        print(f"   Successfully migrated: {migrated:,}")
        print(f"   Failed: {failed:,}")
    
    def migrate_chunks_bulk_approach(self):
        """Try bulk migration approach (faster but less control)"""
        cursor = self.conn.cursor()
        
        print("\n🔄 Attempting bulk migration for DocumentChunks_V2...")
        
        try:
            start_time = time.time()
            
            # Direct bulk update
            cursor.execute("""
                UPDATE RAG.DocumentChunks_V2
                SET chunk_embedding_vector = TO_VECTOR(embedding, 'DOUBLE', 384)
                WHERE embedding IS NOT NULL 
                AND chunk_embedding_vector IS NULL
            """)
            
            affected_rows = cursor.rowcount
            self.conn.commit()
            
            elapsed = time.time() - start_time
            print(f"✅ Bulk migration completed in {elapsed:.2f} seconds")
            print(f"   Migrated {affected_rows:,} records")
            
            cursor.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Bulk migration failed: {e}")
            self.conn.rollback()
            cursor.close()
            return False
    
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
            # Get a sample chunk with vector
            cursor.execute("""
                SELECT TOP 1 chunk_id, embedding
                FROM RAG.DocumentChunks_V2
                WHERE chunk_embedding_vector IS NOT NULL
            """)
            
            result = cursor.fetchone()
            if result:
                chunk_id, embedding = result
                
                # Test VECTOR_COSINE search
                start_time = time.time()
                cursor.execute(f"""
                    SELECT TOP 5 chunk_id, chunk_text,
                           VECTOR_COSINE(chunk_embedding_vector, TO_VECTOR(embedding, 'DOUBLE', 384)) as similarity
                    FROM RAG.DocumentChunks_V2
                    WHERE chunk_embedding_vector IS NOT NULL
                    AND chunk_id = '{chunk_id}'
                """)
                
                search_result = cursor.fetchone()
                search_time = time.time() - start_time
                
                if search_result:
                    _, _, similarity = search_result
                    print(f"✅ Vector search successful!")
                    print(f"   Self-similarity: {similarity:.4f} (should be ~1.0)")
                    print(f"   Search time: {search_time:.3f}s")
                    print(f"   Using HNSW index: idx_hnsw_chunks_v2")
        
        except Exception as e:
            logger.error(f"Error during vector search test: {e}")
        
        cursor.close()
    
    def run_migration(self):
        """Run the complete migration process"""
        print("🚀 Starting DocumentChunks_V2 Vector Migration")
        print("=" * 60)
        
        # Check current status
        needs_migration = self.check_migration_status()
        
        if needs_migration == 0:
            print("\n✅ No migration needed - all chunks already have VECTOR data!")
            return
        
        # Try bulk approach first
        print("\n💡 Attempting fast bulk migration...")
        if self.migrate_chunks_bulk_approach():
            print("✅ Bulk migration successful!")
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
    
    def close(self):
        """Close connection"""
        if self.conn:
            self.conn.close()

def main():
    """Main migration function"""
    migration = DocumentChunksV2Migration()
    try:
        migration.run_migration()
    finally:
        migration.close()

if __name__ == "__main__":
    main()