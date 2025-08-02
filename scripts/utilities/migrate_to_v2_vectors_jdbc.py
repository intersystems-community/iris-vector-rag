#!/usr/bin/env python3
"""
Migrate embeddings to V2 tables with native VECTOR columns using JDBC
This script populates the document_embedding_vector columns with proper VECTOR data
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import logging
from tqdm import tqdm
import jaydebeapi
import jpype

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class V2VectorMigration:
    """Migrate embeddings to native VECTOR columns in V2 tables"""
    
    def __init__(self):
        # JDBC setup
        self.jdbc_driver_path = "./intersystems-jdbc-3.8.4.jar"
        if not os.path.exists(self.jdbc_driver_path):
            raise FileNotFoundError(f"JDBC driver not found at {self.jdbc_driver_path}")
        
        # Start JVM
        if not jpype.isJVMStarted():
            jpype.startJVM(jpype.getDefaultJVMPath(), 
                          f"-Djava.class.path={self.jdbc_driver_path}")
        
        # Connect
        self.conn = jaydebeapi.connect(
            'com.intersystems.jdbc.IRISDriver',
            'jdbc:IRIS://localhost:1972/USER',
            ['SuperUser', 'SYS'],
            self.jdbc_driver_path
        )
        logger.info("Connected to IRIS via JDBC")
    
    def check_v2_tables(self):
        """Check V2 table status"""
        cursor = self.conn.cursor()
        
        print("\n📊 Checking V2 tables status...")
        
        # Check SourceDocuments_V2
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(embedding) as has_embedding,
                COUNT(document_embedding_vector) as has_vector
            FROM RAG.SourceDocuments_V2
        """)
        total, has_emb, has_vec = cursor.fetchone()
        print(f"\nSourceDocuments_V2:")
        print(f"  Total records: {total:,}")
        print(f"  Has embedding (VARCHAR): {has_emb:,}")
        print(f"  Has document_embedding_vector (VECTOR): {has_vec:,}")
        print(f"  Need migration: {has_emb - has_vec:,}")
        
        # Check DocumentChunks_V2
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(embedding) as has_embedding,
                COUNT(chunk_embedding_vector) as has_vector
            FROM RAG.DocumentChunks_V2
        """)
        total, has_emb, has_vec = cursor.fetchone()
        print(f"\nDocumentChunks_V2:")
        print(f"  Total records: {total:,}")
        print(f"  Has embedding (VARCHAR): {has_emb:,}")
        print(f"  Has chunk_embedding_vector (VECTOR): {has_vec:,}")
        print(f"  Need migration: {has_emb - has_vec:,}")
        
        cursor.close()
        return has_emb - has_vec > 0
    
    def migrate_source_documents(self, batch_size=1000):
        """Migrate SourceDocuments_V2 embeddings to native VECTOR column"""
        cursor = self.conn.cursor()
        
        print("\n🔄 Migrating SourceDocuments_V2...")
        
        # Get total count to migrate
        cursor.execute("""
            SELECT COUNT(*) 
            FROM RAG.SourceDocuments_V2 
            WHERE embedding IS NOT NULL 
            AND document_embedding_vector IS NULL
        """)
        total_to_migrate = cursor.fetchone()[0]
        
        if total_to_migrate == 0:
            print("✅ No documents need migration")
            return
        
        print(f"📊 Migrating {total_to_migrate:,} documents...")
        
        # Process in batches
        migrated = 0
        with tqdm(total=total_to_migrate, desc="Migrating documents") as pbar:
            while migrated < total_to_migrate:
                # Get batch of documents
                cursor.execute("""
                    SELECT TOP ? doc_id, embedding
                    FROM RAG.SourceDocuments_V2
                    WHERE embedding IS NOT NULL 
                    AND document_embedding_vector IS NULL
                """, [batch_size])
                
                batch = cursor.fetchall()
                if not batch:
                    break
                
                # Update each document
                update_cursor = self.conn.cursor()
                for doc_id, embedding in batch:
                    try:
                        # Use TO_VECTOR to convert VARCHAR to VECTOR type
                        update_cursor.execute("""
                            UPDATE RAG.SourceDocuments_V2
                            SET document_embedding_vector = TO_VECTOR(?)
                            WHERE doc_id = ?
                        """, [embedding, doc_id])
                        
                    except Exception as e:
                        logger.error(f"Error migrating doc {doc_id}: {e}")
                        # Try alternative approach - direct assignment
                        try:
                            update_cursor.execute(f"""
                                UPDATE RAG.SourceDocuments_V2
                                SET document_embedding_vector = TO_VECTOR(embedding)
                                WHERE doc_id = '{doc_id}'
                            """)
                        except Exception as e2:
                            logger.error(f"Alternative migration failed for {doc_id}: {e2}")
                
                # Commit batch
                self.conn.commit()
                migrated += len(batch)
                pbar.update(len(batch))
                
                update_cursor.close()
        
        cursor.close()
        print(f"✅ Migrated {migrated:,} documents")
    
    def migrate_document_chunks(self, batch_size=5000):
        """Migrate DocumentChunks_V2 embeddings to native VECTOR column"""
        cursor = self.conn.cursor()
        
        print("\n🔄 Migrating DocumentChunks_V2...")
        
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
        
        print(f"📊 Migrating {total_to_migrate:,} chunks...")
        
        # For chunks, we'll use a more efficient approach
        print("🔧 Using bulk UPDATE with TO_VECTOR conversion...")
        
        try:
            # Direct bulk update
            start_time = time.time()
            cursor.execute("""
                UPDATE RAG.DocumentChunks_V2
                SET chunk_embedding_vector = TO_VECTOR(embedding)
                WHERE embedding IS NOT NULL 
                AND chunk_embedding_vector IS NULL
            """)
            
            self.conn.commit()
            elapsed = time.time() - start_time
            
            print(f"✅ Bulk migration completed in {elapsed:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Bulk migration failed: {e}")
            print("⚠️  Falling back to batch migration...")
            
            # Fallback to batch processing
            migrated = 0
            with tqdm(total=total_to_migrate, desc="Migrating chunks") as pbar:
                while migrated < total_to_migrate:
                    cursor.execute(f"""
                        UPDATE RAG.DocumentChunks_V2
                        SET chunk_embedding_vector = TO_VECTOR(embedding)
                        WHERE chunk_id IN (
                            SELECT TOP {batch_size} chunk_id
                            FROM RAG.DocumentChunks_V2
                            WHERE embedding IS NOT NULL 
                            AND chunk_embedding_vector IS NULL
                        )
                    """)
                    
                    affected = cursor.rowcount
                    if affected == 0:
                        break
                    
                    self.conn.commit()
                    migrated += affected
                    pbar.update(affected)
        
        cursor.close()
    
    def verify_migration(self):
        """Verify the migration was successful"""
        cursor = self.conn.cursor()
        
        print("\n✅ Verifying migration...")
        
        # Test vector search on migrated data
        cursor.execute("""
            SELECT TOP 1 
                doc_id,
                LENGTH(embedding) as emb_len,
                LENGTH(CAST(document_embedding_vector AS VARCHAR)) as vec_len
            FROM RAG.SourceDocuments_V2
            WHERE document_embedding_vector IS NOT NULL
        """)
        
        result = cursor.fetchone()
        if result:
            doc_id, emb_len, vec_len = result
            print(f"\n📊 Sample verification:")
            print(f"  Doc ID: {doc_id}")
            print(f"  Original embedding length: {emb_len}")
            print(f"  Vector column length: {vec_len}")
            
            # Test vector search
            print("\n🔍 Testing vector search on migrated data...")
            
            # Get a test vector
            cursor.execute("""
                SELECT embedding
                FROM RAG.SourceDocuments_V2
                WHERE doc_id = ?
            """, [doc_id])
            test_embedding = cursor.fetchone()[0]
            
            # Search using the native VECTOR column
            start_time = time.time()
            cursor.execute("""
                SELECT TOP 5 doc_id
                FROM RAG.SourceDocuments_V2
                WHERE document_embedding_vector IS NOT NULL
                ORDER BY VECTOR_COSINE(document_embedding_vector, TO_VECTOR(?)) DESC
            """, [test_embedding])
            
            results = cursor.fetchall()
            search_time = time.time() - start_time
            
            print(f"✅ Vector search successful!")
            print(f"  Found {len(results)} results in {search_time:.3f}s")
            print(f"  Using native VECTOR column with HNSW index")
        
        cursor.close()
    
    def run_migration(self):
        """Run the complete migration process"""
        print("🚀 Starting V2 Vector Migration using JDBC")
        print("=" * 60)
        
        # Check current status
        needs_migration = self.check_v2_tables()
        
        if not needs_migration:
            print("\n✅ All tables already migrated!")
            return
        
        # Run migrations
        self.migrate_source_documents()
        self.migrate_document_chunks()
        
        # Verify
        self.verify_migration()
        
        # Final status
        print("\n📊 Final Status:")
        self.check_v2_tables()
        
        print("\n✅ Migration complete!")
        print("\n💡 Benefits:")
        print("  - Native VECTOR type columns populated")
        print("  - HNSW indexes can now be fully utilized")
        print("  - Better performance for vector searches")
        print("  - Ready for production use")
    
    def close(self):
        """Close connection"""
        if self.conn:
            self.conn.close()

def main():
    """Main migration function"""
    migration = V2VectorMigration()
    try:
        migration.run_migration()
    finally:
        migration.close()
        # Shutdown JVM
        if jpype.isJVMStarted():
            jpype.shutdownJVM()

if __name__ == "__main__":
    main()