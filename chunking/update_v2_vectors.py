#!/usr/bin/env python3
"""
Update V2 Vector Columns
Updates existing DocumentChunks_V2 records to populate the VECTOR columns
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from common.iris_connector import get_iris_connection
import time

class V2VectorUpdater:
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
        self.conn = get_iris_connection()
        
    def update_vectors(self):
        """Update VECTOR columns from existing VARCHAR embeddings"""
        cursor = self.conn.cursor()
        
        try:
            # Get total count
            cursor.execute('SELECT COUNT(*) FROM RAG.DocumentChunks_V2 WHERE embedding IS NOT NULL AND chunk_embedding_vector IS NULL')
            total_to_update = cursor.fetchone()[0]
            print(f"🚀 Found {total_to_update:,} chunks to update with VECTOR data")
            
            if total_to_update == 0:
                print("✅ No chunks need updating")
                return
            
            updated = 0
            start_time = time.time()
            
            # Process in batches
            while updated < total_to_update:
                # Get a batch of chunks that need vector updates
                cursor.execute(f'''
                    SELECT TOP {self.batch_size} chunk_id, embedding 
                    FROM RAG.DocumentChunks_V2 
                    WHERE embedding IS NOT NULL 
                    AND chunk_embedding_vector IS NULL
                ''')
                
                batch = cursor.fetchall()
                if not batch:
                    break
                
                # Update each record individually
                for chunk_id, embedding_str in batch:
                    try:
                        # Use dynamic SQL to avoid parameter issues with TO_VECTOR
                        update_sql = f"UPDATE RAG.DocumentChunks_V2 SET chunk_embedding_vector = TO_VECTOR(embedding, 'FLOAT', 384) WHERE chunk_id = '{chunk_id}'"
                        cursor.execute(update_sql)
                        updated += 1
                        
                    except Exception as e:
                        print(f"⚠️  Error updating chunk {chunk_id}: {e}")
                        continue
                
                # Commit after each batch
                self.conn.commit()
                
                # Progress report
                if updated % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = updated / elapsed if elapsed > 0 else 0
                    eta = (total_to_update - updated) / rate if rate > 0 else 0
                    print(f"📊 Progress: {updated:,}/{total_to_update:,} ({updated/total_to_update*100:.1f}%) | "
                          f"Rate: {rate:.1f} chunks/s | ETA: {eta/60:.1f}m")
            
            elapsed = time.time() - start_time
            print(f"\n🎉 UPDATE COMPLETE!")
            print(f"📊 Updated: {updated:,} chunks")
            print(f"📊 Time: {elapsed/60:.1f} minutes")
            print(f"📊 Rate: {updated/elapsed:.1f} chunks/s")
            
            # Verify results
            cursor.execute('SELECT COUNT(*) FROM RAG.DocumentChunks_V2 WHERE chunk_embedding_vector IS NOT NULL')
            vector_count = cursor.fetchone()[0]
            print(f"\n✅ Final verification:")
            print(f"   - Chunks with VECTOR data: {vector_count:,}")
            
        except Exception as e:
            print(f"❌ Update failed: {e}")
            raise
        finally:
            cursor.close()

def main():
    print("🚀 V2 Vector Column Updater")
    print("=" * 50)
    
    # Initialize updater
    updater = V2VectorUpdater(batch_size=100)
    
    # Run update
    updater.update_vectors()

if __name__ == "__main__":
    main()