#!/usr/bin/env python3
"""
FAST HNSW VALIDATION SCRIPT - FIXED VERSION
============================================

This script implements a fast approach to prove the HNSW performance improvement concept:
1. Create a small test table with 1000 records
2. Test HNSW index creation on VECTOR columns
3. Compare performance: VARCHAR vs VECTOR with HNSW
4. Demonstrate the 70% performance improvement

FIXED: Handles embedding function correctly to avoid dimension mismatch.
"""

import sys
import time
import json
import logging
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) # Add project root
from common.iris_connector import get_iris_connection
from common.utils import get_embedding_func

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FastHNSWValidatorFixed:
    def __init__(self):
        self.conn = get_iris_connection()
        self.cursor = self.conn.cursor()
        self.embedding_func = get_embedding_func()
        
    def get_single_embedding(self, text: str):
        """Get a single embedding vector, handling the batch return correctly"""
        if not text or not text.strip():
            # Return a zero vector for empty text
            return [0.0] * 384
            
        result = self.embedding_func(text)
        
        # Handle different return formats
        if isinstance(result, list) and len(result) > 0:
            # If it's a list of embeddings, take the first one
            embedding = result[0]
            if hasattr(embedding, 'tolist'):
                # Convert numpy array to list
                return embedding.tolist()
            elif isinstance(embedding, list):
                return embedding
            else:
                return list(embedding)
        else:
            # If it's a single embedding
            if hasattr(result, 'tolist'):
                return result.tolist()
            elif isinstance(result, list):
                return result
            else:
                return list(result)
        
    def step1_create_test_table(self):
        """Create a small test table with 1000 records for fast validation"""
        logger.info("🔧 Step 1: Creating test table with 1000 records")
        
        try:
            # Drop test table if exists
            try:
                self.cursor.execute("DROP TABLE RAG.TestHNSW")
                logger.info("Dropped existing test table")
            except:
                pass
            
            # Create test table with both VARCHAR and VECTOR columns
            create_sql = """
            CREATE TABLE RAG.TestHNSW (
                doc_id VARCHAR(255) NOT NULL,
                title VARCHAR(1000),
                text_content LONGVARCHAR,
                embedding_varchar VARCHAR(50000),
                embedding_vector VECTOR(FLOAT, 384),
                PRIMARY KEY (doc_id)
            )
            """
            self.cursor.execute(create_sql)
            logger.info("✅ Created RAG.TestHNSW table")
            
            # Copy 1000 records from existing data
            logger.info("📊 Copying 1000 records from RAG.SourceDocuments...")
            copy_sql = """
            INSERT INTO RAG.TestHNSW (doc_id, title, text_content, embedding_varchar)
            SELECT TOP 1000 doc_id, title, text_content, embedding
            FROM RAG.SourceDocuments 
            WHERE embedding IS NOT NULL
            """
            self.cursor.execute(copy_sql)
            
            # Convert VARCHAR embeddings to VECTOR format
            logger.info("🔄 Converting VARCHAR embeddings to VECTOR format...")
            self.cursor.execute("SELECT doc_id, embedding_varchar FROM RAG.TestHNSW")
            records = self.cursor.fetchall()
            
            converted_count = 0
            for doc_id, embedding_str in records:
                try:
                    # Parse the embedding string
                    if embedding_str.startswith('['):
                        embedding_list = json.loads(embedding_str)
                    else:
                        embedding_list = [float(x.strip()) for x in embedding_str.split(',')]
                    
                    # Ensure it's exactly 384 dimensions
                    if len(embedding_list) != 384:
                        logger.warning(f"Embedding for {doc_id} has {len(embedding_list)} dimensions, expected 384")
                        continue
                    
                    # Convert to VECTOR format
                    vector_str = f"[{','.join(map(str, embedding_list))}]"
                    
                    update_sql = "UPDATE RAG.TestHNSW SET embedding_vector = TO_VECTOR(?) WHERE doc_id = ?"
                    self.cursor.execute(update_sql, (vector_str, doc_id))
                    converted_count += 1
                    
                    if converted_count % 100 == 0:
                        logger.info(f"Converted {converted_count} embeddings...")
                        
                except Exception as e:
                    logger.warning(f"Failed to convert embedding for {doc_id}: {e}")
            
            self.conn.commit()
            logger.info(f"✅ Successfully created test table with {converted_count} records")
            return converted_count
            
        except Exception as e:
            logger.error(f"❌ Failed to create test table: {e}")
            self.conn.rollback()
            raise
    
    def step2_create_hnsw_index(self):
        """Create HNSW index on the VECTOR column"""
        logger.info("🔧 Step 2: Creating HNSW index on VECTOR column")
        
        try:
            # Drop existing index if exists
            try:
                self.cursor.execute("DROP INDEX RAG.TestHNSW.idx_test_hnsw")
                logger.info("Dropped existing HNSW index")
            except:
                pass
            
            # Create HNSW index
            hnsw_sql = """
            CREATE INDEX idx_test_hnsw 
            ON RAG.TestHNSW (embedding_vector)
            AS HNSW(Distance='COSINE')
            """
            
            start_time = time.time()
            self.cursor.execute(hnsw_sql)
            end_time = time.time()
            
            self.conn.commit()
            index_creation_time = end_time - start_time
            logger.info(f"✅ HNSW index created successfully in {index_creation_time:.2f}s")
            return index_creation_time
            
        except Exception as e:
            logger.error(f"❌ Failed to create HNSW index: {e}")
            raise
    
    def step3_performance_comparison(self, num_queries: int = 10):
        """Compare performance between VARCHAR and VECTOR with HNSW"""
        logger.info(f"🔧 Step 3: Performance comparison with {num_queries} queries")
        
        # Generate test queries
        test_queries = [
            "diabetes treatment",
            "heart disease symptoms", 
            "cancer research",
            "blood pressure medication",
            "mental health therapy",
            "vaccine effectiveness",
            "surgical procedures",
            "diagnostic imaging",
            "patient care",
            "medical research"
        ][:num_queries]
        
        varchar_times = []
        vector_times = []
        
        for i, query in enumerate(test_queries):
            logger.info(f"📊 Testing query {i+1}/{num_queries}: '{query}'")
            
            # Generate query embedding - FIXED to get single embedding
            query_embedding = self.get_single_embedding(query)
            if len(query_embedding) != 384:
                logger.error(f"Query embedding has {len(query_embedding)} dimensions, expected 384")
                continue
                
            query_vector_str = f"[{','.join(map(str, query_embedding))}]"
            
            # Test 1: VARCHAR similarity search (slower)
            varchar_sql = """
            SELECT TOP 5 doc_id, title,
                   VECTOR_COSINE(TO_VECTOR(embedding_varchar), TO_VECTOR(?)) as similarity
            FROM RAG.TestHNSW 
            WHERE embedding_varchar IS NOT NULL
            ORDER BY similarity DESC
            """
            
            start_time = time.time()
            self.cursor.execute(varchar_sql, (query_vector_str,))
            varchar_results = self.cursor.fetchall()
            varchar_time = time.time() - start_time
            varchar_times.append(varchar_time)
            
            # Test 2: VECTOR with HNSW search (faster)
            vector_sql = """
            SELECT TOP 5 doc_id, title,
                   VECTOR_COSINE(embedding_vector, TO_VECTOR(?)) as similarity
            FROM RAG.TestHNSW 
            WHERE embedding_vector IS NOT NULL
            ORDER BY similarity DESC
            """
            
            start_time = time.time()
            self.cursor.execute(vector_sql, (query_vector_str,))
            vector_results = self.cursor.fetchall()
            vector_time = time.time() - start_time
            vector_times.append(vector_time)
            
            improvement = ((varchar_time - vector_time) / varchar_time) * 100
            logger.info(f"  VARCHAR: {varchar_time:.3f}s, VECTOR+HNSW: {vector_time:.3f}s, Improvement: {improvement:.1f}%")
        
        # Calculate averages
        avg_varchar_time = sum(varchar_times) / len(varchar_times)
        avg_vector_time = sum(vector_times) / len(vector_times)
        avg_improvement = ((avg_varchar_time - avg_vector_time) / avg_varchar_time) * 100
        
        logger.info(f"\n📈 PERFORMANCE RESULTS:")
        logger.info(f"Average VARCHAR time: {avg_varchar_time:.3f}s")
        logger.info(f"Average VECTOR+HNSW time: {avg_vector_time:.3f}s")
        logger.info(f"Average improvement: {avg_improvement:.1f}%")
        
        return {
            'varchar_times': varchar_times,
            'vector_times': vector_times,
            'avg_varchar_time': avg_varchar_time,
            'avg_vector_time': avg_vector_time,
            'avg_improvement': avg_improvement
        }
    
    def step4_validate_results(self):
        """Validate that both approaches return similar results"""
        logger.info("🔧 Step 4: Validating result consistency")
        
        query = "diabetes treatment"
        query_embedding = self.get_single_embedding(query)
        query_vector_str = f"[{','.join(map(str, query_embedding))}]"
        
        # Get results from both approaches
        varchar_sql = """
        SELECT TOP 3 doc_id, VECTOR_COSINE(TO_VECTOR(embedding_varchar), TO_VECTOR(?)) as similarity
        FROM RAG.TestHNSW 
        WHERE embedding_varchar IS NOT NULL
        ORDER BY similarity DESC
        """
        
        vector_sql = """
        SELECT TOP 3 doc_id, VECTOR_COSINE(embedding_vector, TO_VECTOR(?)) as similarity
        FROM RAG.TestHNSW 
        WHERE embedding_vector IS NOT NULL
        ORDER BY similarity DESC
        """
        
        self.cursor.execute(varchar_sql, (query_vector_str,))
        varchar_results = self.cursor.fetchall()
        
        self.cursor.execute(vector_sql, (query_vector_str,))
        vector_results = self.cursor.fetchall()
        
        logger.info("📊 Result comparison:")
        logger.info("VARCHAR results:")
        for doc_id, sim in varchar_results:
            logger.info(f"  {doc_id}: {sim:.4f}")
        
        logger.info("VECTOR+HNSW results:")
        for doc_id, sim in vector_results:
            logger.info(f"  {doc_id}: {sim:.4f}")
        
        # Check if top results are similar
        varchar_top = [r[0] for r in varchar_results]
        vector_top = [r[0] for r in vector_results]
        overlap = len(set(varchar_top) & set(vector_top))
        
        logger.info(f"✅ Result overlap: {overlap}/{len(varchar_top)} documents match")
        return overlap >= len(varchar_top) * 0.7  # 70% overlap is good
    
    def cleanup(self):
        """Clean up test resources"""
        try:
            self.cursor.execute("DROP TABLE RAG.TestHNSW")
            self.conn.commit()
            logger.info("🧹 Cleaned up test table")
        except:
            pass
        finally:
            self.cursor.close()
    
    def run_full_validation(self):
        """Run the complete fast validation process"""
        logger.info("🚀 STARTING FAST HNSW VALIDATION (FIXED)")
        logger.info("=" * 60)
        
        try:
            # Step 1: Create test table
            record_count = self.step1_create_test_table()
            
            # Step 2: Create HNSW index
            index_time = self.step2_create_hnsw_index()
            
            # Step 3: Performance comparison
            perf_results = self.step3_performance_comparison()
            
            # Step 4: Validate results
            results_valid = self.step4_validate_results()
            
            # Summary
            logger.info("\n🎯 VALIDATION SUMMARY:")
            logger.info("=" * 60)
            logger.info(f"✅ Test records: {record_count}")
            logger.info(f"✅ HNSW index creation: {index_time:.2f}s")
            logger.info(f"✅ Performance improvement: {perf_results['avg_improvement']:.1f}%")
            logger.info(f"✅ Result consistency: {'PASS' if results_valid else 'FAIL'}")
            
            if perf_results['avg_improvement'] > 30:
                logger.info("🎉 SUCCESS: HNSW provides significant performance improvement!")
                logger.info("💡 Ready to proceed with full migration strategy")
            else:
                logger.warning("⚠️  Performance improvement less than expected")
            
            return perf_results
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            raise
        finally:
            self.cleanup()

def main():
    """Main execution function"""
    validator = FastHNSWValidatorFixed()
    
    try:
        results = validator.run_full_validation()
        
        print("\n" + "="*60)
        print("🎯 FAST HNSW VALIDATION COMPLETE")
        print("="*60)
        print(f"Performance improvement: {results['avg_improvement']:.1f}%")
        print(f"Average VARCHAR time: {results['avg_varchar_time']:.3f}s")
        print(f"Average VECTOR+HNSW time: {results['avg_vector_time']:.3f}s")
        
        if results['avg_improvement'] > 30:
            print("✅ HNSW validation successful - proceed with migration!")
        else:
            print("⚠️  Performance improvement below expectations")
            
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())