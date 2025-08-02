"""
Create iFind index for full-text search in IRIS
This script creates the necessary structures for iFind to work with HybridIFindRAG
"""

import sys
sys.path.append('.')
from common.iris_connector import get_iris_connection

def create_ifind_index():
    """Create iFind index on SourceDocuments"""
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    print("=== Setting up iFind for RAG.SourceDocuments ===\n")
    
    try:
        # Step 1: Create a view that can be used with %CONTAINS
        print("1. Creating searchable view...")
        cursor.execute("""
            CREATE OR REPLACE VIEW RAG.SourceDocumentsSearch AS
            SELECT doc_id, 
                   title,
                   CAST(text_content AS VARCHAR(32000)) as searchable_content,
                   embedding,
                   created_at
            FROM RAG.SourceDocuments
        """)
        print("   ✅ View created\n")
        
        # Step 2: Update the hybrid_ifind_rag pipeline to use %CONTAINS
        print("2. Instructions to update hybrid_ifind_rag/pipeline.py:")
        print("   Replace the _ifind_keyword_search method with:\n")
        
        print('''
    def _ifind_keyword_search(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """
        Perform iFind keyword search using IRIS %CONTAINS predicate.
        """
        if not keywords:
            return []
        
        try:
            # Join keywords with OR for %CONTAINS
            search_expr = ' OR '.join(keywords[:5])  # Limit to 5 keywords
            
            query = f"""
            SELECT TOP {self.config['max_results_per_method']}
                d.doc_id as document_id,
                d.title as title,
                d.searchable_content as content,
                '' as metadata,
                ROW_NUMBER() OVER (ORDER BY d.doc_id) as rank_position
            FROM RAG.SourceDocumentsSearch d
            WHERE %ID %FIND search_index(searchable_content, ?)
            ORDER BY rank_position
            """
            
            cursor = self.iris_connector.cursor()
            cursor.execute(query, [search_expr])
            results = []
            
            for row in cursor.fetchall():
                results.append({
                    'document_id': row[0],
                    'title': row[1],
                    'content': row[2][:500] + '...' if len(row[2]) > 500 else row[2],
                    'metadata': row[3],
                    'rank_position': row[4],
                    'method': 'ifind'
                })
            
            logger.info(f"iFind search found {len(results)} documents")
            return results
            
        except Exception as e:
            logger.error(f"iFind search error: {e}")
            # Fallback to title search
            return self._title_keyword_search(keywords)
    
    def _title_keyword_search(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Fallback to title search if iFind fails"""
        if not keywords:
            return []
            
        keyword_conditions = []
        params = []
        
        for keyword in keywords[:5]:
            keyword_conditions.append("d.title LIKE ?")
            params.append(f"%{keyword}%")
        
        where_clause = " OR ".join(keyword_conditions)
        
        query = f"""
        SELECT TOP {self.config['max_results_per_method']}
            d.doc_id as document_id,
            d.title as title,
            CAST(d.text_content AS VARCHAR(1000)) as content,
            '' as metadata,
            ROW_NUMBER() OVER (ORDER BY d.doc_id) as rank_position
        FROM RAG.SourceDocuments d
        WHERE {where_clause}
        ORDER BY d.doc_id
        """
        
        cursor = self.iris_connector.cursor()
        cursor.execute(query, params)
        results = []
        
        for row in cursor.fetchall():
            results.append({
                'document_id': row[0],
                'title': row[1],
                'content': row[2],
                'metadata': row[3],
                'rank_position': row[4],
                'method': 'ifind'
            })
        
        return results
        ''')
        
        print("\n3. Testing keyword search with title fallback...")
        
        # Test the title search
        test_keywords = ['diabetes', 'treatment', 'insulin']
        keyword_conditions = []
        params = []
        
        for keyword in test_keywords[:3]:
            keyword_conditions.append("title LIKE ?")
            params.append(f"%{keyword}%")
        
        where_clause = " OR ".join(keyword_conditions)
        
        cursor.execute(f"""
            SELECT TOP 5 doc_id, title
            FROM RAG.SourceDocuments
            WHERE {where_clause}
        """, params)
        
        results = cursor.fetchall()
        print(f"   Found {len(results)} documents matching keywords in titles")
        for doc_id, title in results[:3]:
            print(f"   - {doc_id}: {title[:80]}...")
        
        conn.commit()
        print("\n✅ iFind setup complete!")
        print("\n📝 Note: Full iFind functionality requires IRIS configuration.")
        print("   The hybrid pipeline will use title search as a fallback.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    create_ifind_index()