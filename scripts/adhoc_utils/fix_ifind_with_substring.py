"""
Fix for Hybrid iFind RAG using SUBSTRING for stream field search
Based on IRIS documentation, STREAM fields only support:
- NULL testing
- Length testing (CHARACTER_LENGTH, CHAR_LENGTH, DATALENGTH)
- Substring extraction (SUBSTRING)
"""

def generate_substring_search_method():
    """Generate the fixed _ifind_keyword_search method using SUBSTRING"""
    
    print("=== Fix for Hybrid iFind RAG using SUBSTRING ===\n")
    print("Based on IRIS documentation, we can use SUBSTRING to search in STREAM fields.\n")
    
    print("Replace the _ifind_keyword_search method in hybrid_ifind_rag/pipeline.py with:\n")
    
    fixed_method = '''    def _ifind_keyword_search(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """
        Perform keyword search using SUBSTRING on stream fields and title search.
        Since IRIS doesn't support LIKE on STREAM fields, we use a combination of:
        1. Title search (VARCHAR field)
        2. SUBSTRING search on first 5000 chars of text_content
        
        Args:
            keywords: List of keywords to search for
            
        Returns:
            List of documents with keyword match scores
        """
        if not keywords:
            return []
        
        try:
            # Build conditions for both title and content search
            conditions = []
            params = []
            
            for keyword in keywords[:5]:  # Limit to 5 keywords
                # Title search (case-insensitive)
                conditions.append("UPPER(d.title) LIKE UPPER(?)")
                params.append(f"%{keyword}%")
                
                # Content search using SUBSTRING on first 5000 characters
                # This checks if the keyword appears in the beginning of the document
                conditions.append("""
                    POSITION(UPPER(?), UPPER(SUBSTRING(d.text_content, 1, 5000))) > 0
                """)
                params.append(keyword)
            
            where_clause = " OR ".join(conditions)
            
            query = f"""
            SELECT DISTINCT TOP {self.config['max_results_per_method']}
                d.doc_id as document_id,
                d.title as title,
                SUBSTRING(d.text_content, 1, 1000) as content,
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
                    'content': row[2] if row[2] else 'Content preview not available',
                    'metadata': row[3],
                    'rank_position': row[4],
                    'method': 'ifind'
                })
            
            cursor.close()
            logger.info(f"iFind keyword search found {len(results)} documents")
            return results
            
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            # Fallback to title-only search
            return self._title_only_search(keywords)
    
    def _title_only_search(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Fallback to title-only search"""
        if not keywords:
            return []
            
        try:
            keyword_conditions = []
            params = []
            
            for keyword in keywords[:5]:
                keyword_conditions.append("UPPER(d.title) LIKE UPPER(?)")
                params.append(f"%{keyword}%")
            
            where_clause = " OR ".join(keyword_conditions)
            
            query = f"""
            SELECT TOP {self.config['max_results_per_method']}
                d.doc_id as document_id,
                d.title as title,
                SUBSTRING(d.text_content, 1, 500) as content,
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
                    'content': row[2] if row[2] else 'Content preview not available',
                    'metadata': row[3],
                    'rank_position': row[4],
                    'method': 'ifind'
                })
            
            cursor.close()
            return results
            
        except Exception as e:
            logger.error(f"Error in title search: {e}")
            return []'''
    
    print(fixed_method)
    
    print("\n\n=== Key Points ===")
    print("1. IRIS STREAM fields don't support LIKE operator")
    print("2. We can use SUBSTRING to extract portions of the stream")
    print("3. POSITION function finds substring positions")
    print("4. This searches in both title and first 5000 chars of content")
    print("5. Falls back to title-only search if needed")
    print("\nThis provides a working keyword search for HybridIFindRAG!")

if __name__ == "__main__":
    generate_substring_search_method()