"""
Minimal test with 1000 documents for TDD approach
"""

import logging
import random
from typing import Dict, Any, List
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Minimum number of documents required
MIN_DOCUMENTS = 1000

class Document:
    """Simple document class for testing"""
    def __init__(self, id, title, content, score=1.0):
        self.id = id
        self.title = title
        self.content = content
        self.score = score
        
    def __str__(self):
        return f"Document({self.id}: {self.title})"

class SimpleMockCursor:
    """Simple mock cursor for testing with 1000+ documents"""
    def __init__(self):
        self.data = self._generate_mock_data(MIN_DOCUMENTS)
        self.query_history = []
        
    def _generate_mock_data(self, count):
        """Generate mock data with specified number of documents"""
        data = []
        topics = ["diabetes", "insulin", "cancer", "treatment", "vaccine", "research"]
        
        for i in range(count):
            topic = random.choice(topics)
            data.append({
                "doc_id": f"doc_{i:04d}",
                "title": f"Document {i} about {topic}",
                "content": f"This is document {i} with information about {topic} research.",
                "embedding": "[0.1, 0.2, 0.3, 0.4, 0.5]"
            })
        
        return data
        
    def execute(self, query, params=None):
        """Execute a SQL query"""
        self.query_history.append((query, params))
        return self
        
    def fetchone(self):
        """Fetch one result"""
        if "COUNT" in self.query_history[-1][0].upper():
            return [len(self.data)]
        return None
        
    def fetchall(self):
        """Fetch all results"""
        if "SELECT" in self.query_history[-1][0].upper():
            # Return top 5 documents for any query
            return [(doc["doc_id"], doc["title"], doc["content"]) for doc in self.data[:5]]
        return []
        
    def __enter__(self):
        return self
        
    def __exit__(self, *args):
        pass

class MockIRISConnector:
    """Mock IRIS connector with 1000+ documents"""
    def __init__(self):
        self._cursor = SimpleMockCursor()
        
    def cursor(self):
        return self._cursor

class MockBasicRAG:
    """Mock BasicRAG pipeline for testing"""
    
    def __init__(self, iris_connector, embedding_func, llm_func):
        """Initialize with required components"""
        self.iris_connector = iris_connector
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        logger.info("MockBasicRAG initialized")
        
    def run(self, query, top_k=5):
        """Run the pipeline with a query"""
        logger.info(f"Running RAG query: '{query}'")
        
        # Use connector to retrieve documents
        docs = self._retrieve_documents(query, top_k)
        
        # Generate answer
        context = "\n\n".join([doc.content for doc in docs])
        prompt = f"Query: {query}\n\nContext:\n{context}\n\nAnswer:"
        answer = self.llm_func(prompt)
        
        # Return result
        return {
            "query": query,
            "answer": answer,
            "retrieved_documents": docs
        }
        
    def _retrieve_documents(self, query, top_k):
        """Retrieve documents from the database"""
        start_time = time.time()
        
        # Get embedding for query
        query_embedding = self.embedding_func(query)
        
        # Retrieve documents from database
        with self.iris_connector.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM SourceDocuments")
            doc_count = cursor.fetchone()[0]
            logger.info(f"Database has {doc_count} documents")
            
            # Simulate vector search by just returning top k documents
            cursor.execute("SELECT doc_id, title, content FROM SourceDocuments LIMIT ?", (top_k,))
            results = cursor.fetchall()
            
        # Convert to Document objects
        documents = []
        for row in results:
            doc_id, title, content = row
            documents.append(Document(doc_id, title, content, score=0.9))
        
        duration = time.time() - start_time
        logger.info(f"Retrieved {len(documents)} documents in {duration:.2f} seconds")
        
        return documents

def test_basic_rag_with_1000_docs():
    """Test a basic RAG pipeline with 1000+ documents."""
    # Create mock components
    connector = MockIRISConnector()
    
    # Simple embedding function
    def mock_embedding_func(text):
        return [0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Simple LLM function
    def mock_llm_func(prompt):
        # Count mentions of topics in the prompt
        topics = {
            "diabetes": prompt.lower().count("diabetes"),
            "insulin": prompt.lower().count("insulin"),
            "cancer": prompt.lower().count("cancer")
        }
        most_mentioned = max(topics.items(), key=lambda x: x[1])[0]
        return f"Based on the context, {most_mentioned} is an important topic in medical research."
    
    # Create pipeline with mock components
    pipeline = MockBasicRAG(
        iris_connector=connector,
        embedding_func=mock_embedding_func,
        llm_func=mock_llm_func
    )
    
    # Run query
    query = "What is the role of insulin in diabetes?"
    start_time = time.time()
    result = pipeline.run(query, top_k=5)
    duration = time.time() - start_time
    
    # Verify result format
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "query" in result, "Result should contain the query"
    assert "answer" in result, "Result should contain an answer"
    assert "retrieved_documents" in result, "Result should contain retrieved documents"
    
    # Verify we got documents
    assert len(result["retrieved_documents"]) > 0, "Should retrieve at least one document"
    
    # Verify document count in database (via cursor)
    with connector.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM SourceDocuments")
        doc_count = cursor.fetchone()[0]
        assert doc_count >= MIN_DOCUMENTS, f"Database should have at least {MIN_DOCUMENTS} documents"
    
    # Log results
    logger.info(f"Query execution time: {duration:.2f} seconds")
    logger.info(f"Retrieved {len(result['retrieved_documents'])} documents")
    logger.info(f"Answer: {result['answer']}")
    
    logger.info("Test with 1000 documents passed successfully")
    return result

if __name__ == "__main__":
    result = test_basic_rag_with_1000_docs()
    print(f"Retrieved {len(result['retrieved_documents'])} documents")
    print(f"Answer: {result['answer']}")
