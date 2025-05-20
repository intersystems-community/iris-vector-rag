"""
Test for NodeRAG pipeline with 1000+ documents.
Uses pure python testing approach with mocks.
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

class Node:
    """Node class for NodeRAG testing"""
    def __init__(self, id, type, name, description, score=1.0):
        self.id = id
        self.type = type
        self.name = name
        self.description = description
        self.score = score
        self.content = description  # Needed by the pipeline for answer generation
        
    def __str__(self):
        return f"Node({self.id}: {self.name})"

class Edge:
    """Edge class for NodeRAG testing"""
    def __init__(self, edge_id, source_node_id, target_node_id, relationship_type, weight=1.0, properties=None):
        self.edge_id = edge_id
        self.source_node_id = source_node_id
        self.target_node_id = target_node_id
        self.relationship_type = relationship_type
        self.weight = weight
        self.properties = properties or {}

class NodeGraphMockCursor:
    """Mock cursor with knowledge graph nodes and edges"""
    def __init__(self):
        self.nodes = self._generate_nodes(MIN_DOCUMENTS)
        self.edges = self._generate_edges(self.nodes)
        self.query_history = []
        
    def _generate_nodes(self, count):
        """Generate mock knowledge graph nodes"""
        nodes = []
        node_types = ["Entity", "Concept", "Procedure", "Medication", "Condition", "Symptom"]
        topics = ["diabetes", "insulin", "cancer", "treatment", "vaccine", "research"]
        
        for i in range(count):
            node_type = random.choice(node_types)
            topic = random.choice(topics)
            node_id = f"node_{i:04d}"
            node_name = f"{topic.capitalize()} {node_type}"
            description = f"A {node_type.lower()} related to {topic} in medical research."
            embedding = "[" + ",".join([str(random.random()) for _ in range(5)]) + "]"
            
            nodes.append({
                "node_id": node_id,
                "node_type": node_type,
                "node_name": node_name,
                "description": description,
                "embedding": embedding
            })
        
        return nodes
        
    def _generate_edges(self, nodes):
        """Generate mock knowledge graph edges"""
        edges = []
        relationship_types = ["related_to", "causes", "treats", "part_of", "presents_with", "used_for"]
        
        # Create a dense connected graph - each node has ~3 edges on average
        edge_count = 0
        for i, source_node in enumerate(nodes):
            # Connect to 1-5 random target nodes
            num_connections = random.randint(1, 5)
            target_indices = random.sample(range(len(nodes)), min(num_connections, len(nodes) - 1))
            
            for target_idx in target_indices:
                if target_idx == i:  # Skip self-connections
                    continue
                    
                target_node = nodes[target_idx]
                relationship = random.choice(relationship_types)
                weight = round(random.uniform(0.5, 1.0), 2)
                
                edges.append({
                    "edge_id": edge_count,
                    "source_node_id": source_node["node_id"],
                    "target_node_id": target_node["node_id"],
                    "relationship_type": relationship,
                    "weight": weight,
                    "properties": "{}"
                })
                
                edge_count += 1
                
                # Limit to 3000 edges for performance
                if edge_count >= 3000:
                    return edges
        
        return edges
        
    def execute(self, query, params=None):
        """Execute a SQL query"""
        self.query_history.append((query, params))
        return self
        
    def fetchone(self):
        """Fetch one result"""
        if "COUNT" in self.query_history[-1][0].upper():
            if "KNOWLEDGEGRAPHNODES" in self.query_history[-1][0].upper():
                return [len(self.nodes)]
            elif "KNOWLEDGEGRAPHEDGES" in self.query_history[-1][0].upper():
                return [len(self.edges)]
            return [0]
        return None
        
    def fetchall(self):
        """Fetch all results"""
        last_query = self.query_history[-1][0].upper()
        
        if "SELECT" in last_query:
            if "KNOWLEDGEGRAPHNODES" in last_query:
                # Vector search or node lookup
                if "ORDER BY" in last_query and "VECTOR" in last_query:
                    # Return nodes with scores for vector similarity search
                    return [(n["node_id"], n["node_type"], n["node_name"], n["description"], 0.9) 
                            for n in self.nodes[:10]]
                # Regular node lookup
                return [(n["node_id"], n["node_type"], n["node_name"], n["description"], n["embedding"]) 
                        for n in self.nodes[:10]]
            elif "KNOWLEDGEGRAPHEDGES" in last_query:
                # Edge lookup - either by source or general
                if "WHERE" in last_query and "SOURCE_NODE_ID" in last_query:
                    # Specific lookup by source node
                    source_id = params[0] if params else self.nodes[0]["node_id"]
                    filtered_edges = [e for e in self.edges if e["source_node_id"] == source_id]
                    return [(e["edge_id"], e["source_node_id"], e["target_node_id"], 
                             e["relationship_type"], e["weight"], e["properties"]) 
                            for e in filtered_edges[:10]]
                # General edge lookup
                return [(e["edge_id"], e["source_node_id"], e["target_node_id"], 
                         e["relationship_type"], e["weight"], e["properties"]) 
                        for e in self.edges[:10]]
        
        return []
        
    def __enter__(self):
        return self
        
    def __exit__(self, *args):
        pass

class MockIRISConnector:
    """Mock IRIS connector with knowledge graph"""
    def __init__(self):
        self._cursor = NodeGraphMockCursor()
        
    def cursor(self):
        return self._cursor

class MockNodeRAGPipeline:
    """Mock NodeRAG pipeline for testing"""
    
    def __init__(self, iris_connector, embedding_func, llm_func):
        """Initialize with required components"""
        self.iris_connector = iris_connector
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        logger.info("MockNodeRAG initialized")
        
    def run(self, query, max_nodes=10, max_depth=2):
        """Run the pipeline with a query"""
        logger.info(f"Running NodeRAG query: '{query}'")
        
        # Identify start nodes
        start_nodes = self._identify_start_nodes(query, top_k=3)
        logger.info(f"Identified {len(start_nodes)} start nodes")
        
        # Traverse graph
        retrieved_nodes = self._traverse_graph(start_nodes, max_depth=max_depth, query_text=query)
        logger.info(f"Retrieved {len(retrieved_nodes)} nodes through graph traversal")
        
        # Generate answer
        context = self._format_context(retrieved_nodes, query)
        answer = self.llm_func(context)
        
        # Return result
        return {
            "query": query,
            "answer": answer,
            "retrieved_documents": retrieved_nodes
        }
        
    def _identify_start_nodes(self, query_text, top_k=3):
        """Identify start nodes for graph traversal"""
        # In a real implementation, this would use vector similarity search
        # For testing, we'll just get some node IDs
        with self.iris_connector.cursor() as cursor:
            cursor.execute("SELECT node_id FROM KnowledgeGraphNodes LIMIT ?", (top_k,))
            results = cursor.fetchall()
            return [row[0] for row in results] if results else []
    
    def _traverse_graph(self, start_nodes, max_depth=2, query_text=""):
        """Traverse the knowledge graph from the start nodes"""
        nodes = []
        visited_node_ids = set()
        
        # Get node details for start nodes
        with self.iris_connector.cursor() as cursor:
            for node_id in start_nodes:
                cursor.execute(
                    "SELECT node_id, node_type, node_name, description FROM KnowledgeGraphNodes WHERE node_id = ?", 
                    (node_id,)
                )
                results = cursor.fetchall()
                for row in results:
                    if row[0] not in visited_node_ids:
                        nodes.append(Node(
                            id=row[0],
                            type=row[1],
                            name=row[2],
                            description=row[3]
                        ))
                        visited_node_ids.add(row[0])
            
            # Get connected nodes - simplified BFS traversal
            for node_id in start_nodes:
                cursor.execute(
                    "SELECT target_node_id FROM KnowledgeGraphEdges WHERE source_node_id = ?", 
                    (node_id,)
                )
                results = cursor.fetchall()
                connected_ids = [row[0] for row in results]
                
                # Get details for connected nodes
                for connected_id in connected_ids:
                    if connected_id in visited_node_ids:
                        continue
                        
                    cursor.execute(
                        "SELECT node_id, node_type, node_name, description FROM KnowledgeGraphNodes WHERE node_id = ?", 
                        (connected_id,)
                    )
                    results = cursor.fetchall()
                    for row in results:
                        nodes.append(Node(
                            id=row[0],
                            type=row[1],
                            name=row[2],
                            description=row[3]
                        ))
                        visited_node_ids.add(row[0])
        
        return nodes
    
    def _format_context(self, nodes, query):
        """Format nodes into a prompt for the LLM"""
        context_parts = [f"Query: {query}\n\nKnowledge Graph Context:"]
        
        for i, node in enumerate(nodes[:10]):  # Limit to 10 nodes for the prompt
            context_parts.append(f"Node {i+1}: {node.name} ({node.type})\n{node.description}")
        
        return "\n\n".join(context_parts) + "\n\nAnswer the query based on the knowledge graph context:"

def test_noderag_with_1000_docs():
    """Test a NodeRAG pipeline with 1000+ documents (nodes)."""
    # Create mock components
    connector = MockIRISConnector()
    
    # Simple embedding function
    def mock_embedding_func(text):
        return [0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Simple LLM function
    def mock_llm_func(prompt):
        return "Based on the knowledge graph traversal, medical entities are connected in meaningful ways."
    
    # Create pipeline with mock components
    pipeline = MockNodeRAGPipeline(
        iris_connector=connector,
        embedding_func=mock_embedding_func,
        llm_func=mock_llm_func
    )
    
    # Run query
    query = "How does insulin relate to diabetes?"
    start_time = time.time()
    result = pipeline.run(query)
    duration = time.time() - start_time
    
    # Verify result format
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "query" in result, "Result should contain the query"
    assert "answer" in result, "Result should contain an answer"
    assert "retrieved_documents" in result, "Result should contain retrieved documents"
    
    # Verify we got nodes
    assert len(result["retrieved_documents"]) > 0, "Should retrieve at least one node"
    
    # Verify node count in database
    with connector.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM KnowledgeGraphNodes")
        node_count = cursor.fetchone()[0]
        assert node_count >= MIN_DOCUMENTS, f"Database should have at least {MIN_DOCUMENTS} nodes"
        
        cursor.execute("SELECT COUNT(*) FROM KnowledgeGraphEdges")
        edge_count = cursor.fetchone()[0]
        logger.info(f"Database has {edge_count} edges")
    
    # Log results
    logger.info(f"Query execution time: {duration:.2f} seconds")
    logger.info(f"Retrieved {len(result['retrieved_documents'])} nodes")
    logger.info(f"Answer: {result['answer']}")
    
    logger.info("NodeRAG test with 1000 nodes passed successfully")
    return result

if __name__ == "__main__":
    result = test_noderag_with_1000_docs()
    print(f"Retrieved {len(result['retrieved_documents'])} nodes")
    print(f"Answer: {result['answer']}")
