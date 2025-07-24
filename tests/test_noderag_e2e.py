from unittest.mock import patch

# Add project root to sys.path to allow imports
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.utils import get_embedding_func

from iris_rag.pipelines.noderag import NodeRAGPipeline

# Test Data for NodeRAG
# Document 1: Alpha Protocol
DOC1_ID = "noderag_doc_001"
DOC1_CONTENT = "The Alpha Protocol details project A. Section 1 discusses its goals. Section 2 covers the methodology, including the use of Gamma particles. Section 3 outlines the expected results."
DOC1_CHUNKS_DATA = [
    {"id": "noderag_chunk_001_01", "doc_id": DOC1_ID, "text": "Section 1 discusses its goals for Project A.", "index": 0},
    {"id": "noderag_chunk_001_02", "doc_id": DOC1_ID, "text": "Section 2 covers the methodology for Project A, including the use of Gamma particles.", "index": 1},
    {"id": "noderag_chunk_001_03", "doc_id": DOC1_ID, "text": "Section 3 outlines the expected results for Project A.", "index": 2}
]

# Document 2: Project B
DOC2_ID = "noderag_doc_002"
DOC2_CONTENT = "Project B is a follow-up to Project A. It aims to verify the results obtained using Gamma particles and explore Beta waves."
DOC2_CHUNKS_DATA = [
    {"id": "noderag_chunk_002_01", "doc_id": DOC2_ID, "text": "Project B is a follow-up to Project A.", "index": 0},
    {"id": "noderag_chunk_002_02", "doc_id": DOC2_ID, "text": "Project B aims to verify the results obtained using Gamma particles and explore Beta waves.", "index": 1}
]

# Document 3: Unrelated
DOC3_ID = "noderag_doc_003"
DOC3_CONTENT = "The Delta project focuses on renewable energy sources, primarily solar power."
DOC3_CHUNKS_DATA = [
    {"id": "noderag_chunk_003_01", "doc_id": DOC3_ID, "text": "The Delta project focuses on renewable energy sources.", "index": 0},
    {"id": "noderag_chunk_003_02", "doc_id": DOC3_ID, "text": "Primary focus of Delta is solar power.", "index": 1}
]

TEST_DOCS_DATA_NODERAG = [
    {"id": DOC1_ID, "title": "Alpha Protocol", "content": DOC1_CONTENT},
    {"id": DOC2_ID, "title": "Project B Details", "content": DOC2_CONTENT},
    {"id": DOC3_ID, "title": "Delta Project Overview", "content": DOC3_CONTENT},
]
TEST_DOC_IDS_NODERAG = [doc["id"] for doc in TEST_DOCS_DATA_NODERAG]

ALL_CHUNKS_DATA_NODERAG = DOC1_CHUNKS_DATA + DOC2_CHUNKS_DATA + DOC3_CHUNKS_DATA
TEST_CHUNK_IDS_NODERAG = [chunk["id"] for chunk in ALL_CHUNKS_DATA_NODERAG]


def setup_test_data_noderag(iris_connection, embedding_function):
    """Inserts test documents and chunks with embeddings into RAG.SourceDocuments and RAG.DocumentChunks."""
    cursor = iris_connection.cursor()
    
    # Insert SourceDocuments
    for doc_data in TEST_DOCS_DATA_NODERAG:
        doc_id = doc_data["id"]
        title = doc_data["title"]
        content = doc_data["content"]
        
        doc_embedding_vector = embedding_function([content])[0]
        embedding_str = f"[{','.join(map(str, doc_embedding_vector))}]"
        
        try:
            cursor.execute("SELECT doc_id FROM RAG.SourceDocuments WHERE doc_id = ?", (doc_id,))
            if cursor.fetchone() is None:
                cursor.execute(
                    "INSERT INTO RAG.SourceDocuments (doc_id, title, text_content, embedding) VALUES (?, ?, ?, ?)",
                    (doc_id, title, content, embedding_str)
                )
            else:
                cursor.execute(
                    "UPDATE RAG.SourceDocuments SET title = ?, text_content = ?, embedding = ? WHERE doc_id = ?",
                    (title, content, embedding_str, doc_id)
                )
        except Exception as e:
            print(f"Error inserting/updating source document {doc_id} for NodeRAG: {e}")
            raise
            
    # Insert DocumentChunks
    for chunk_data in ALL_CHUNKS_DATA_NODERAG:
        chunk_id = chunk_data["id"]
        doc_id = chunk_data["doc_id"]
        chunk_text = chunk_data["text"]
        chunk_index = chunk_data["index"]
        
        chunk_embedding_vector = embedding_function([chunk_text])[0]
        embedding_str = f"[{','.join(map(str, chunk_embedding_vector))}]"
        
        try:
            cursor.execute("SELECT chunk_id FROM RAG.DocumentChunks WHERE chunk_id = ?", (chunk_id,))
            if cursor.fetchone() is None:
                cursor.execute(
                    "INSERT INTO RAG.DocumentChunks (chunk_id, doc_id, chunk_text, chunk_index, embedding) VALUES (?, ?, ?, ?, ?)",
                    (chunk_id, doc_id, chunk_text, chunk_index, embedding_str)
                )
            else:
                cursor.execute(
                    "UPDATE RAG.DocumentChunks SET doc_id = ?, chunk_text = ?, chunk_index = ?, embedding = ? WHERE chunk_id = ?",
                    (doc_id, chunk_text, chunk_index, embedding_str, chunk_id)
                )
        except Exception as e:
            print(f"Error inserting/updating document chunk {chunk_id} for NodeRAG: {e}")
            raise

    iris_connection.commit()
    cursor.close()
    print(f"Setup NodeRAG: Ensured {len(TEST_DOCS_DATA_NODERAG)} documents and {len(ALL_CHUNKS_DATA_NODERAG)} chunks are present.")

def cleanup_test_data_noderag(iris_connection):
    """Removes test documents and chunks."""
    cursor = iris_connection.cursor()
    try:
        if TEST_CHUNK_IDS_NODERAG:
            chunk_placeholders = ','.join(['?' for _ in TEST_CHUNK_IDS_NODERAG])
            cursor.execute(f"DELETE FROM RAG.DocumentChunks WHERE chunk_id IN ({chunk_placeholders})", TEST_CHUNK_IDS_NODERAG)
            print(f"Cleanup NodeRAG: Deleted {cursor.rowcount} document chunks.")

        if TEST_DOC_IDS_NODERAG:
            doc_placeholders = ','.join(['?' for _ in TEST_DOC_IDS_NODERAG])
            cursor.execute(f"DELETE FROM RAG.SourceDocuments WHERE doc_id IN ({doc_placeholders})", TEST_DOC_IDS_NODERAG)
            print(f"Cleanup NodeRAG: Deleted {cursor.rowcount} source documents.")
        
        iris_connection.commit()
    except Exception as e:
        print(f"Error during NodeRAG cleanup: {e}")
        iris_connection.rollback()
    finally:
        cursor.close()

def mock_llm_for_noderag_test(prompt: str) -> str:
    """Mock LLM specifically for this NodeRAG test."""
    context_lower = prompt.lower()
    # print(f"Mock LLM NodeRAG received prompt context (first 500 chars):\n{context_lower[:500]}...")

    has_gamma_methodology = "gamma particles" in context_lower and "methodology for project a" in context_lower
    has_project_b_relation = "project b is a follow-up to project a" in context_lower or \
                             ("project b" in context_lower and "gamma particles" in context_lower and "verify" in context_lower)

    if has_gamma_methodology and has_project_b_relation:
        return "Project A's methodology included Gamma particles. Project B is a follow-up that aims to verify results from Gamma particles. (NodeRAG Test)"
    elif has_gamma_methodology:
        return "Project A's methodology involved Gamma particles. (NodeRAG Test)"
    elif has_project_b_relation:
        return "Project B is related to Project A and Gamma particles. (NodeRAG Test)"
    
    return "Based on the provided context, I cannot definitively answer the question. (NodeRAG Test)"


def test_noderag_e2e_relationship_query(iris_testcontainer_connection):
    """
    Tests the NodeRAG V2 pipeline's end-to-end flow, focusing on its ability
    to retrieve and use related nodes (documents and chunks) for answering.
    """
    real_embedding_function = get_embedding_func(mock=False) # Use real embeddings
    mock_llm_function = mock_llm_for_noderag_test

    try:
        print("Setting up NodeRAG test data in testcontainer...")
        setup_test_data_noderag(iris_testcontainer_connection, real_embedding_function)
        
        pipeline = NodeRAGPipeline(
            iris_connector=iris_testcontainer_connection,
            embedding_func=real_embedding_function,
            llm_func=mock_llm_function
        )

        query = "What was Project A's methodology regarding Gamma particles and how is Project B related?"
        
        # Expecting to retrieve chunk "noderag_chunk_001_02" (Gamma methodology)
        # and chunk "noderag_chunk_002_01" or "noderag_chunk_002_02" (Project B relation)
        # or potentially the full documents DOC1_ID, DOC2_ID if they score high enough.
        
        results = pipeline.run(query=query, top_k=3, similarity_threshold=0.1) # top_k for merged results

        print(f"NodeRAG Query: {results['query']}")
        print(f"NodeRAG Answer: {results['answer']}")
        retrieved_nodes_info = []
        for node in results.get("retrieved_nodes", []):
            node_type = node.get('type', 'unknown')
            node_id = node.get('id')
            metadata = node.get('metadata', {})
            score = metadata.get('similarity_score', 0)
            content_preview = node.get('content', '')[:100]
            retrieved_nodes_info.append(f"  - Type: {node_type}, ID: {node_id}, Score: {score:.4f}, Content: '{content_preview}...'")
        print(f"NodeRAG Retrieved Nodes ({len(results.get('retrieved_nodes', []))}):\n" + "\n".join(retrieved_nodes_info))

        assert "answer" in results
        assert "retrieved_nodes" in results # NodeRAG uses 'retrieved_nodes'
        
        retrieved_nodes = results["retrieved_nodes"]
        assert len(retrieved_nodes) > 0, "NodeRAG: No nodes were retrieved."
        # We asked for top_k=3 merged results
        assert len(retrieved_nodes) <= 3, f"NodeRAG: Expected up to 3 nodes, got {len(retrieved_nodes)}"

        retrieved_node_ids = [node['id'] for node in retrieved_nodes]
        
        # Check for specific key information providers
        # Chunk "noderag_chunk_001_02" (Project A methodology with Gamma)
        # Chunk "noderag_chunk_002_01" or "noderag_chunk_002_02" (Project B relation)
        # Or their parent documents DOC1_ID, DOC2_ID
        
        found_gamma_methodology_node = any(
            node_id == "noderag_chunk_001_02" or 
            (node_id == DOC1_ID and "gamma particles" in node.get('content', '').lower() and "methodology" in node.get('content','').lower())
            for node_id, node in zip(retrieved_node_ids, retrieved_nodes)
        )
        
        found_project_b_relation_node = any(
            node_id in ["noderag_chunk_002_01", "noderag_chunk_002_02"] or
            (node_id == DOC2_ID and "project a" in node.get('content', '').lower() and "follow-up" in node.get('content','').lower())
            for node_id, node in zip(retrieved_node_ids, retrieved_nodes)
        )

        assert found_gamma_methodology_node, \
            f"NodeRAG: Expected a node related to Gamma particle methodology (e.g., chunk 'noderag_chunk_001_02' or doc '{DOC1_ID}'). Got IDs: {retrieved_node_ids}"
        
        assert found_project_b_relation_node, \
            f"NodeRAG: Expected a node related to Project B's relation to A (e.g., chunks 'noderag_chunk_002_01/02' or doc '{DOC2_ID}'). Got IDs: {retrieved_node_ids}"

        answer_lower = results["answer"].lower()
        assert "gamma particles" in answer_lower
        assert "project a" in answer_lower
        assert "project b" in answer_lower
        assert "methodology" in answer_lower or "follow-up" in answer_lower or "verify" in answer_lower
        assert "(noderag test)" in answer_lower # To confirm mock was hit correctly

        # More specific check based on mock LLM logic
        expected_answer_keywords = ["project a's methodology included gamma particles", "project b is a follow-up"]
        assert all(keyword in answer_lower for keyword in expected_answer_keywords), \
            f"NodeRAG: Answer '{results['answer']}' did not contain all expected keywords based on mock LLM logic."

    finally:
        print("Cleaning up NodeRAG test data from testcontainer...")
        cleanup_test_data_noderag(iris_testcontainer_connection)