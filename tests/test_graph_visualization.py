"""
Comprehensive test suite for GraphRAG visualization capabilities.

Tests cover graph building, visualization generation, pipeline integration,
and error handling for the GraphVisualizer system.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch, mock_open
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple

import networkx as nx

from iris_rag.visualization.graph_visualizer import GraphVisualizer, GraphVisualizationException
from iris_rag.pipelines.graphrag_merged import GraphRAGPipeline
from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.models import Document


class TestGraphVisualizer(unittest.TestCase):
    """Test cases for the GraphVisualizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_connection_manager = Mock(spec=ConnectionManager)
        self.mock_config_manager = Mock(spec=ConfigurationManager)
        
        # Mock configuration
        self.mock_config_manager.get.return_value = {
            "default_node_size": 20,
            "default_edge_width": 2,
            "layout_algorithm": "spring"
        }
        
        # Mock database connection
        self.mock_connection = Mock()
        self.mock_cursor = Mock()
        self.mock_connection.cursor.return_value = self.mock_cursor
        self.mock_connection_manager.get_connection.return_value = self.mock_connection
        
        self.visualizer = GraphVisualizer(
            connection_manager=self.mock_connection_manager,
            config_manager=self.mock_config_manager
        )
    
    def test_initialization(self):
        """Test GraphVisualizer initialization."""
        self.assertIsNotNone(self.visualizer)
        self.assertEqual(self.visualizer.default_node_size, 20)
        self.assertEqual(self.visualizer.default_edge_width, 2)
        self.assertEqual(self.visualizer.layout_algorithm, "spring")
        self.assertIn("DISEASE", self.visualizer.entity_colors)
        self.assertIn("DRUG", self.visualizer.entity_colors)
    
    def test_build_graph_from_query_empty_inputs(self):
        """Test building graph with empty inputs."""
        # Mock empty entity details
        self.mock_cursor.fetchall.return_value = []
        
        graph = self.visualizer.build_graph_from_query(
            "test query", 
            [], 
            []
        )
        
        self.assertIsInstance(graph, nx.Graph)
        self.assertEqual(graph.number_of_nodes(), 0)
        self.assertEqual(graph.number_of_edges(), 0)
        self.assertEqual(graph.graph['query'], "test query")
    
    def test_build_graph_from_query_with_data(self):
        """Test building graph with actual entity and relationship data."""
        # Mock entity details
        self.mock_cursor.fetchall.return_value = [
            ("entity1", "Metformin", "DRUG", 3),
            ("entity2", "Diabetes", "DISEASE", 5),
            ("entity3", "Insulin", "DRUG", 2)
        ]
        
        entities = ["entity1", "entity2", "entity3"]
        relationships = [
            {
                "source_entity_id": "entity1",
                "target_entity_id": "entity2", 
                "relationship_type": "treats",
                "confidence_score": 0.8,
                "source_doc_id": "doc1"
            },
            {
                "source_entity_id": "entity2",
                "target_entity_id": "entity3",
                "relationship_type": "requires",
                "confidence_score": 0.9,
                "source_doc_id": "doc2"
            }
        ]
        
        graph = self.visualizer.build_graph_from_query(
            "What treats diabetes?",
            entities,
            relationships
        )
        
        self.assertEqual(graph.number_of_nodes(), 3)
        self.assertEqual(graph.number_of_edges(), 2)
        
        # Check node attributes
        self.assertIn("entity1", graph.nodes())
        self.assertEqual(graph.nodes["entity1"]["name"], "Metformin")
        self.assertEqual(graph.nodes["entity1"]["entity_type"], "DRUG")
        
        # Check edge attributes
        edge_data = graph.edges["entity1", "entity2"]
        self.assertEqual(edge_data["relationship_type"], "treats")
        self.assertEqual(edge_data["confidence"], 0.8)
    
    def test_build_graph_from_traversal(self):
        """Test building graph from traversal results."""
        # Mock entity details
        self.mock_cursor.fetchall.side_effect = [
            # First call: entity details
            [
                ("seed1", "Heart Disease", "DISEASE", 4),
                ("entity1", "Statin", "DRUG", 2),
                ("entity2", "Cholesterol", "SYMPTOM", 3)
            ],
            # Second call: relationships
            [
                ("seed1", "entity1", "treated_by", 0.9, "doc1"),
                ("entity1", "entity2", "reduces", 0.8, "doc2")
            ]
        ]
        
        seed_entities = [("seed1", "Heart Disease", 0.9)]
        traversal_result = {"seed1", "entity1", "entity2"}
        
        graph = self.visualizer.build_graph_from_traversal(
            seed_entities,
            traversal_result
        )
        
        self.assertEqual(graph.number_of_nodes(), 3)
        self.assertEqual(graph.number_of_edges(), 2)
        
        # Check seed entity marking
        self.assertTrue(graph.nodes["seed1"]["is_seed"])
        self.assertFalse(graph.nodes["entity1"]["is_seed"])
        
        # Check graph metadata
        self.assertEqual(graph.graph["seed_entities"], ["seed1"])
        self.assertTrue(graph.graph["traversal_complete"])
    
    @patch('plotly.offline.plot')
    @patch('plotly.graph_objects.Figure')
    def test_generate_plotly_visualization(self, mock_figure, mock_plot):
        """Test Plotly visualization generation."""
        # Create a simple test graph
        graph = nx.Graph()
        graph.add_node("node1", name="Test Node", entity_type="DISEASE", color="#FF0000")
        graph.add_node("node2", name="Another Node", entity_type="DRUG", color="#00FF00")
        graph.add_edge("node1", "node2", relationship_type="treats", confidence=0.8)
        graph.graph["query"] = "test query"
        
        # Mock plotly components
        mock_plot.return_value = "<div>Mock Plotly HTML</div>"
        
        result = self.visualizer.generate_plotly_visualization(graph)
        
        self.assertEqual(result, "<div>Mock Plotly HTML</div>")
        mock_plot.assert_called_once()
    
    def test_generate_plotly_visualization_import_error(self):
        """Test Plotly visualization when plotly is not installed."""
        with patch('iris_rag.visualization.graph_visualizer.go', side_effect=ImportError("No module named 'plotly'")):
            graph = nx.Graph()
            graph.graph["query"] = "test"
            
            with self.assertRaises(GraphVisualizationException) as context:
                self.visualizer.generate_plotly_visualization(graph)
            
            self.assertIn("Plotly not installed", str(context.exception))
    
    def test_export_to_graphml(self):
        """Test exporting graph to GraphML format."""
        # Create test graph
        graph = nx.Graph()
        graph.add_node("node1", name="Test", entity_type="DISEASE")
        graph.add_edge("node1", "node2", relationship_type="connects")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "test_graph.graphml")
            
            self.visualizer.export_to_graphml(graph, output_file)
            
            # Verify file was created
            self.assertTrue(os.path.exists(output_file))
            
            # Verify file contains GraphML content
            with open(output_file, 'r') as f:
                content = f.read()
                self.assertIn('<?xml version="1.0" encoding="UTF-8"?>', content)
                self.assertIn('<graphml', content)
    
    def test_fetch_entity_details_empty(self):
        """Test fetching entity details with empty input."""
        result = self.visualizer._fetch_entity_details([])
        self.assertEqual(result, {})
    
    def test_fetch_entity_details_with_data(self):
        """Test fetching entity details with actual data."""
        self.mock_cursor.fetchall.return_value = [
            ("entity1", "Metformin", "DRUG", 3),
            ("entity2", "Diabetes", "DISEASE", 5)
        ]
        
        result = self.visualizer._fetch_entity_details(["entity1", "entity2"])
        
        expected = {
            "entity1": {
                "entity_name": "Metformin",
                "entity_type": "DRUG", 
                "doc_count": 3
            },
            "entity2": {
                "entity_name": "Diabetes",
                "entity_type": "DISEASE",
                "doc_count": 5
            }
        }
        
        self.assertEqual(result, expected)
    
    def test_fetch_entity_details_database_error(self):
        """Test handling database errors in entity details fetch."""
        self.mock_cursor.execute.side_effect = Exception("Database error")
        
        result = self.visualizer._fetch_entity_details(["entity1"])
        self.assertEqual(result, {})
    
    def test_fetch_relationships_empty(self):
        """Test fetching relationships with empty input."""
        result = self.visualizer._fetch_relationships([])
        self.assertEqual(result, [])
    
    def test_fetch_relationships_with_data(self):
        """Test fetching relationships with actual data."""
        self.mock_cursor.fetchall.return_value = [
            ("entity1", "entity2", "treats", 0.8, "doc1"),
            ("entity2", "entity3", "causes", 0.9, "doc2")
        ]
        
        result = self.visualizer._fetch_relationships(["entity1", "entity2", "entity3"])
        
        expected = [
            {
                "source_entity_id": "entity1",
                "target_entity_id": "entity2",
                "relationship_type": "treats",
                "confidence_score": 0.8,
                "source_doc_id": "doc1"
            },
            {
                "source_entity_id": "entity2",
                "target_entity_id": "entity3", 
                "relationship_type": "causes",
                "confidence_score": 0.9,
                "source_doc_id": "doc2"
            }
        ]
        
        self.assertEqual(result, expected)


class TestGraphRAGVisualizationIntegration(unittest.TestCase):
    """Test integration of visualization with GraphRAG pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_connection_manager = Mock(spec=ConnectionManager)
        self.mock_config_manager = Mock(spec=ConfigurationManager)
        
        # Mock configuration
        self.mock_config_manager.get.side_effect = lambda key, default=None: {
            "pipelines:graphrag": {
                "default_top_k": 10,
                "max_depth": 2,
                "max_entities": 50,
                "enable_vector_fallback": False
            },
            "visualization": {
                "default_node_size": 20,
                "default_edge_width": 2,
                "layout_algorithm": "spring"
            }
        }.get(key, default)
        
        # Mock database connection and cursor
        self.mock_connection = Mock()
        self.mock_cursor = Mock()
        self.mock_connection.cursor.return_value = self.mock_cursor
        self.mock_connection_manager.get_connection.return_value = self.mock_connection
        
        # Mock LLM function
        self.mock_llm = Mock(return_value="Test answer")
    
    @patch('iris_rag.pipelines.graphrag_merged.EntityExtractionService')
    def test_pipeline_initialization_with_visualization(self, mock_entity_service):
        """Test GraphRAG pipeline initialization with visualization support."""
        mock_entity_service.side_effect = Exception("Service unavailable")
        
        pipeline = GraphRAGPipeline(
            connection_manager=self.mock_connection_manager,
            config_manager=self.mock_config_manager,
            llm_func=self.mock_llm
        )
        
        # Check that visualization is properly initialized
        self.assertIsNotNone(pipeline.graph_visualizer)
        self.assertTrue(pipeline.visualization_enabled)
        self.assertEqual(pipeline.last_traversal_data, {})
    
    @patch('iris_rag.pipelines.graphrag_merged.EntityExtractionService')
    def test_query_with_visualization_disabled(self, mock_entity_service):
        """Test query execution without visualization."""
        mock_entity_service.side_effect = Exception("Service unavailable")
        
        # Mock knowledge graph validation
        self.mock_cursor.fetchone.return_value = [10]  # 10 entities exist
        
        # Mock query execution components
        self.mock_cursor.fetchall.side_effect = [
            [("entity1", "Test Entity", "DISEASE")],  # Seed entities
            [("entity2",)],  # Traversal results
            [("doc1", "Test content", "Test title")]  # Documents
        ]
        
        pipeline = GraphRAGPipeline(
            connection_manager=self.mock_connection_manager,
            config_manager=self.mock_config_manager,
            llm_func=self.mock_llm
        )
        
        result = pipeline.query("test query", visualize=False)
        
        self.assertIn("query", result)
        self.assertIn("answer", result)
        self.assertNotIn("visualization", result)
        self.assertFalse(result["metadata"].get("visualization_generated", False))
    
    @patch('iris_rag.pipelines.graphrag_merged.EntityExtractionService')
    @patch('plotly.offline.plot')
    def test_query_with_plotly_visualization(self, mock_plot, mock_entity_service):
        """Test query execution with Plotly visualization."""
        mock_entity_service.side_effect = Exception("Service unavailable")
        mock_plot.return_value = "<div>Plotly visualization</div>"
        
        # Mock knowledge graph validation
        self.mock_cursor.fetchone.return_value = [10]  # 10 entities exist
        
        # Mock query execution components
        self.mock_cursor.fetchall.side_effect = [
            [("entity1", "Test Entity", "DISEASE")],  # Seed entities
            [("entity2",)],  # Traversal results
            [("doc1", "Test content", "Test title")],  # Documents
            [("entity1", "Test Entity", "DISEASE", 1)],  # Entity details for visualization
            [("entity1", "entity2", "treats", 0.8, "doc1")]  # Relationships for visualization
        ]
        
        pipeline = GraphRAGPipeline(
            connection_manager=self.mock_connection_manager,
            config_manager=self.mock_config_manager,
            llm_func=self.mock_llm
        )
        
        result = pipeline.query("test query", visualize=True, visualization_type="plotly")
        
        self.assertIn("query", result)
        self.assertIn("answer", result)
        self.assertIn("visualization", result)
        self.assertTrue(result["metadata"].get("visualization_generated", False))
        self.assertEqual(result["visualization"], "<div>Plotly visualization</div>")
    
    @patch('iris_rag.pipelines.graphrag_merged.EntityExtractionService')
    def test_query_with_traversal_visualization(self, mock_entity_service):
        """Test query execution with traversal path visualization."""
        mock_entity_service.side_effect = Exception("Service unavailable")
        
        # Mock knowledge graph validation
        self.mock_cursor.fetchone.return_value = [10]  # 10 entities exist
        
        # Mock query execution components
        self.mock_cursor.fetchall.side_effect = [
            [("entity1", "Test Entity", "DISEASE")],  # Seed entities
            [("entity2",)],  # Traversal results
            [("doc1", "Test content", "Test title")]  # Documents
        ]
        
        pipeline = GraphRAGPipeline(
            connection_manager=self.mock_connection_manager,
            config_manager=self.mock_config_manager,
            llm_func=self.mock_llm
        )
        
        # Mock the template loading to avoid file system dependency
        with patch.object(pipeline.graph_visualizer, '_load_template') as mock_template:
            mock_template.return_value = "<html>Traversal visualization</html>"
            
            result = pipeline.query("test query", visualize=True, visualization_type="traversal")
            
            self.assertIn("visualization", result)
            self.assertTrue(result["metadata"].get("visualization_generated", False))
            self.assertEqual(result["visualization"], "<html>Traversal visualization</html>")
    
    @patch('iris_rag.pipelines.graphrag_merged.EntityExtractionService')
    def test_visualization_error_handling(self, mock_entity_service):
        """Test error handling when visualization fails."""
        mock_entity_service.side_effect = Exception("Service unavailable")
        
        # Mock knowledge graph validation
        self.mock_cursor.fetchone.return_value = [10]  # 10 entities exist
        
        # Mock query execution components
        self.mock_cursor.fetchall.side_effect = [
            [("entity1", "Test Entity", "DISEASE")],  # Seed entities
            [("entity2",)],  # Traversal results
            [("doc1", "Test content", "Test title")]  # Documents
        ]
        
        pipeline = GraphRAGPipeline(
            connection_manager=self.mock_connection_manager,
            config_manager=self.mock_config_manager,
            llm_func=self.mock_llm
        )
        
        # Force visualization error
        with patch.object(pipeline, '_generate_visualization', side_effect=Exception("Visualization failed")):
            result = pipeline.query("test query", visualize=True, visualization_type="plotly")
            
            self.assertIn("query", result)
            self.assertIn("answer", result)
            self.assertNotIn("visualization", result)
            self.assertFalse(result["metadata"].get("visualization_generated", False))
            self.assertIn("visualization_error", result["metadata"])
            self.assertEqual(result["metadata"]["visualization_error"], "Visualization failed")


class TestGraphVisualizationEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for graph visualization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_connection_manager = Mock(spec=ConnectionManager)
        self.mock_config_manager = Mock(spec=ConfigurationManager)
        self.mock_config_manager.get.return_value = {}
        
        self.visualizer = GraphVisualizer(
            connection_manager=self.mock_connection_manager,
            config_manager=self.mock_config_manager
        )
    
    def test_build_graph_invalid_relationships(self):
        """Test building graph with invalid relationship data."""
        # Mock connection and cursor
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = []
        mock_connection.cursor.return_value = mock_cursor
        self.mock_connection_manager.get_connection.return_value = mock_connection
        
        # Test with malformed relationship data
        invalid_relationships = [
            {"source_entity_id": "entity1"},  # Missing target
            {"target_entity_id": "entity2"},  # Missing source
            {}  # Empty relationship
        ]
        
        # Should not raise exception, just ignore invalid relationships
        graph = self.visualizer.build_graph_from_query(
            "test query",
            [],
            invalid_relationships
        )
        
        self.assertIsInstance(graph, nx.Graph)
        self.assertEqual(graph.number_of_edges(), 0)
    
    def test_empty_traversal_result(self):
        """Test building graph from empty traversal result."""
        # Mock connection and cursor
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = []
        mock_connection.cursor.return_value = mock_cursor
        self.mock_connection_manager.get_connection.return_value = mock_connection
        
        graph = self.visualizer.build_graph_from_traversal([], set())
        
        self.assertIsInstance(graph, nx.Graph)
        self.assertEqual(graph.number_of_nodes(), 0)
        self.assertEqual(graph.number_of_edges(), 0)
    
    def test_visualization_with_unknown_type(self):
        """Test visualization generation with unknown type."""
        graph = nx.Graph()
        
        with self.assertRaises(GraphVisualizationException):
            self.visualizer.generate_d3_visualization(graph)  # This should call _load_template
    
    def test_export_to_invalid_path(self):
        """Test exporting graph to invalid file path."""
        graph = nx.Graph()
        
        with self.assertRaises(GraphVisualizationException):
            self.visualizer.export_to_graphml(graph, "/invalid/path/file.graphml")


class TestVisualizationPerformance(unittest.TestCase):
    """Performance tests for visualization components."""
    
    def setUp(self):
        """Set up performance test fixtures."""
        self.mock_connection_manager = Mock(spec=ConnectionManager)
        self.mock_config_manager = Mock(spec=ConfigurationManager)
        self.mock_config_manager.get.return_value = {}
        
        # Mock connection and cursor
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        self.mock_connection_manager.get_connection.return_value = mock_connection
        
        self.visualizer = GraphVisualizer(
            connection_manager=self.mock_connection_manager,
            config_manager=self.mock_config_manager
        )
    
    def test_large_graph_building(self):
        """Test building visualization with large graph."""
        # Mock large entity dataset
        entities = [f"entity_{i}" for i in range(100)]
        relationships = []
        for i in range(50):
            relationships.append({
                "source_entity_id": f"entity_{i}",
                "target_entity_id": f"entity_{i+1}",
                "relationship_type": "connects",
                "confidence_score": 0.8,
                "source_doc_id": f"doc_{i}"
            })
        
        # Mock database response for entity details
        mock_cursor = self.mock_connection_manager.get_connection().cursor()
        mock_cursor.fetchall.return_value = [
            (f"entity_{i}", f"Entity {i}", "GENERIC", 1) for i in range(100)
        ]
        
        graph = self.visualizer.build_graph_from_query(
            "large graph test",
            entities,
            relationships
        )
        
        self.assertEqual(graph.number_of_nodes(), 100)
        self.assertEqual(graph.number_of_edges(), 50)
        
        # Graph should still be well-formed
        self.assertIsInstance(graph, nx.Graph)
        self.assertTrue(nx.is_connected(graph))


if __name__ == '__main__':
    unittest.main()