#!/usr/bin/env python3
"""
GraphRAG Visualization Demo Script

This script demonstrates the interactive knowledge graph visualization capabilities
of the GraphRAG pipeline, including Plotly, D3.js, and traversal path visualizations.
"""

import os
import sys
import logging
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from iris_rag.pipelines.graphrag_merged import GraphRAGPipeline
from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.models import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GraphVisualizationDemo:
    """Demo class for GraphRAG visualization capabilities."""
    
    def __init__(self):
        """Initialize the demo with GraphRAG pipeline."""
        logger.info("Initializing GraphRAG Visualization Demo")
        
        try:
            # Initialize managers
            self.config_manager = ConfigurationManager()
            self.connection_manager = ConnectionManager(self.config_manager)
            
            # Initialize GraphRAG pipeline with visualization support
            self.graphrag = GraphRAGPipeline(
                connection_manager=self.connection_manager,
                config_manager=self.config_manager,
                llm_func=self._dummy_llm  # Simple LLM mock for demo
            )
            
            # Create output directory for visualizations
            self.output_dir = Path("outputs/graph_visualizations")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Demo initialized. Visualization enabled: {self.graphrag.visualization_enabled}")
            
        except Exception as e:
            logger.error(f"Failed to initialize demo: {e}")
            raise
    
    def _dummy_llm(self, prompt: str) -> str:
        """Simple LLM mock for demonstration purposes."""
        return f"Demo response based on the provided context. Query: {prompt[:100]}..."
    
    def check_knowledge_graph_status(self):
        """Check if the knowledge graph has entities and relationships."""
        logger.info("Checking knowledge graph status...")
        
        try:
            connection = self.connection_manager.get_connection()
            cursor = connection.cursor()
            
            # Check entities
            cursor.execute("SELECT COUNT(*) FROM RAG.Entities")
            entity_count = cursor.fetchone()[0]
            
            # Check relationships
            cursor.execute("SELECT COUNT(*) FROM RAG.EntityRelationships")
            relationship_count = cursor.fetchone()[0]
            
            # Check entity types distribution
            cursor.execute("""
                SELECT entity_type, COUNT(*) as count 
                FROM RAG.Entities 
                GROUP BY entity_type 
                ORDER BY count DESC 
                LIMIT 10
            """)
            entity_types = cursor.fetchall()
            
            cursor.close()
            
            logger.info(f"Knowledge Graph Status:")
            logger.info(f"  - Total Entities: {entity_count}")
            logger.info(f"  - Total Relationships: {relationship_count}")
            logger.info(f"  - Entity Type Distribution:")
            for entity_type, count in entity_types:
                logger.info(f"    * {entity_type}: {count}")
            
            if entity_count == 0:
                logger.warning("Knowledge graph is empty. Load documents first.")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to check knowledge graph status: {e}")
            return False
    
    def load_sample_documents(self):
        """Load sample documents if knowledge graph is empty."""
        logger.info("Loading sample documents...")
        
        sample_docs = [
            Document(
                page_content="""
                Metformin is a medication used to treat type 2 diabetes. It works by decreasing glucose 
                production in the liver and improving insulin sensitivity. Common side effects include 
                nausea, diarrhea, and stomach upset. Metformin interacts with vitamin B12 absorption 
                and may cause lactic acidosis in rare cases. It is often prescribed alongside other 
                diabetes medications like insulin or sulfonylureas.
                """,
                metadata={"title": "Metformin Overview", "source": "medical_knowledge"},
                id="doc_metformin_1"
            ),
            Document(
                page_content="""
                Type 2 diabetes is a chronic condition that affects blood sugar regulation. The pancreas 
                produces insulin, but the body becomes resistant to its effects. Risk factors include 
                obesity, family history, and sedentary lifestyle. Treatment options include lifestyle 
                changes, oral medications like metformin, and insulin therapy. Complications can include 
                heart disease, kidney damage, and nerve damage.
                """,
                metadata={"title": "Type 2 Diabetes", "source": "medical_knowledge"},
                id="doc_diabetes_1"
            ),
            Document(
                page_content="""
                Insulin is a hormone produced by beta cells in the pancreas. It regulates blood glucose 
                levels by facilitating cellular glucose uptake. In diabetes, insulin production is 
                impaired or the body becomes resistant to insulin. Insulin therapy involves injecting 
                synthetic insulin to manage blood sugar. Types include rapid-acting, long-acting, and 
                intermediate-acting insulin preparations.
                """,
                metadata={"title": "Insulin Function", "source": "medical_knowledge"},
                id="doc_insulin_1"
            ),
            Document(
                page_content="""
                Glucose is the primary source of energy for cells. Blood glucose levels are tightly 
                regulated by hormones including insulin and glucagon. After eating, blood glucose rises 
                and insulin is released to promote glucose uptake by muscles and liver. During fasting, 
                glucagon stimulates glucose production by the liver. Dysregulation of this system leads 
                to diabetes.
                """,
                metadata={"title": "Glucose Regulation", "source": "medical_knowledge"},
                id="doc_glucose_1"
            )
        ]
        
        try:
            self.graphrag.load_documents("", documents=sample_docs)
            logger.info(f"Successfully loaded {len(sample_docs)} sample documents")
            return True
        except Exception as e:
            logger.error(f"Failed to load sample documents: {e}")
            return False
    
    def demo_plotly_visualization(self):
        """Demonstrate Plotly interactive graph visualization."""
        logger.info("=" * 60)
        logger.info("DEMO: Plotly Interactive Graph Visualization")
        logger.info("=" * 60)
        
        query = "What are the effects of metformin on diabetes?"
        
        try:
            result = self.graphrag.query(
                query, 
                top_k=5,
                visualize=True,
                visualization_type="plotly"
            )
            
            if "visualization" in result:
                output_file = self.output_dir / "plotly_graph_demo.html"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(result["visualization"])
                
                logger.info(f"Query: {query}")
                logger.info(f"Retrieved {result['metadata']['num_retrieved']} documents")
                logger.info(f"Execution time: {result['metadata']['processing_time_ms']:.1f}ms")
                logger.info(f"Plotly visualization saved to: {output_file}")
                
                if result["metadata"].get("step_timings_ms"):
                    timings = result["metadata"]["step_timings_ms"]
                    logger.info("Step timings:")
                    for step, time_ms in timings.items():
                        logger.info(f"  - {step}: {time_ms:.1f}ms")
            else:
                logger.warning("No visualization generated")
                
        except Exception as e:
            logger.error(f"Plotly demo failed: {e}")
    
    def demo_d3_visualization(self):
        """Demonstrate D3.js force-directed graph visualization."""
        logger.info("=" * 60)
        logger.info("DEMO: D3.js Force-Directed Graph Visualization")
        logger.info("=" * 60)
        
        query = "How does insulin interact with glucose regulation?"
        
        try:
            result = self.graphrag.query(
                query, 
                top_k=5,
                visualize=True,
                visualization_type="d3"
            )
            
            if "visualization" in result:
                output_file = self.output_dir / "d3_graph_demo.html"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(result["visualization"])
                
                logger.info(f"Query: {query}")
                logger.info(f"Retrieved {result['metadata']['num_retrieved']} documents")
                logger.info(f"Execution time: {result['metadata']['processing_time_ms']:.1f}ms")
                logger.info(f"D3.js visualization saved to: {output_file}")
                
                if result.get("answer"):
                    logger.info(f"Generated answer: {result['answer'][:200]}...")
            else:
                logger.warning("No visualization generated")
                
        except Exception as e:
            logger.error(f"D3.js demo failed: {e}")
    
    def demo_traversal_visualization(self):
        """Demonstrate traversal path visualization."""
        logger.info("=" * 60)
        logger.info("DEMO: Traversal Path Visualization")
        logger.info("=" * 60)
        
        query = "What medications treat diabetes and their side effects?"
        
        try:
            result = self.graphrag.query(
                query, 
                top_k=5,
                visualize=True,
                visualization_type="traversal"
            )
            
            if "visualization" in result:
                output_file = self.output_dir / "traversal_path_demo.html"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(result["visualization"])
                
                logger.info(f"Query: {query}")
                logger.info(f"Retrieved {result['metadata']['num_retrieved']} documents")
                logger.info(f"Execution time: {result['metadata']['processing_time_ms']:.1f}ms")
                logger.info(f"Retrieval method: {result['metadata']['retrieval_method']}")
                logger.info(f"Traversal visualization saved to: {output_file}")
                
                # Show database execution stats
                if "db_exec_count" in result["metadata"]:
                    logger.info(f"Database queries executed: {result['metadata']['db_exec_count']}")
            else:
                logger.warning("No visualization generated")
                
        except Exception as e:
            logger.error(f"Traversal demo failed: {e}")
    
    def demo_export_capabilities(self):
        """Demonstrate graph export capabilities."""
        logger.info("=" * 60)
        logger.info("DEMO: Graph Export Capabilities")
        logger.info("=" * 60)
        
        if not self.graphrag.visualization_enabled:
            logger.warning("Visualization not enabled, skipping export demo")
            return
        
        query = "Relationship between diabetes and insulin"
        
        try:
            # Run query to build traversal data
            result = self.graphrag.query(query, top_k=3)
            
            if self.graphrag.last_traversal_data:
                # Build graph from traversal data
                graph = self.graphrag.graph_visualizer.build_graph_from_traversal(
                    self.graphrag.last_traversal_data["seed_entities"],
                    self.graphrag.last_traversal_data["traversal_result"]
                )
                
                # Export to GraphML for Gephi/Cytoscape
                graphml_file = self.output_dir / "exported_graph.graphml"
                self.graphrag.graph_visualizer.export_to_graphml(graph, str(graphml_file))
                
                logger.info(f"Graph exported to GraphML: {graphml_file}")
                logger.info(f"Graph statistics:")
                logger.info(f"  - Nodes: {graph.number_of_nodes()}")
                logger.info(f"  - Edges: {graph.number_of_edges()}")
                logger.info(f"  - Seed entities: {len([n for n, d in graph.nodes(data=True) if d.get('is_seed', False)])}")
            else:
                logger.warning("No traversal data available for export")
                
        except Exception as e:
            logger.error(f"Export demo failed: {e}")
    
    def run_performance_comparison(self):
        """Compare query performance with and without visualization."""
        logger.info("=" * 60)
        logger.info("DEMO: Performance Comparison")
        logger.info("=" * 60)
        
        query = "Effects of metformin on blood glucose"
        iterations = 3
        
        # Test without visualization
        times_no_viz = []
        for i in range(iterations):
            start_time = time.perf_counter()
            result = self.graphrag.query(query, top_k=5, visualize=False)
            end_time = time.perf_counter()
            times_no_viz.append((end_time - start_time) * 1000)
        
        # Test with visualization
        times_with_viz = []
        for i in range(iterations):
            start_time = time.perf_counter()
            result = self.graphrag.query(query, top_k=5, visualize=True, visualization_type="plotly")
            end_time = time.perf_counter()
            times_with_viz.append((end_time - start_time) * 1000)
        
        avg_no_viz = sum(times_no_viz) / len(times_no_viz)
        avg_with_viz = sum(times_with_viz) / len(times_with_viz)
        
        logger.info(f"Performance Comparison ({iterations} iterations):")
        logger.info(f"  - Without visualization: {avg_no_viz:.1f}ms (avg)")
        logger.info(f"  - With visualization: {avg_with_viz:.1f}ms (avg)")
        logger.info(f"  - Overhead: {avg_with_viz - avg_no_viz:.1f}ms ({((avg_with_viz/avg_no_viz - 1) * 100):.1f}%)")
    
    def run_all_demos(self):
        """Run all visualization demos."""
        logger.info("Starting GraphRAG Visualization Demo Suite")
        logger.info(f"Output directory: {self.output_dir.absolute()}")
        
        # Check knowledge graph status
        if not self.check_knowledge_graph_status():
            logger.info("Loading sample documents...")
            if not self.load_sample_documents():
                logger.error("Failed to load sample documents. Exiting.")
                return
            
            # Recheck status
            if not self.check_knowledge_graph_status():
                logger.error("Knowledge graph still empty after loading documents. Exiting.")
                return
        
        # Run visualization demos
        try:
            self.demo_plotly_visualization()
            time.sleep(1)  # Brief pause between demos
            
            self.demo_d3_visualization()
            time.sleep(1)
            
            self.demo_traversal_visualization()
            time.sleep(1)
            
            self.demo_export_capabilities()
            time.sleep(1)
            
            self.run_performance_comparison()
            
        except Exception as e:
            logger.error(f"Demo execution failed: {e}")
            
        logger.info("=" * 60)
        logger.info("DEMO COMPLETE")
        logger.info("=" * 60)
        logger.info("Generated files:")
        for file in self.output_dir.glob("*.html"):
            logger.info(f"  - {file.name}")
        for file in self.output_dir.glob("*.graphml"):
            logger.info(f"  - {file.name}")
        logger.info(f"Open the HTML files in a web browser to view the interactive visualizations.")


def main():
    """Main demo function."""
    try:
        demo = GraphVisualizationDemo()
        demo.run_all_demos()
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()