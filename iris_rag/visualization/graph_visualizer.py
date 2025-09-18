"""
GraphRAG Knowledge Graph Visualizer.

Provides interactive visualization capabilities for exploring knowledge graphs,
entity relationships, and traversal paths in the GraphRAG pipeline.
"""

import logging
import json
import time
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path

import networkx as nx
from ..core.connection import ConnectionManager
from ..config.manager import ConfigurationManager

logger = logging.getLogger(__name__)


class GraphVisualizationException(Exception):
    """Exception raised when graph visualization operations fail."""
    pass


class GraphVisualizer:
    """Interactive knowledge graph visualization for GraphRAG."""
    
    def __init__(self, connection_manager: Optional[ConnectionManager] = None,
                 config_manager: Optional[ConfigurationManager] = None):
        """
        Initialize the graph visualizer.
        
        Args:
            connection_manager: Database connection manager
            config_manager: Configuration manager for visualization settings
        """
        self.connection_manager = connection_manager or ConnectionManager()
        self.config_manager = config_manager or ConfigurationManager()
        self.graph_data = {}
        
        # Visualization configuration
        self.viz_config = self.config_manager.get("visualization", {})
        self.default_node_size = self.viz_config.get("default_node_size", 20)
        self.default_edge_width = self.viz_config.get("default_edge_width", 2)
        self.layout_algorithm = self.viz_config.get("layout_algorithm", "spring")
        
        # Color schemes for different entity types
        self.entity_colors = {
            "PERSON": "#FF6B6B",
            "DISEASE": "#4ECDC4", 
            "DRUG": "#45B7D1",
            "TREATMENT": "#96CEB4",
            "SYMPTOM": "#FECA57",
            "GENE": "#FF9FF3",
            "PROTEIN": "#54A0FF", 
            "ANATOMY": "#5F27CD",
            "PROCEDURE": "#00D2D3",
            "DEVICE": "#FF9F43",
            "ORGANIZATION": "#A55EEA",
            "LOCATION": "#26DE81",
            "KEYWORD": "#778CA3",
            "DEFAULT": "#95A5A6"
        }
        
        logger.info("GraphVisualizer initialized")

    def build_graph_from_query(self, query: str, entities: List[str], 
                              relationships: List[dict]) -> nx.Graph:
        """
        Build NetworkX graph from query results.
        
        Args:
            query: The original query text
            entities: List of entity names/IDs involved in the query
            relationships: List of relationship dictionaries
            
        Returns:
            NetworkX Graph object with nodes and edges
        """
        try:
            graph = nx.Graph()
            
            # Add query metadata
            graph.graph['query'] = query
            graph.graph['created_at'] = time.time()
            
            # Get entity details from database
            entity_details = self._fetch_entity_details(entities)
            
            # Add nodes for entities
            for entity_id, entity_info in entity_details.items():
                graph.add_node(
                    entity_id,
                    name=entity_info.get('entity_name', entity_id),
                    entity_type=entity_info.get('entity_type', 'UNKNOWN'),
                    source_doc_count=entity_info.get('doc_count', 0),
                    color=self.entity_colors.get(
                        entity_info.get('entity_type', 'UNKNOWN'), 
                        self.entity_colors['DEFAULT']
                    )
                )
            
            # Add edges for relationships
            for rel in relationships:
                source_id = str(rel.get('source_entity_id'))
                target_id = str(rel.get('target_entity_id'))
                
                if source_id in graph and target_id in graph:
                    graph.add_edge(
                        source_id,
                        target_id,
                        relationship_type=rel.get('relationship_type', 'UNKNOWN'),
                        confidence=rel.get('confidence_score', 0.0),
                        source_doc_id=rel.get('source_doc_id', ''),
                        weight=rel.get('confidence_score', 0.5)
                    )
            
            logger.info(f"Built graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
            return graph
            
        except Exception as e:
            raise GraphVisualizationException(f"Failed to build graph from query: {e}")

    def build_graph_from_traversal(self, seed_entities: List[Tuple[str, str, float]], 
                                  traversal_result: Set[str]) -> nx.Graph:
        """
        Build graph from traversal results with highlighted paths.
        
        Args:
            seed_entities: List of (entity_id, entity_name, confidence) tuples
            traversal_result: Set of all entity IDs found during traversal
            
        Returns:
            NetworkX Graph with traversal path highlighted
        """
        try:
            graph = nx.Graph()
            
            # Get all entity details
            all_entity_ids = list(traversal_result)
            entity_details = self._fetch_entity_details(all_entity_ids)
            
            # Add nodes
            for entity_id, entity_info in entity_details.items():
                is_seed = any(seed[0] == entity_id for seed in seed_entities)
                
                graph.add_node(
                    entity_id,
                    name=entity_info.get('entity_name', entity_id),
                    entity_type=entity_info.get('entity_type', 'UNKNOWN'),
                    is_seed=is_seed,
                    traversal_depth=0 if is_seed else 1,  # Simplified - could track actual depth
                    color=self.entity_colors.get(
                        entity_info.get('entity_type', 'UNKNOWN'),
                        self.entity_colors['DEFAULT']
                    )
                )
            
            # Get relationships between these entities
            relationships = self._fetch_relationships(all_entity_ids)
            
            # Add edges
            for rel in relationships:
                source_id = str(rel['source_entity_id'])
                target_id = str(rel['target_entity_id'])
                
                if source_id in graph and target_id in graph:
                    graph.add_edge(
                        source_id,
                        target_id,
                        relationship_type=rel.get('relationship_type', 'UNKNOWN'),
                        confidence=rel.get('confidence_score', 0.0),
                        weight=rel.get('confidence_score', 0.5)
                    )
            
            graph.graph['seed_entities'] = [seed[0] for seed in seed_entities]
            graph.graph['traversal_complete'] = True
            
            logger.info(f"Built traversal graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
            return graph
            
        except Exception as e:
            raise GraphVisualizationException(f"Failed to build traversal graph: {e}")

    def generate_plotly_visualization(self, graph: nx.Graph, 
                                     highlight_path: List[str] = None) -> str:
        """
        Generate interactive Plotly HTML visualization.
        
        Args:
            graph: NetworkX graph to visualize
            highlight_path: List of entity IDs to highlight as traversal path
            
        Returns:
            HTML string containing the Plotly visualization
        """
        try:
            # Import plotly here to avoid dependency issues if not installed
            import plotly.graph_objects as go
            import plotly.offline as pyo
            
            # Calculate layout
            if self.layout_algorithm == "spring":
                pos = nx.spring_layout(graph, k=3, iterations=50)
            elif self.layout_algorithm == "circular": 
                pos = nx.circular_layout(graph)
            elif self.layout_algorithm == "kamada_kawai":
                pos = nx.kamada_kawai_layout(graph)
            else:
                pos = nx.spring_layout(graph)
            
            # Extract node information
            node_x = []
            node_y = []
            node_info = []
            node_colors = []
            node_sizes = []
            
            for node in graph.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                node_data = graph.nodes[node]
                info_text = f"Entity: {node_data.get('name', node)}<br>"
                info_text += f"Type: {node_data.get('entity_type', 'Unknown')}<br>"
                info_text += f"Connections: {graph.degree(node)}"
                
                if node_data.get('is_seed', False):
                    info_text += "<br><b>SEED ENTITY</b>"
                
                node_info.append(info_text)
                
                # Highlight seed entities and path entities
                if highlight_path and node in highlight_path:
                    node_colors.append("#FF0000")  # Red for highlighted path
                    node_sizes.append(25)
                elif node_data.get('is_seed', False):
                    node_colors.append("#00FF00")  # Green for seed entities
                    node_sizes.append(30)
                else:
                    node_colors.append(node_data.get('color', self.entity_colors['DEFAULT']))
                    node_sizes.append(self.default_node_size)
            
            # Extract edge information
            edge_x = []
            edge_y = []
            edge_info = []
            
            for edge in graph.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                
                edge_data = graph.edges[edge]
                rel_type = edge_data.get('relationship_type', 'UNKNOWN')
                confidence = edge_data.get('confidence', 0.0)
                edge_info.append(f"{rel_type} (conf: {confidence:.2f})")
            
            # Create traces
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=self.default_edge_width, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=[graph.nodes[node].get('name', node)[:15] for node in graph.nodes()],
                textposition="middle center",
                hovertext=node_info,
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    line=dict(width=2, color='white')
                )
            )
            
            # Create figure
            fig = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=f'Knowledge Graph Visualization<br>Query: {graph.graph.get("query", "Unknown")}',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="Interactive Knowledge Graph - Hover for details",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor='left', yanchor='bottom',
                        font=dict(color="#888", size=12)
                    )],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=700
                )
            )
            
            # Generate HTML
            html_content = pyo.plot(fig, output_type='div', include_plotlyjs=True)
            
            logger.info("Generated Plotly visualization")
            return html_content
            
        except ImportError:
            raise GraphVisualizationException(
                "Plotly not installed. Install with: pip install plotly"
            )
        except Exception as e:
            raise GraphVisualizationException(f"Failed to generate Plotly visualization: {e}")

    def generate_d3_visualization(self, graph: nx.Graph) -> str:
        """
        Generate D3.js force-directed graph visualization HTML.
        
        Args:
            graph: NetworkX graph to visualize
            
        Returns:
            HTML string containing the D3.js visualization
        """
        return self._load_template('graph_d3_template.html', graph)

    def export_to_graphml(self, graph: nx.Graph, filepath: str):
        """
        Export graph to GraphML format for Gephi/Cytoscape.
        
        Args:
            graph: NetworkX graph to export
            filepath: Path where to save the GraphML file
        """
        try:
            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            # Export to GraphML
            nx.write_graphml(graph, filepath)
            
            logger.info(f"Graph exported to GraphML: {filepath}")
            
        except Exception as e:
            raise GraphVisualizationException(f"Failed to export graph to GraphML: {e}")

    def visualize_traversal_path(self, query_result: dict) -> str:
        """
        Visualize the traversal path from a GraphRAG query result.
        
        Args:
            query_result: Dictionary containing GraphRAG query results with metadata
            
        Returns:
            HTML string containing traversal path visualization
        """
        return self._load_template('traversal_path_template.html', query_result)

    def _load_template(self, template_name: str, data: Any) -> str:
        """Load and render HTML template with data."""
        try:
            template_path = Path(__file__).parent / 'templates' / template_name
            
            if not template_path.exists():
                raise GraphVisualizationException(f"Template not found: {template_name}")
            
            with open(template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
            
            # Simple template rendering - replace placeholders
            if template_name == 'traversal_path_template.html':
                return self._render_traversal_template(template_content, data)
            elif template_name == 'graph_d3_template.html':
                return self._render_d3_template(template_content, data)
            else:
                return template_content
                
        except Exception as e:
            logger.error(f"Failed to load template {template_name}: {e}")
            return f"<html><body><h1>Template Error</h1><p>{e}</p></body></html>"

    def _render_d3_template(self, template: str, graph: nx.Graph) -> str:
        """Render D3.js template with graph data."""
        # Convert NetworkX graph to D3.js format
        nodes = []
        links = []
        
        # Create nodes
        for node in graph.nodes():
            node_data = graph.nodes[node]
            nodes.append({
                "id": node,
                "name": node_data.get("name", node),
                "type": node_data.get("entity_type", "UNKNOWN"),
                "color": node_data.get("color", "#95A5A6"),
                "size": 30 if node_data.get("is_seed", False) else 20,
                "isSeed": node_data.get("is_seed", False)
            })
        
        # Create links
        for edge in graph.edges():
            edge_data = graph.edges[edge]
            links.append({
                "source": edge[0],
                "target": edge[1],
                "type": edge_data.get("relationship_type", "UNKNOWN"),
                "confidence": edge_data.get("confidence", 0.0),
                "weight": edge_data.get("weight", 1.0)
            })
        
        # Replace template placeholders
        template = template.replace('{{GRAPH_NODES}}', json.dumps(nodes))
        template = template.replace('{{GRAPH_LINKS}}', json.dumps(links))
        template = template.replace('{{QUERY_TEXT}}', graph.graph.get('query', 'Unknown'))
        
        return template

    def _render_traversal_template(self, template: str, query_result: dict) -> str:
        """Render traversal path template with query result data."""
        metadata = query_result.get('metadata', {})
        step_timings = metadata.get('step_timings_ms', {})
        
        # Replace template placeholders
        replacements = {
            '{{QUERY_TEXT}}': query_result.get('query', 'Unknown'),
            '{{QUERY_ENTITY_EXTRACTION_TIME}}': f"{step_timings.get('query_entity_extraction_ms', 0):.1f}",
            '{{FIND_SEED_ENTITIES_TIME}}': f"{step_timings.get('find_seed_entities_ms', 0):.1f}",
            '{{TRAVERSE_GRAPH_TIME}}': f"{step_timings.get('traverse_graph_ms', 0):.1f}",
            '{{GET_DOCUMENTS_TIME}}': f"{step_timings.get('get_documents_ms', 0):.1f}",
            '{{TOTAL_TIME}}': f"{metadata.get('processing_time_ms', 0):.1f}",
            '{{NUM_RETRIEVED}}': str(metadata.get('num_retrieved', 0)),
            '{{DB_EXEC_COUNT}}': str(metadata.get('db_exec_count', 0)),
            '{{RETRIEVAL_METHOD}}': metadata.get('retrieval_method', 'unknown').replace('_', ' ').title()
        }
        
        for placeholder, value in replacements.items():
            template = template.replace(placeholder, value)
        
        return template

    def _fetch_entity_details(self, entity_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch detailed entity information from database."""
        if not entity_ids:
            return {}
            
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()
        
        try:
            placeholders = ','.join(['?' for _ in entity_ids])
            query = f"""
                SELECT entity_id, entity_name, entity_type, COUNT(DISTINCT source_doc_id) as doc_count
                FROM RAG.Entities
                WHERE entity_id IN ({placeholders})
                GROUP BY entity_id, entity_name, entity_type
            """
            
            cursor.execute(query, entity_ids)
            results = cursor.fetchall()
            
            entity_details = {}
            for entity_id, entity_name, entity_type, doc_count in results:
                entity_details[str(entity_id)] = {
                    'entity_name': str(entity_name),
                    'entity_type': str(entity_type),
                    'doc_count': int(doc_count)
                }
            
            return entity_details
            
        except Exception as e:
            logger.error(f"Failed to fetch entity details: {e}")
            return {}
        finally:
            cursor.close()

    def _fetch_relationships(self, entity_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch relationships between given entities."""
        if not entity_ids:
            return []
            
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()
        
        try:
            placeholders = ','.join(['?' for _ in entity_ids])
            query = f"""
                SELECT source_entity_id, target_entity_id, relationship_type, 
                       confidence_score, source_doc_id
                FROM RAG.EntityRelationships
                WHERE source_entity_id IN ({placeholders}) 
                   OR target_entity_id IN ({placeholders})
            """
            
            cursor.execute(query, entity_ids + entity_ids)
            results = cursor.fetchall()
            
            relationships = []
            for source_id, target_id, rel_type, confidence, doc_id in results:
                relationships.append({
                    'source_entity_id': str(source_id),
                    'target_entity_id': str(target_id),
                    'relationship_type': str(rel_type),
                    'confidence_score': float(confidence) if confidence else 0.0,
                    'source_doc_id': str(doc_id)
                })
            
            return relationships
            
        except Exception as e:
            logger.error(f"Failed to fetch relationships: {e}")
            return []
        finally:
            cursor.close()