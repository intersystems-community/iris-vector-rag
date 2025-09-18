"""
Extended GraphVisualizer methods for D3.js, export, and dashboard functionality.
"""

import logging
import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

import networkx as nx

logger = logging.getLogger(__name__)


class GraphVisualizerExtended:
    """Extended visualization methods for GraphVisualizer."""
    
    @staticmethod
    def generate_d3_visualization(graph: nx.Graph) -> str:
        """
        Generate D3.js force-directed graph visualization HTML.
        
        Args:
            graph: NetworkX graph to visualize
            
        Returns:
            HTML string containing the D3.js visualization
        """
        try:
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
            
            # Create the D3.js HTML template
            d3_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Knowledge Graph - D3.js Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .graph-container {{
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-top: 20px;
        }}
        
        .node {{
            cursor: pointer;
            stroke: #fff;
            stroke-width: 2px;
        }}
        
        .node.seed {{
            stroke: #00ff00;
            stroke-width: 4px;
        }}
        
        .link {{
            stroke: #999;
            stroke-opacity: 0.6;
        }}
        
        .node-label {{
            font-size: 12px;
            text-anchor: middle;
            pointer-events: none;
            fill: #333;
        }}
        
        .tooltip {{
            position: absolute;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 8px;
            border-radius: 4px;
            font-size: 12px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.3s;
        }}
        
        .controls {{
            margin-bottom: 20px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 4px;
        }}
        
        .control-btn {{
            margin: 5px;
            padding: 8px 16px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }}
        
        .control-btn:hover {{
            background: #0056b3;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Knowledge Graph Visualization</h1>
        <p>Query: <strong>{graph.graph.get('query', 'Unknown')}</strong></p>
        
        <div class="controls">
            <button class="control-btn" onclick="restartSimulation()">Restart Layout</button>
            <button class="control-btn" onclick="centerGraph()">Center Graph</button>
            <button class="control-btn" onclick="toggleLabels()">Toggle Labels</button>
            <button class="control-btn" onclick="exportPNG()">Export PNG</button>
        </div>
        
        <div class="graph-container">
            <svg id="graph"></svg>
        </div>
    </div>
    
    <div class="tooltip" id="tooltip"></div>
    
    <script>
        // Graph data
        const graphData = {{
            nodes: {json.dumps(nodes)},
            links: {json.dumps(links)}
        }};
        
        // Set up SVG
        const width = 1160;
        const height = 600;
        
        const svg = d3.select("#graph")
            .attr("width", width)
            .attr("height", height);
        
        // Create groups for links and nodes
        const linkGroup = svg.append("g").attr("class", "links");
        const nodeGroup = svg.append("g").attr("class", "nodes");
        const labelGroup = svg.append("g").attr("class", "labels");
        
        // Set up simulation
        const simulation = d3.forceSimulation(graphData.nodes)
            .force("link", d3.forceLink(graphData.links).id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(25));
        
        // Create links
        const link = linkGroup.selectAll("line")
            .data(graphData.links)
            .enter().append("line")
            .attr("class", "link")
            .attr("stroke-width", d => Math.sqrt(d.weight * 3));
        
        // Create nodes
        const node = nodeGroup.selectAll("circle")
            .data(graphData.nodes)
            .enter().append("circle")
            .attr("class", d => d.isSeed ? "node seed" : "node")
            .attr("r", d => d.size)
            .attr("fill", d => d.color)
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));
        
        // Create labels
        let labelsVisible = true;
        const label = labelGroup.selectAll("text")
            .data(graphData.nodes)
            .enter().append("text")
            .attr("class", "node-label")
            .text(d => d.name.length > 15 ? d.name.substring(0, 15) + "..." : d.name);
        
        // Tooltip
        const tooltip = d3.select("#tooltip");
        
        // Add hover events
        node.on("mouseover", function(event, d) {{
            tooltip.style("opacity", 1)
                .html(`
                    <strong>${{d.name}}</strong><br/>
                    Type: ${{d.type}}<br/>
                    ID: ${{d.id}}<br/>
                    ${{d.isSeed ? '<strong>SEED ENTITY</strong>' : ''}}
                `)
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 10) + "px");
        }})
        .on("mouseout", function() {{
            tooltip.style("opacity", 0);
        }});
        
        // Update positions on simulation tick
        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            
            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);
            
            label
                .attr("x", d => d.x)
                .attr("y", d => d.y + 4);
        }});
        
        // Drag functions
        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}
        
        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}
        
        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}
        
        // Control functions
        function restartSimulation() {{
            simulation.alpha(1).restart();
        }}
        
        function centerGraph() {{
            simulation.force("center", d3.forceCenter(width / 2, height / 2));
            simulation.alpha(0.3).restart();
        }}
        
        function toggleLabels() {{
            labelsVisible = !labelsVisible;
            label.style("display", labelsVisible ? "block" : "none");
        }}
        
        function exportPNG() {{
            // Create a canvas element
            const canvas = document.createElement('canvas');
            canvas.width = width;
            canvas.height = height;
            const context = canvas.getContext('2d');
            
            // Convert SVG to data URL and draw on canvas
            const svgData = new XMLSerializer().serializeToString(svg.node());
            const svgBlob = new Blob([svgData], {{type: 'image/svg+xml;charset=utf-8'}});
            const url = URL.createObjectURL(svgBlob);
            
            const img = new Image();
            img.onload = function() {{
                context.drawImage(img, 0, 0);
                URL.revokeObjectURL(url);
                
                // Download the image
                const link = document.createElement('a');
                link.download = 'knowledge-graph.png';
                link.href = canvas.toDataURL();
                link.click();
            }};
            img.src = url;
        }}
    </script>
</body>
</html>"""
            
            logger.info("Generated D3.js visualization")
            return d3_html
            
        except Exception as e:
            raise Exception(f"Failed to generate D3.js visualization: {e}")

    @staticmethod
    def export_to_graphml(graph: nx.Graph, filepath: str):
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
            raise Exception(f"Failed to export graph to GraphML: {e}")

    @staticmethod
    def visualize_traversal_path(query_result: dict) -> str:
        """
        Visualize the traversal path from a GraphRAG query result.
        
        Args:
            query_result: Dictionary containing GraphRAG query results with metadata
            
        Returns:
            HTML string containing traversal path visualization
        """
        try:
            # Extract traversal information from metadata
            metadata = query_result.get('metadata', {})
            step_timings = metadata.get('step_timings_ms', {})
            retrieval_method = metadata.get('retrieval_method', 'unknown')
            
            # Create traversal path visualization
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>GraphRAG Traversal Path</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #f0f0f0;
        }}
        
        .header h1 {{
            color: #333;
            margin: 0 0 10px 0;
        }}
        
        .query-text {{
            background: #f8f9fa;
            padding: 15px;
            border-left: 4px solid #007bff;
            border-radius: 4px;
            font-style: italic;
        }}
        
        .traversal-steps {{
            margin: 30px 0;
        }}
        
        .step {{
            display: flex;
            align-items: center;
            margin: 20px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #28a745;
        }}
        
        .step-number {{
            background: #007bff;
            color: white;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 15px;
            font-weight: bold;
        }}
        
        .step-content {{
            flex: 1;
        }}
        
        .step-title {{
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
        }}
        
        .step-timing {{
            color: #666;
            font-size: 0.9em;
        }}
        
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        
        .metric-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        
        .documents-section {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 2px solid #f0f0f0;
        }}
        
        .document {{
            margin: 15px 0;
            padding: 15px;
            background: #fff;
            border: 1px solid #ddd;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .document-title {{
            font-weight: bold;
            color: #007bff;
            margin-bottom: 8px;
        }}
        
        .document-preview {{
            color: #666;
            font-size: 0.9em;
            line-height: 1.4;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>GraphRAG Traversal Analysis</h1>
            <div class="query-text">
                <strong>Query:</strong> {query_result.get('query', 'Unknown')}
            </div>
        </div>
        
        <div class="traversal-steps">
            <h2>Traversal Steps</h2>
            
            <div class="step">
                <div class="step-number">1</div>
                <div class="step-content">
                    <div class="step-title">Query Entity Extraction</div>
                    <div class="step-timing">
                        Time: {step_timings.get('query_entity_extraction_ms', 0):.1f}ms
                    </div>
                </div>
            </div>
            
            <div class="step">
                <div class="step-number">2</div>
                <div class="step-content">
                    <div class="step-title">Find Seed Entities</div>
                    <div class="step-timing">
                        Time: {step_timings.get('find_seed_entities_ms', 0):.1f}ms
                    </div>
                </div>
            </div>
            
            <div class="step">
                <div class="step-number">3</div>
                <div class="step-content">
                    <div class="step-title">Graph Traversal</div>
                    <div class="step-timing">
                        Time: {step_timings.get('traverse_graph_ms', 0):.1f}ms
                    </div>
                </div>
            </div>
            
            <div class="step">
                <div class="step-number">4</div>
                <div class="step-content">
                    <div class="step-title">Document Retrieval</div>
                    <div class="step-timing">
                        Time: {step_timings.get('get_documents_ms', 0):.1f}ms
                    </div>
                </div>
            </div>
        </div>
        
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value">{metadata.get('processing_time_ms', 0):.1f}</div>
                <div class="metric-label">Total Time (ms)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metadata.get('num_retrieved', 0)}</div>
                <div class="metric-label">Documents Retrieved</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metadata.get('db_exec_count', 0)}</div>
                <div class="metric-label">Database Queries</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{retrieval_method.replace('_', ' ').title()}</div>
                <div class="metric-label">Retrieval Method</div>
            </div>
        </div>
        
        <div class="documents-section">
            <h2>Retrieved Documents</h2>
"""
            
            # Add retrieved documents
            documents = query_result.get('retrieved_documents', [])
            for i, doc in enumerate(documents[:5]):  # Show first 5 documents
                title = doc.metadata.get('title', 'Untitled') if hasattr(doc, 'metadata') and doc.metadata else 'Untitled'
                content_preview = str(doc.page_content)[:200] + "..." if hasattr(doc, 'page_content') else "No content"
                
                html_content += f"""
            <div class="document">
                <div class="document-title">Document {i+1}: {title}</div>
                <div class="document-preview">{content_preview}</div>
            </div>
"""
            
            html_content += """
        </div>
    </div>
</body>
</html>"""
            
            logger.info("Generated traversal path visualization")
            return html_content
            
        except Exception as e:
            raise Exception(f"Failed to generate traversal visualization: {e}")