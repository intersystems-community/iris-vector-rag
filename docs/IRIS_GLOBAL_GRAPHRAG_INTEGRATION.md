# IRIS Global GraphRAG Integration

This document describes the integration of the IRIS-Global-GraphRAG project into our RAG framework, providing academic paper retrieval with interactive graph visualization.

## Overview

The IRIS Global GraphRAG pipeline integrates the core functionality from the external IRIS-Global-GraphRAG project while leveraging our enterprise RAG framework features. This creates a powerful combination of:

- **IRIS Globals-based graph storage** from the original project
- **Interactive 3D graph visualization** with academic paper networks
- **Side-by-side comparison** of LLM vs RAG vs GraphRAG approaches
- **Enterprise pipeline management** from our framework

## Architecture

### Integration Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                RAG-Templates Framework                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │         IRISGlobalGraphRAGPipeline                  │    │
│  │                                                     │    │
│  │  ┌──────────────────────────────────────────────┐   │    │
│  │  │     IRIS-Global-GraphRAG Core Functions     │   │    │
│  │  │                                              │   │    │
│  │  │  • iris_db.ask_question_rag()               │   │    │
│  │  │  • iris_db.ask_question_graphrag()          │   │    │
│  │  │  • iris_db.prepare_combined_results()       │   │    │
│  │  │  • iris_db.combine_graphs()                 │   │    │
│  │  │  • IRIS Globals storage functions           │   │    │
│  │  └──────────────────────────────────────────────┘   │    │
│  │                                                     │    │
│  │  Framework Integration:                             │    │
│  │  • ConfigurationManager                            │    │
│  │  • ConnectionManager                               │    │
│  │  • EmbeddingManager                                │    │
│  │  • Pipeline Factory                                │    │
│  │  • Validation & Monitoring                         │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Document Ingestion**: Academic papers → SQL tables + IRIS Globals
2. **Vector Search**: HNSW index on paper embeddings (384-dimensional)
3. **Graph Storage**: Entities/relationships stored in `^GraphContent` and `^GraphRelations`
4. **Query Processing**: Multi-modal retrieval combining vector + graph
5. **Visualization**: Interactive 3D graph network with D3.js Force Graph

## Setup and Configuration

### Prerequisites

1. **IRIS-Global-GraphRAG Project**: Must be available as sibling directory or configured path
2. **Dependencies**: `sentence-transformers`, `flask`, `iris-python`
3. **IRIS Database**: With globals and vector support

### Installation

1. **Clone IRIS-Global-GraphRAG** (if not already available):
   ```bash
   cd /Users/tdyar/ws/
   git clone https://github.com/your-colleague/IRIS-Global-GraphRAG.git
   ```

2. **Configure Pipeline** in `config/pipelines.yaml`:
   ```yaml
   - name: "IRISGlobalGraphRAG"
     module: "iris_rag.pipelines.iris_global_graphrag"
     class: "IRISGlobalGraphRAGPipeline"
     enabled: true
     params:
       project_path: "/Users/tdyar/ws/IRIS-Global-GraphRAG"  # Optional: auto-discovery
       embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
       enable_visualization: true
       enable_comparison_ui: true
   ```

3. **Environment Variables** (optional):
   ```bash
   export IRIS_GLOBAL_GRAPHRAG_PATH="/path/to/IRIS-Global-GraphRAG"
   ```

## Usage

### Basic Pipeline Usage

```python
from iris_rag.pipelines.factory import create_pipeline

# Create pipeline
pipeline = create_pipeline("IRISGlobalGraphRAG")

# Load academic papers
documents = [
    Document(
        id="paper_1",
        page_content="GraphRAG combines vector similarity with knowledge graph traversal...",
        metadata={
            "title": "Enhanced Retrieval with GraphRAG",
            "authors": "John Doe, Jane Smith",
            "entities": [...],
            "relationships": [...]
        }
    )
]
pipeline.load_documents(documents)

# Query with different modes
rag_result = pipeline.query("What is GraphRAG?", mode="rag")
graphrag_result = pipeline.query("What is GraphRAG?", mode="graphrag", enable_visualization=True)

# Compare all approaches
comparison = pipeline.compare_modes("What is GraphRAG?")
```

### Web Interface

```python
from iris_rag.visualization.iris_global_graphrag_interface import IRISGlobalGraphRAGInterface

# Create interface
interface = IRISGlobalGraphRAGInterface(pipeline)

# Run standalone web server
interface.run_standalone(host="0.0.0.0", port=8000)
```

**Available Endpoints**:
- `/` or `/graphrag` - Main GraphRAG interface with 3D visualization
- `/rag` - RAG-only interface
- `/llm` - LLM-only interface
- `/llm-vs-rag` - Side-by-side LLM vs RAG comparison
- `/rag-vs-graphrag` - Side-by-side RAG vs GraphRAG comparison

**API Endpoints**:
- `POST /api/ask` - RAG queries
- `POST /api/graphrag` - GraphRAG queries with visualization data
- `POST /api/compare` - Multi-mode comparison
- `POST /api/llm` - LLM-only queries
- `GET /api/pipeline/info` - Pipeline status and configuration

### Command Line Interface

```bash
# Run tests
python scripts/test_iris_global_graphrag.py

# Run web interface
python scripts/test_iris_global_graphrag.py --web

# Direct interface
python -m iris_rag.visualization.iris_global_graphrag_interface --host 0.0.0.0 --port 8000
```

## Features

### 1. Multi-Modal Retrieval

- **RAG Mode**: Vector similarity search only
- **GraphRAG Mode**: Vector + graph traversal with entity relationships
- **LLM Mode**: Direct LLM without retrieval

### 2. Interactive Visualization

- **3D Force Graph**: Real-time network visualization
- **Entity-Relationship Display**: Academic paper connections
- **Interactive Navigation**: Zoom, rotate, node selection
- **Graph Modal**: Popup overlay with full graph view

### 3. Side-by-Side Comparison

- **LLM vs RAG**: Compare retrieval vs direct LLM responses
- **RAG vs GraphRAG**: Compare vector-only vs hybrid approaches
- **Performance Metrics**: Response times and processing details

### 4. Academic Paper Support

- **Entity Extraction**: Authors, concepts, methods
- **Relationship Mapping**: Paper citations, concept relationships
- **Metadata Handling**: Publication dates, URLs, abstracts
- **IRIS Globals Storage**: Efficient graph data persistence

## Schema and Data Model

### SQL Schema

```sql
CREATE TABLE paper_content (
    docid VARCHAR(255),
    title VARCHAR(255),
    abstract VARCHAR(2000),
    url VARCHAR(255),
    published VARCHAR(255),
    authors VARCHAR(255),
    combined VARCHAR(10000),
    paper_vector VECTOR(FLOAT, 384)
);

CREATE INDEX HNSWIndex ON TABLE paper_content (paper_vector)
AS HNSW(Distance='DotProduct');
```

### IRIS Globals Schema

```
^GraphContent(docid, "title") = "Paper Title"
^GraphContent(docid, "abstract") = "Paper Abstract"
^GraphContent(docid, "authors") = "Author Names"
^GraphContent(docid, "url") = "Paper URL"
^GraphContent(docid, "published") = "Publication Date"

^GraphRelations(docid, "Node", entity_name) = entity_type
^GraphRelations(docid, "Edge", source_entity, target_entity) = relationship_type
```

## Configuration Options

### Pipeline Configuration

```yaml
iris_global_graphrag:
  enabled: true
  project_path: null  # Auto-discovery or explicit path
  graph_content_global: "GraphContent"
  graph_relations_global: "GraphRelations"
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  vector_dimension: 384
  top_k: 5
  enable_visualization: true
  enable_comparison_ui: true
```

### Discovery Priority

1. **Configuration setting**: `project_path` in config
2. **Environment variable**: `IRIS_GLOBAL_GRAPHRAG_PATH`
3. **Sibling directory search**: Auto-discovery of adjacent project

## Performance and Scaling

### Optimizations

- **HNSW Vector Index**: Fast approximate nearest neighbor search
- **IRIS Globals**: Efficient graph storage and traversal
- **Lazy Loading**: Components loaded on-demand
- **Connection Reuse**: Shared IRIS connections
- **Template Caching**: Static asset optimization

### Limitations

- **Academic Paper Focus**: Optimized for research paper workflows
- **Vector Dimension**: Fixed to 384-dimensional embeddings
- **Memory Usage**: 3D visualization requires client-side resources
- **Concurrent Users**: Single-user demo interface (can be extended)

## Comparison with Other Pipelines

| Feature | IRIS Global GraphRAG | HybridGraphRAG | Standard GraphRAG |
|---------|---------------------|----------------|-------------------|
| **Graph Storage** | IRIS Globals | SQL Tables + NodePK | SQL Tables |
| **Visualization** | 3D Interactive | Performance Dashboard | Basic |
| **Use Case** | Academic Papers | Enterprise Scale | General Purpose |
| **Performance** | Demo-optimized | Production-optimized | Balanced |
| **Comparison UI** | Side-by-side | Single pipeline | Single pipeline |
| **Entity Extraction** | Academic entities | General entities | General entities |

## Troubleshooting

### Common Issues

1. **Project Not Found**:
   ```
   IRISGlobalGraphRAGException: IRIS-Global-GraphRAG project not found
   ```
   **Solution**: Set `project_path` in config or `IRIS_GLOBAL_GRAPHRAG_PATH` environment variable

2. **Import Errors**:
   ```
   ImportError: Failed to import IRIS-Global-GraphRAG modules
   ```
   **Solution**: Ensure the project has `app/iris_db.py` and dependencies are installed

3. **Template Not Found**:
   ```
   Template directory not found
   ```
   **Solution**: Run setup script to copy visualization assets

4. **IRIS Connection Failed**:
   ```
   Failed to establish IRIS connections
   ```
   **Solution**: Check IRIS database configuration and credentials

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test pipeline components
pipeline = create_pipeline("IRISGlobalGraphRAG")
info = pipeline.get_pipeline_info()
print(f"Pipeline status: {info['status']}")
```

## Contributing

### Adding Features

1. **Extend Pipeline**: Add methods to `IRISGlobalGraphRAGPipeline`
2. **Enhance Interface**: Modify `IRISGlobalGraphRAGInterface`
3. **Update Templates**: Edit HTML/CSS in `visualization/iris_global_graphrag/`
4. **Add Tests**: Extend `scripts/test_iris_global_graphrag.py`

### Best Practices

- **Preserve Original Functions**: Don't modify IRIS-Global-GraphRAG code
- **Framework Integration**: Use our config/connection managers
- **Error Handling**: Graceful fallbacks for missing components
- **Documentation**: Update this guide for new features

## Future Enhancements

### Planned Features

1. **NodePK Integration**: Combine with explicit node identity
2. **Multi-User Support**: Session management and user isolation
3. **Advanced Visualizations**: Additional graph layout algorithms
4. **Export Capabilities**: Graph data export to various formats
5. **Real-time Updates**: Live graph updates during queries
6. **Mobile Interface**: Responsive design for mobile devices

### Integration Opportunities

- **Hybrid Performance**: Combine with HybridGraphRAG optimizations
- **Enterprise Features**: Add authentication, audit logging
- **Batch Processing**: Bulk paper ingestion workflows
- **API Integration**: Connect with academic databases (arXiv, PubMed)

---

*This integration demonstrates the power of combining specialized projects with enterprise frameworks, creating solutions that leverage the best of both approaches.*