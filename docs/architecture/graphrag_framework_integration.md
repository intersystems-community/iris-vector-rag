# GraphRAG Framework Integration Architecture

## Overview

This document defines how GraphRAG integrates into the existing IRIS RAG framework, leveraging the comprehensive configuration, factory, and validation architecture already in place.

## Current Framework Architecture Analysis

### 1. Pipeline Factory System
- **`PipelineConfigService`**: Loads pipeline definitions from YAML files
- **`ModuleLoader`**: Dynamically loads pipeline classes  
- **`PipelineFactory`**: Creates pipeline instances with dependency injection
- **Framework Dependencies**: `connection_manager`, `config_manager`, `llm_func`, `vector_store`

### 2. Validation & Setup System
- **`ValidatedPipelineFactory`**: Validates requirements before pipeline creation
- **`PreConditionValidator`**: Checks table/embedding prerequisites
- **`SetupOrchestrator`**: Auto-setup missing requirements
- **`PipelineRequirements`**: Define what each pipeline needs

### 3. Configuration System
- **YAML Configuration**: Pipeline definitions in `config/pipelines.yaml`
- **ConfigurationManager**: Structured access to configuration
- **Pipeline-specific Parameters**: Via `params` section in pipeline definitions

### 4. Requirements Registry
- **`PIPELINE_REQUIREMENTS_REGISTRY`**: Maps pipeline types to requirement classes
- **Schema Integration**: Schema manager uses requirements to create tables

## GraphRAG Integration Issues

### Current Problems
1. **Missing from Factory**: GraphRAG not in `ValidatedPipelineFactory._create_pipeline_instance()`
2. **Missing from Registry**: No `GraphRAGRequirements` in `PIPELINE_REQUIREMENTS_REGISTRY`  
3. **No Configuration**: No GraphRAG pipeline definition in configuration system
4. **Missing Dependencies**: No entity extraction service in framework dependencies
5. **Schema Disconnect**: Schema manager unaware of GraphRAG table requirements

## Proposed Integration Architecture

### 1. Framework Dependencies Extension

```python
# Enhanced framework dependencies with entity extraction service
framework_dependencies = {
    "connection_manager": connection_manager,
    "config_manager": config_manager, 
    "llm_func": llm_func,
    "vector_store": vector_store,
    "entity_extraction_service": EntityExtractionService(config_manager)  # NEW
}
```

### 2. Entity Extraction Service Integration

```python
class EntityExtractionService:
    """Framework-level entity extraction service."""
    
    def __init__(self, config_manager: ConfigurationManager):
        """Initialize with configuration-driven extractors."""
        self.config_manager = config_manager
        self.extractors = self._load_extractors()
        self.default_strategy = config_manager.get('entity_extraction.default_strategy', 'nlp')
    
    def _load_extractors(self) -> Dict[str, IEntityExtractor]:
        """Load extractors based on configuration."""
        extractors = {}
        
        # Load enabled extractors from config
        extractor_configs = self.config_manager.get('entity_extraction.extractors', {})
        
        for extractor_type, config in extractor_configs.items():
            if config.get('enabled', False):
                extractors[extractor_type] = self._create_extractor(extractor_type, config)
        
        return extractors
    
    def extract_entities(self, document: Document, strategy: Optional[str] = None) -> List[Entity]:
        """Extract entities using specified or default strategy."""
        strategy = strategy or self.default_strategy
        extractor = self.extractors.get(strategy)
        
        if not extractor:
            raise ValueError(f"Extractor strategy '{strategy}' not available")
        
        return extractor.extract_entities(document)
```

### 3. GraphRAG Requirements Definition

```python
class GraphRAGRequirements(PipelineRequirements):
    """Requirements for GraphRAG pipeline."""
    
    @property
    def pipeline_name(self) -> str:
        return "graphrag"
    
    @property
    def required_tables(self) -> List[TableRequirement]:
        return [
            TableRequirement(
                name="SourceDocuments", 
                schema="RAG", 
                description="Document storage with embeddings",
                min_rows=1,
                supports_vector_search=True
            ),
            TableRequirement(
                name="Entities", 
                schema="RAG", 
                description="Extracted entities with embeddings",
                min_rows=5,  # Require minimum entities for graph queries
                supports_vector_search=True
            ),
            TableRequirement(
                name="EntityRelationships", 
                schema="RAG", 
                description="Entity relationships for graph traversal",
                min_rows=2,  # Require minimum relationships for connectivity
                supports_vector_search=False
            )
        ]
    
    @property
    def required_embeddings(self) -> List[EmbeddingRequirement]:
        return [
            EmbeddingRequirement(
                name="document_embeddings",
                table="RAG.SourceDocuments", 
                column="embedding",
                description="Document-level embeddings for fallback search"
            ),
            EmbeddingRequirement(
                name="entity_embeddings",
                table="RAG.Entities",
                column="embedding", 
                description="Entity embeddings for similarity-based linking"
            )
        ]

# Update registry
PIPELINE_REQUIREMENTS_REGISTRY = {
    "basic": BasicRAGRequirements,
    "basic_rerank": BasicRAGRerankingRequirements,
    "crag": CRAGRequirements,
    "graphrag": GraphRAGRequirements  # NEW
}
```

### 4. ValidatedPipelineFactory Extension

```python
class ValidatedPipelineFactory:
    """Enhanced factory with GraphRAG support."""
    
    def __init__(self, connection_manager: ConnectionManager, config_manager: ConfigurationManager):
        self.connection_manager = connection_manager
        self.config_manager = config_manager
        self.embedding_manager = EmbeddingManager(config_manager)
        self.entity_extraction_service = EntityExtractionService(config_manager)  # NEW
        self.validator = PreConditionValidator(connection_manager)
        self.orchestrator = SetupOrchestrator(connection_manager, config_manager)
    
    def _create_pipeline_instance(
        self, pipeline_type: str, llm_func: Optional[Callable[[str], str]], **kwargs
    ) -> RAGPipeline:
        """Create pipeline instance with GraphRAG support."""
        
        # Common dependencies for all pipelines
        common_args = {
            "connection_manager": self.connection_manager,
            "config_manager": self.config_manager,
            "llm_func": llm_func
        }
        
        if pipeline_type == "basic":
            return BasicRAGPipeline(**common_args)
        elif pipeline_type == "crag":
            return CRAGPipeline(**common_args)
        elif pipeline_type == "basic_rerank":
            return BasicRAGRerankingPipeline(**common_args)
        elif pipeline_type == "graphrag":
            # GraphRAG gets entity extraction service as additional dependency
            return GraphRAGPipeline(
                **common_args,
                entity_extraction_service=self.entity_extraction_service  # NEW
            )
        else:
            available_types = ["basic", "basic_rerank", "crag", "graphrag"]
            raise ValueError(f"Unknown pipeline type: {pipeline_type}. Available: {available_types}")
```

### 5. Enhanced GraphRAG Pipeline

```python
class GraphRAGPipeline(RAGPipeline):
    """GraphRAG pipeline with proper framework integration."""
    
    def __init__(
        self,
        connection_manager: Optional[ConnectionManager] = None,
        config_manager: Optional[ConfigurationManager] = None,
        llm_func: Optional[Callable[[str], str]] = None,
        vector_store=None,
        entity_extraction_service: Optional[EntityExtractionService] = None
    ):
        super().__init__(connection_manager, config_manager, vector_store)
        self.llm_func = llm_func
        
        # Get entity extraction service from framework or create default
        if entity_extraction_service:
            self.entity_extraction_service = entity_extraction_service
        else:
            self.entity_extraction_service = EntityExtractionService(config_manager)
        
        # Load GraphRAG-specific configuration
        self.pipeline_config = self.config_manager.get("pipelines:graphrag", {})
        self.traversal_config = self.pipeline_config.get("traversal", {})
        self.extraction_config = self.pipeline_config.get("entity_extraction", {})
```

### 6. Configuration Schema

```yaml
# config/default_config.yaml - GraphRAG Extensions
pipelines:
  basic:
    chunk_size: 1000
    chunk_overlap: 200
    default_top_k: 5
    embedding_batch_size: 32
  
  # NEW: GraphRAG Configuration
  graphrag:
    default_top_k: 10
    traversal:
      max_depth: 2
      max_entities: 50
      min_confidence: 0.7
      fallback_to_vector: true
    entity_extraction:
      default_strategy: "hybrid"
      batch_size: 5
      confidence_threshold: 0.7
    relationship_extraction:
      max_distance: 3
      confidence_threshold: 0.6

# NEW: Entity Extraction Configuration
entity_extraction:
  default_strategy: "nlp"  # nlp, llm, pattern, hybrid
  
  extractors:
    nlp:
      enabled: true
      model: "en_core_web_sm"
      confidence_threshold: 0.7
      custom_patterns:
        DRUG: ["aspirin", "ibuprofen", "acetaminophen"]
        DISEASE: ["diabetes", "hypertension", "covid-19"]
    
    llm:
      enabled: false  # Enable if LLM is available
      model: "gpt-4"
      max_retries: 3
      rate_limit_delay: 1.0
      batch_size: 3
    
    pattern:
      enabled: true
      regex_patterns:
        GENE: "\\b[A-Z]{2,}[0-9]*\\b"
        PROTEIN: "\\bp[0-9]+\\b"
  
  domain:
    name: "biomedical"
    entity_types: ["PERSON", "ORG", "DISEASE", "DRUG", "TREATMENT", "SYMPTOM"]
    confidence_thresholds:
      DISEASE: 0.8
      DRUG: 0.9
      default: 0.7
```

### 7. Pipeline Definition Configuration

```yaml
# config/pipelines.yaml - Pipeline Definitions
pipelines:
  - name: "basic"
    module: "iris_rag.pipelines.basic"
    class: "BasicRAGPipeline"
    enabled: true
    params:
      description: "Basic RAG with vector search"
  
  - name: "crag"
    module: "iris_rag.pipelines.crag"
    class: "CRAGPipeline"
    enabled: true
    params:
      description: "Corrective RAG with retrieval evaluation"
  
  # NEW: GraphRAG Pipeline Definition
  - name: "graphrag"
    module: "iris_rag.pipelines.graphrag"
    class: "GraphRAGPipeline"
    enabled: true
    params:
      description: "Knowledge Graph RAG with entity-based retrieval"
      requires_entity_extraction: true
```

### 8. Schema Manager Integration

```python
class SchemaManager:
    """Extended with GraphRAG table support."""
    
    def _build_table_configurations(self):
        """Build table configurations including GraphRAG tables."""
        self._table_configs = {
            # Existing configurations
            "SourceDocuments": {...},
            "DocumentChunks": {...},
            
            # NEW: GraphRAG table configurations
            "Entities": {
                "embedding_column": "embedding",
                "uses_entity_embeddings": True,
                "default_model": self.base_embedding_model,
                "dimension": self.base_embedding_dimension,
                "table_type": "knowledge_graph",
                "supports_graph_traversal": True,
                "pipeline_type": "graphrag"
            },
            "EntityRelationships": {
                "embedding_column": None,
                "table_type": "knowledge_graph", 
                "supports_graph_traversal": True,
                "pipeline_type": "graphrag"
            }
        }
    
    def _get_expected_schema_config(self, table_name: str, pipeline_type: str = None) -> Dict[str, Any]:
        """Enhanced to handle GraphRAG table requirements."""
        config = super()._get_expected_schema_config(table_name, pipeline_type)
        
        # Get requirements from pipeline type if specified
        if pipeline_type:
            try:
                requirements = get_pipeline_requirements(pipeline_type)
                # Find table requirement for this table
                for table_req in requirements.required_tables + requirements.optional_tables:
                    if table_req.name == table_name:
                        config.update({
                            "supports_vector_search": table_req.supports_vector_search,
                            "text_content_type": getattr(table_req, 'text_content_type', 'LONGVARCHAR'),
                            "min_rows": table_req.min_rows
                        })
                        break
            except Exception as e:
                logger.warning(f"Could not get requirements for {pipeline_type}: {e}")
        
        return config
```

### 9. Enhanced Document Loading

```python
class GraphRAGPipeline(RAGPipeline):
    """Enhanced with framework-integrated entity extraction."""
    
    def load_documents(self, documents_path: str, **kwargs) -> None:
        """Load documents with entity extraction."""
        start_time = time.time()
        
        # Step 1: Load and store documents (existing)
        if "documents" in kwargs:
            documents = kwargs["documents"]
        else:
            documents = self._load_documents_from_path(documents_path)
        
        # Store documents with embeddings
        generate_embeddings = kwargs.get("generate_embeddings", True)
        if generate_embeddings:
            self.vector_store.add_documents(documents, auto_chunk=True)
        else:
            self._store_documents(documents)
        
        # Step 2: Extract entities using framework service (NEW)
        extract_entities = kwargs.get("extract_entities", True)
        if extract_entities and self.entity_extraction_service:
            self._extract_and_store_entities(documents)
        
        processing_time = time.time() - start_time
        logger.info(f"GraphRAG: Loaded {len(documents)} documents with entities in {processing_time:.2f}s")
    
    def _extract_and_store_entities(self, documents: List[Document]) -> None:
        """Extract and store entities using framework service."""
        batch_size = self.extraction_config.get("batch_size", 5)
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            for document in batch:
                try:
                    # Extract entities
                    entities = self.entity_extraction_service.extract_entities(document)
                    
                    # Extract relationships
                    relationships = self.entity_extraction_service.extract_relationships(
                        document, entities
                    )
                    
                    # Store in knowledge graph
                    self._store_entities(entities, document.id)
                    self._store_relationships(relationships, document.id)
                    
                except Exception as e:
                    logger.warning(f"Entity extraction failed for document {document.id}: {e}")
```

## Framework Integration Benefits

### 1. Configuration-Driven
- **Entity Extraction Strategy**: Configurable via YAML
- **Pipeline Parameters**: Standard configuration system
- **Domain-Specific Settings**: Centralized in configuration

### 2. Validation & Setup
- **Prerequisites Checked**: Before pipeline creation
- **Auto-Setup**: Missing tables/embeddings created automatically
- **Error Reporting**: Clear validation messages with setup suggestions

### 3. Dependency Injection
- **Entity Extraction Service**: Injected as framework dependency
- **Configuration Access**: Via standard ConfigurationManager
- **Database Connection**: Via ConnectionManager

### 4. Modular Architecture
- **Pluggable Extractors**: Add new extraction strategies
- **Framework Consistency**: Same patterns as other pipelines
- **Service Boundaries**: Clean separation of concerns

### 5. Production Ready
- **Error Handling**: Comprehensive exception handling
- **Performance Monitoring**: Built-in metrics and logging
- **Scalability**: Batch processing and resource management
- **Maintainability**: Standard framework patterns

This integration approach ensures GraphRAG becomes a first-class citizen in the IRIS RAG framework while maintaining backward compatibility and leveraging all existing framework capabilities.