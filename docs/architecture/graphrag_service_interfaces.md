# GraphRAG Service Interfaces

## Overview

This document defines the service interfaces and contracts for the GraphRAG Entity Extraction System. These interfaces ensure clean separation of concerns and extensibility.

## Core Interfaces

### 1. Entity Extraction Interface

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class EntityType(Enum):
    """Supported entity types for biomedical domain."""
    PERSON = "PERSON"
    ORGANIZATION = "ORG"
    DISEASE = "DISEASE"
    DRUG = "DRUG"
    TREATMENT = "TREATMENT"
    SYMPTOM = "SYMPTOM"
    GENE = "GENE"
    PROTEIN = "PROTEIN"
    ANATOMICAL_STRUCTURE = "ANATOMICAL_STRUCTURE"
    MEDICAL_DEVICE = "MEDICAL_DEVICE"

@dataclass
class Entity:
    """Represents an extracted entity."""
    entity_id: str
    entity_name: str
    entity_type: EntityType
    confidence: float
    source_doc_id: str
    char_start: int
    char_end: int
    canonical_name: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class Relationship:
    """Represents a relationship between entities."""
    relationship_id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    confidence: float
    source_doc_id: str
    metadata: Dict[str, Any] = None

@dataclass
class DomainConfig:
    """Configuration for domain-specific extraction."""
    domain_name: str
    entity_types: List[EntityType]
    relationship_types: List[str]
    extraction_patterns: Dict[str, Any]
    confidence_thresholds: Dict[str, float]

class IEntityExtractor(ABC):
    """Interface for entity extraction services."""
    
    @abstractmethod
    def extract_entities(self, document: Document) -> List[Entity]:
        """
        Extract entities from a document.
        
        Args:
            document: The document to extract entities from
            
        Returns:
            List of extracted entities with confidence scores
            
        Raises:
            ExtractionError: If extraction fails
        """
        pass
    
    @abstractmethod
    def extract_relationships(self, document: Document, 
                            entities: List[Entity]) -> List[Relationship]:
        """
        Extract relationships between entities in a document.
        
        Args:
            document: The document containing the entities
            entities: List of entities found in the document
            
        Returns:
            List of extracted relationships
            
        Raises:
            ExtractionError: If relationship extraction fails
        """
        pass
    
    @abstractmethod
    def set_domain_config(self, domain_config: DomainConfig) -> None:
        """
        Configure the extractor for a specific domain.
        
        Args:
            domain_config: Domain-specific configuration
        """
        pass
    
    @abstractmethod
    def get_supported_entity_types(self) -> List[EntityType]:
        """
        Get the entity types this extractor can identify.
        
        Returns:
            List of supported entity types
        """
        pass
    
    @abstractmethod
    def batch_extract_entities(self, documents: List[Document]) -> Dict[str, List[Entity]]:
        """
        Extract entities from multiple documents efficiently.
        
        Args:
            documents: List of documents to process
            
        Returns:
            Dictionary mapping document IDs to extracted entities
        """
        pass
```

### 2. Knowledge Graph Storage Interface

```python
@dataclass
class GraphStatistics:
    """Statistics about the knowledge graph."""
    total_entities: int
    total_relationships: int
    entity_type_distribution: Dict[EntityType, int]
    relationship_type_distribution: Dict[str, int]
    avg_entities_per_document: float
    avg_relationships_per_document: float
    graph_density: float
    last_updated: datetime

@dataclass
class TraversalConfig:
    """Configuration for graph traversal."""
    max_depth: int = 2
    max_entities: int = 50
    relationship_types: Optional[List[str]] = None
    entity_types: Optional[List[EntityType]] = None
    min_confidence: float = 0.5

class IKnowledgeGraphStorage(ABC):
    """Interface for knowledge graph storage operations."""
    
    @abstractmethod
    def store_entities(self, entities: List[Entity]) -> List[str]:
        """
        Store entities in the knowledge graph.
        
        Args:
            entities: List of entities to store
            
        Returns:
            List of entity IDs that were stored
            
        Raises:
            StorageError: If storage operation fails
        """
        pass
    
    @abstractmethod
    def store_relationships(self, relationships: List[Relationship]) -> List[str]:
        """
        Store relationships in the knowledge graph.
        
        Args:
            relationships: List of relationships to store
            
        Returns:
            List of relationship IDs that were stored
            
        Raises:
            StorageError: If storage operation fails
        """
        pass
    
    @abstractmethod
    def get_entities_by_name(self, entity_name: str, 
                           entity_type: Optional[EntityType] = None) -> List[Entity]:
        """
        Find entities by name and optionally type.
        
        Args:
            entity_name: Name to search for
            entity_type: Optional entity type filter
            
        Returns:
            List of matching entities
        """
        pass
    
    @abstractmethod
    def get_related_entities(self, entity_id: str, 
                           config: TraversalConfig) -> List[Entity]:
        """
        Get entities related to the given entity.
        
        Args:
            entity_id: Starting entity ID
            config: Traversal configuration
            
        Returns:
            List of related entities
        """
        pass
    
    @abstractmethod
    def traverse_graph(self, start_entities: List[str], 
                      config: TraversalConfig) -> List[str]:
        """
        Traverse the graph from starting entities.
        
        Args:
            start_entities: List of entity IDs to start from
            config: Traversal configuration
            
        Returns:
            List of entity IDs found during traversal
        """
        pass
    
    @abstractmethod
    def get_graph_statistics(self) -> GraphStatistics:
        """
        Get statistics about the knowledge graph.
        
        Returns:
            Graph statistics
        """
        pass
    
    @abstractmethod
    def cleanup_orphaned_entities(self) -> int:
        """
        Remove entities with no relationships.
        
        Returns:
            Number of entities removed
        """
        pass
```

### 3. Entity Linking Interface

```python
@dataclass
class LinkingCandidate:
    """Candidate for entity linking."""
    entity_id: str
    canonical_name: str
    similarity_score: float
    source: str  # "knowledge_base", "external_api", etc.

class IEntityLinker(ABC):
    """Interface for entity linking and disambiguation."""
    
    @abstractmethod
    def link_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Link entities to canonical forms.
        
        Args:
            entities: List of entities to link
            
        Returns:
            List of entities with updated canonical names
        """
        pass
    
    @abstractmethod
    def find_linking_candidates(self, entity: Entity, 
                              threshold: float = 0.8) -> List[LinkingCandidate]:
        """
        Find possible canonical forms for an entity.
        
        Args:
            entity: Entity to find candidates for
            threshold: Minimum similarity threshold
            
        Returns:
            List of linking candidates
        """
        pass
    
    @abstractmethod
    def add_canonical_entity(self, canonical_name: str, 
                           entity_type: EntityType, 
                           aliases: List[str]) -> str:
        """
        Add a canonical entity to the knowledge base.
        
        Args:
            canonical_name: Primary name for the entity
            entity_type: Type of entity
            aliases: Alternative names
            
        Returns:
            Canonical entity ID
        """
        pass
```

### 4. Validation Interface

```python
@dataclass
class ValidationResult:
    """Result of validation operation."""
    success: bool
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]

class IGraphRAGValidator(ABC):
    """Interface for GraphRAG validation operations."""
    
    @abstractmethod
    def validate_prerequisites(self) -> ValidationResult:
        """
        Validate that GraphRAG prerequisites are met.
        
        Returns:
            Validation result with errors/warnings
        """
        pass
    
    @abstractmethod
    def validate_graph_quality(self) -> ValidationResult:
        """
        Validate the quality of the knowledge graph.
        
        Returns:
            Validation result with quality metrics
        """
        pass
    
    @abstractmethod
    def validate_performance(self) -> ValidationResult:
        """
        Validate GraphRAG performance metrics.
        
        Returns:
            Validation result with performance data
        """
        pass
```

## Service Implementation Architecture

### 1. Entity Extraction Service

```python
class EntityExtractionService(IEntityExtractor):
    """
    Main entity extraction service with pluggable extractors.
    
    Coordinates multiple extraction strategies and provides
    unified interface for entity extraction.
    """
    
    def __init__(self, 
                 config_manager: ConfigurationManager,
                 llm_extractor: Optional[LLMEntityExtractor] = None,
                 nlp_extractor: Optional[NLPEntityExtractor] = None,
                 pattern_extractor: Optional[PatternEntityExtractor] = None):
        """
        Initialize with pluggable extractors.
        
        Args:
            config_manager: Configuration manager
            llm_extractor: Optional LLM-based extractor
            nlp_extractor: Optional NLP-based extractor  
            pattern_extractor: Optional pattern-based extractor
        """
        self.config_manager = config_manager
        self.extractors = {}
        
        if llm_extractor:
            self.extractors['llm'] = llm_extractor
        if nlp_extractor:
            self.extractors['nlp'] = nlp_extractor
        if pattern_extractor:
            self.extractors['pattern'] = pattern_extractor
            
        self.default_strategy = config_manager.get(
            'entity_extraction.default_strategy', 'nlp'
        )
        
        # Performance tracking
        self.extraction_times = []
        self.error_counts = defaultdict(int)
```

### 2. LLM-Based Entity Extractor

```python
class LLMEntityExtractor(IEntityExtractor):
    """
    LLM-based entity extraction using structured prompts.
    
    Uses language models to extract entities with high accuracy
    but potentially lower performance.
    """
    
    def __init__(self, 
                 llm_func: Callable[[str], str],
                 config: Dict[str, Any]):
        """
        Initialize LLM extractor.
        
        Args:
            llm_func: Function to call LLM
            config: Configuration for prompts and parsing
        """
        self.llm_func = llm_func
        self.config = config
        self.prompt_template = config.get('prompt_template', self._default_prompt())
        self.max_retries = config.get('max_retries', 3)
        self.rate_limit_delay = config.get('rate_limit_delay', 1.0)
    
    def _default_prompt(self) -> str:
        """Get default extraction prompt."""
        return """
        Extract entities from the following medical text. 
        Return a JSON list with entity_name, entity_type, confidence, char_start, char_end.
        
        Entity types: PERSON, ORG, DISEASE, DRUG, TREATMENT, SYMPTOM, GENE, PROTEIN
        
        Text: {text}
        
        Entities:
        """
```

### 3. NLP-Based Entity Extractor

```python
class NLPEntityExtractor(IEntityExtractor):
    """
    NLP-based entity extraction using spaCy/NLTK.
    
    Fast, reliable extraction using traditional NLP techniques
    with domain-specific models.
    """
    
    def __init__(self, 
                 model_name: str = "en_core_web_sm",
                 custom_patterns: Optional[Dict[str, List[str]]] = None):
        """
        Initialize NLP extractor.
        
        Args:
            model_name: spaCy model name
            custom_patterns: Custom patterns for entity types
        """
        import spacy
        from spacy.matcher import Matcher
        
        self.nlp = spacy.load(model_name)
        self.matcher = Matcher(self.nlp.vocab)
        
        if custom_patterns:
            self._add_custom_patterns(custom_patterns)
```

## Configuration Schema

```yaml
# GraphRAG Entity Extraction Configuration
graphrag:
  entity_extraction:
    # Default extraction strategy
    default_strategy: "hybrid"  # llm, nlp, pattern, hybrid
    
    # Strategy-specific configurations
    strategies:
      llm:
        model: "gpt-4"
        prompt_template: "custom_biomedical_prompt.txt"
        max_retries: 3
        rate_limit_delay: 1.0
        batch_size: 5
        
      nlp:
        model: "en_core_web_sm"
        custom_patterns:
          DRUG: ["aspirin", "ibuprofen", "acetaminophen"]
          DISEASE: ["diabetes", "hypertension", "covid-19"]
        confidence_threshold: 0.7
        
      pattern:
        regex_patterns:
          GENE: "\\b[A-Z]{2,}[0-9]*\\b"
          PROTEIN: "\\bp[0-9]+\\b"
        
    # Domain configuration
    domain:
      name: "biomedical"
      entity_types: ["PERSON", "ORG", "DISEASE", "DRUG", "TREATMENT", "SYMPTOM"]
      confidence_thresholds:
        DISEASE: 0.8
        DRUG: 0.9
        default: 0.7
      
    # Performance settings
    performance:
      batch_size: 10
      max_concurrent: 5
      cache_enabled: true
      cache_ttl: 3600  # seconds

  knowledge_graph:
    # Storage configuration  
    storage:
      connection_pool_size: 10
      query_timeout: 30
      
    # Traversal configuration
    traversal:
      max_depth: 3
      max_entities: 100
      min_confidence: 0.5
      
    # Validation thresholds
    validation:
      min_entities_per_document: 2
      min_relationships_per_document: 1
      min_graph_connectivity: 0.1
```

## Error Handling Strategy

### Exception Hierarchy

```python
class GraphRAGError(Exception):
    """Base exception for GraphRAG operations."""
    pass

class ExtractionError(GraphRAGError):
    """Error during entity/relationship extraction."""
    pass

class StorageError(GraphRAGError):
    """Error during knowledge graph storage."""
    pass

class ValidationError(GraphRAGError):
    """Error during validation operations."""
    pass

class TraversalError(GraphRAGError):
    """Error during graph traversal."""
    pass
```

### Error Recovery Patterns

```python
class EntityExtractionService:
    def extract_entities_with_fallback(self, document: Document) -> List[Entity]:
        """Extract entities with fallback strategies."""
        strategies = ['llm', 'nlp', 'pattern']
        
        for strategy in strategies:
            try:
                extractor = self.extractors.get(strategy)
                if extractor:
                    entities = extractor.extract_entities(document)
                    if entities:  # Got some results
                        return entities
            except Exception as e:
                logger.warning(f"Strategy {strategy} failed: {e}")
                self.error_counts[strategy] += 1
                continue
        
        # All strategies failed
        logger.error(f"All extraction strategies failed for document {document.id}")
        return []
```

## Integration Points

### 1. Schema Manager Extension

```python
class GraphRAGSchemaManager(SchemaManager):
    """Extended schema manager for GraphRAG tables."""
    
    def ensure_graphrag_tables(self) -> bool:
        """Ensure all GraphRAG tables exist with proper schema."""
        tables = [
            'RAG.Entities',
            'RAG.EntityRelationships', 
            'RAG.EntityEmbeddings'
        ]
        
        for table in tables:
            if not self.ensure_table_schema(table, pipeline_type='graphrag'):
                return False
        
        return True
```

### 2. Validation Extension

```python
class GraphRAGRequirements(PipelineRequirements):
    """Requirements for GraphRAG pipeline."""
    
    @property
    def pipeline_name(self) -> str:
        return "graphrag"
    
    def validate_graph_quality(self, connection) -> ValidationResult:
        """Validate knowledge graph quality."""
        errors = []
        warnings = []
        metrics = {}
        
        # Check entity count
        entity_count = self._count_entities(connection)
        metrics['entity_count'] = entity_count
        
        if entity_count < 10:
            errors.append(f"Insufficient entities: {entity_count} < 10")
        
        # Check relationship count
        relationship_count = self._count_relationships(connection)
        metrics['relationship_count'] = relationship_count
        
        if relationship_count < 5:
            errors.append(f"Insufficient relationships: {relationship_count} < 5")
        
        # Check graph connectivity
        connectivity = self._calculate_connectivity(connection)
        metrics['connectivity'] = connectivity
        
        if connectivity < 0.1:
            warnings.append(f"Low graph connectivity: {connectivity}")
        
        return ValidationResult(
            success=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metrics=metrics
        )
```

This interface design ensures clean separation of concerns, extensibility, and robust error handling while maintaining compatibility with the existing IRIS RAG framework.