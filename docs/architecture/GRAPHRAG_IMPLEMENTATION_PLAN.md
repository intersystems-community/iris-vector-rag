# GraphRAG Knowledge Graph Implementation Plan

## Overview

This document provides a detailed implementation plan for the GraphRAG knowledge graph infrastructure, building upon the architecture defined in [`GRAPHRAG_KNOWLEDGE_GRAPH_ARCHITECTURE.md`](./GRAPHRAG_KNOWLEDGE_GRAPH_ARCHITECTURE.md).

## Implementation Strategy

### Phase-Based Approach

The implementation follows a pragmatic, incremental approach:

1. **Phase 1**: Core Infrastructure (Basic SQL-based implementation)
2. **Phase 2**: Enhanced Extraction (Domain-specific capabilities)
3. **Phase 3**: Query Engine & Optimization (Performance improvements)
4. **Phase 4**: IRIS Globals Optimization (Advanced performance)
5. **Phase 5**: Integration & Testing (Production readiness)

## Phase 1: Core Infrastructure (Weeks 1-2)

### Objectives
- Establish basic knowledge graph functionality
- Integrate with existing RAG pipeline architecture
- Provide foundation for future enhancements

### Deliverables

#### 1.1 Enhanced Schema Manager Integration

**File**: `iris_rag/storage/schema_manager.py` (extend existing)

```python
# Add to existing _build_table_configurations method
def _build_table_configurations(self):
    """Extended table configurations including knowledge graph tables."""
    # ... existing configurations ...
    
    # Knowledge graph table configurations
    self._table_configs.update({
        "Entities": {
            "embedding_column": "embedding",
            "content_column": "description", 
            "id_column": "entity_id",
            "uses_document_embeddings": True,
            "default_model": self.base_embedding_model,
            "dimension": self.base_embedding_dimension,
            "supports_vector_search": True,
            "supports_globals_optimization": True,
            "schema_definition": {
                "entity_id": "VARCHAR(255) PRIMARY KEY",
                "entity_name": "VARCHAR(500) NOT NULL",
                "canonical_name": "VARCHAR(500) NOT NULL", 
                "entity_type": "VARCHAR(100) NOT NULL",
                "description": "TEXT",
                "properties": "TEXT", # JSON
                "embedding": f"VECTOR(FLOAT, {self.base_embedding_dimension})",
                "confidence_score": "FLOAT DEFAULT 1.0",
                "source_documents": "TEXT", # JSON array
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "updated_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            }
        },
        "Relationships": {
            "embedding_column": None,
            "uses_document_embeddings": False,
            "supports_graph_traversal": True,
            "supports_globals_optimization": True,
            "optimization_strategy": "iris_globals",
            "schema_definition": {
                "relationship_id": "VARCHAR(255) PRIMARY KEY",
                "source_entity_id": "VARCHAR(255) NOT NULL",
                "target_entity_id": "VARCHAR(255) NOT NULL", 
                "relationship_type": "VARCHAR(100) NOT NULL",
                "description": "TEXT",
                "properties": "TEXT", # JSON
                "strength": "FLOAT DEFAULT 1.0",
                "confidence": "FLOAT DEFAULT 1.0",
                "evidence": "TEXT", # JSON array
                "source_documents": "TEXT", # JSON array
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "updated_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            }
        },
        "EntityMentions": {
            "embedding_column": None,
            "uses_document_embeddings": False,
            "supports_full_text_search": True,
            "schema_definition": {
                "mention_id": "VARCHAR(255) PRIMARY KEY",
                "entity_id": "VARCHAR(255) NOT NULL",
                "document_id": "VARCHAR(255) NOT NULL",
                "text": "VARCHAR(1000) NOT NULL",
                "start_offset": "INTEGER",
                "end_offset": "INTEGER", 
                "context": "TEXT",
                "confidence": "FLOAT DEFAULT 1.0",
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            }
        }
    })
```

#### 1.2 Basic Entity Extraction Service

**File**: `iris_rag/storage/knowledge_graph/entity_extractor.py`

```python
"""
Basic entity extraction service implementation.
"""

import logging
import re
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..core.models import Document
from .interfaces import IEntityExtractor
from .models import Entity, DomainConfig, ENTITY_TYPE_REGISTRY

logger = logging.getLogger(__name__)


class EntityExtractionService(IEntityExtractor):
    """Basic entity extraction with configurable complexity."""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.extraction_mode = config_manager.get("graphrag.entity_extraction.mode", "basic")
        self.domain_config = None
        
        # Load extraction patterns
        self.patterns = self._load_extraction_patterns()
        
        logger.info(f"EntityExtractionService initialized in {self.extraction_mode} mode")
    
    def extract_entities(self, document: Document) -> List[Entity]:
        """Extract entities using configured strategy."""
        if self.extraction_mode == "basic":
            return self._extract_basic_entities(document)
        elif self.extraction_mode == "pattern":
            return self._extract_pattern_entities(document)
        else:
            return self._extract_basic_entities(document)
    
    def set_domain_config(self, domain_config: DomainConfig) -> None:
        """Configure for specific domain."""
        self.domain_config = domain_config
        domain_config.apply_to_registries()
        logger.info(f"Applied domain config: {domain_config.domain_name}")
    
    def get_supported_entity_types(self) -> List[str]:
        """Get supported entity types."""
        return list(ENTITY_TYPE_REGISTRY.get_types().keys())
    
    def _extract_basic_entities(self, document: Document) -> List[Entity]:
        """Basic entity extraction using simple patterns."""
        entities = []
        text = document.page_content
        
        # Extract capitalized words/phrases as potential entities
        patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Proper nouns
            r'\b[A-Z]{2,}\b',  # Acronyms
        ]
        
        found_entities = set()
        for pattern in patterns:
            matches = re.findall(pattern, text)
            found_entities.update(matches)
        
        # Filter and create entities
        for i, entity_text in enumerate(found_entities):
            if len(entity_text) > 2 and entity_text not in ['The', 'This', 'That']:
                entity_type = self._classify_entity_type(entity_text, text)
                
                entity = Entity(
                    entity_id=f"{document.id}_entity_{i}",
                    entity_name=entity_text,
                    entity_type=entity_type,
                    canonical_name=entity_text.lower(),
                    description=self._extract_entity_context(entity_text, text),
                    confidence_score=0.7,
                    source_documents=[document.id],
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                entities.append(entity)
        
        logger.debug(f"Extracted {len(entities)} entities from document {document.id}")
        return entities
    
    def _classify_entity_type(self, entity_text: str, context: str) -> str:
        """Basic entity type classification."""
        entity_lower = entity_text.lower()
        
        # Biomedical patterns
        if self.domain_config and self.domain_config.domain_name == "biomedical":
            if any(term in entity_lower for term in ['gene', 'protein', 'enzyme']):
                return "PROTEIN"
            elif any(term in entity_lower for term in ['disease', 'syndrome', 'disorder']):
                return "DISEASE"
            elif any(term in entity_lower for term in ['drug', 'compound', 'inhibitor']):
                return "DRUG"
            elif any(term in entity_lower for term in ['cell', 'tissue', 'organ']):
                return "CELL_TYPE"
        
        # General patterns
        if entity_text.isupper() and len(entity_text) > 1:
            return "ORGANIZATION"  # Likely acronym
        elif any(term in context.lower() for term in ['university', 'hospital', 'institute']):
            return "ORGANIZATION"
        else:
            return "CONCEPT"
    
    def _extract_entity_context(self, entity_text: str, full_text: str) -> Optional[str]:
        """Extract context around entity mention."""
        # Find sentence containing the entity
        sentences = full_text.split('.')
        for sentence in sentences:
            if entity_text in sentence:
                return sentence.strip()
        return None
    
    def _load_extraction_patterns(self) -> Dict[str, Any]:
        """Load domain-specific extraction patterns."""
        return self.config_manager.get("graphrag.entity_extraction.patterns", {})
```

#### 1.3 Basic Relationship Extraction Service

**File**: `iris_rag/storage/knowledge_graph/relationship_extractor.py`

```python
"""
Basic relationship extraction service implementation.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..core.models import Document
from .interfaces import IRelationshipExtractor
from .models import Relationship, RelationshipEvidence, Entity, DomainConfig, RELATIONSHIP_TYPE_REGISTRY

logger = logging.getLogger(__name__)


class RelationshipExtractionService(IRelationshipExtractor):
    """Basic relationship extraction with co-occurrence analysis."""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.extraction_mode = config_manager.get("graphrag.relationship_extraction.mode", "cooccurrence")
        self.domain_config = None
        self.cooccurrence_window = config_manager.get("graphrag.relationship_extraction.window_size", 100)
        
        logger.info(f"RelationshipExtractionService initialized in {self.extraction_mode} mode")
    
    def extract_relationships(self, document: Document, entities: List[Entity]) -> List[Relationship]:
        """Extract relationships using configured strategy."""
        if self.extraction_mode == "cooccurrence":
            return self._extract_cooccurrence_relationships(document, entities)
        elif self.extraction_mode == "pattern":
            return self._extract_pattern_relationships(document, entities)
        else:
            return self._extract_cooccurrence_relationships(document, entities)
    
    def set_domain_config(self, domain_config: DomainConfig) -> None:
        """Configure for specific domain."""
        self.domain_config = domain_config
        domain_config.apply_to_registries()
        logger.info(f"Applied domain config: {domain_config.domain_name}")
    
    def get_supported_relationship_types(self) -> List[str]:
        """Get supported relationship types."""
        return list(RELATIONSHIP_TYPE_REGISTRY.get_types().keys())
    
    def _extract_cooccurrence_relationships(self, document: Document, entities: List[Entity]) -> List[Relationship]:
        """Extract relationships based on entity co-occurrence."""
        relationships = []
        text = document.page_content
        
        # Create entity position map
        entity_positions = {}
        for entity in entities:
            positions = []
            start = 0
            while True:
                pos = text.find(entity.entity_name, start)
                if pos == -1:
                    break
                positions.append(pos)
                start = pos + 1
            entity_positions[entity.entity_id] = positions
        
        # Find co-occurring entities within window
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                if self._entities_cooccur(entity1, entity2, entity_positions, text):
                    relationship = self._create_cooccurrence_relationship(
                        entity1, entity2, document, text)
                    if relationship:
                        relationships.append(relationship)
        
        logger.debug(f"Extracted {len(relationships)} relationships from document {document.id}")
        return relationships
    
    def _entities_cooccur(self, entity1: Entity, entity2: Entity, 
                         entity_positions: Dict[str, List[int]], text: str) -> bool:
        """Check if two entities co-occur within the window."""
        pos1_list = entity_positions.get(entity1.entity_id, [])
        pos2_list = entity_positions.get(entity2.entity_id, [])
        
        for pos1 in pos1_list:
            for pos2 in pos2_list:
                if abs(pos1 - pos2) <= self.cooccurrence_window:
                    return True
        return False
    
    def _create_cooccurrence_relationship(self, entity1: Entity, entity2: Entity, 
                                        document: Document, text: str) -> Optional[Relationship]:
        """Create a co-occurrence relationship."""
        relationship_type = self._determine_relationship_type(entity1, entity2, text)
        
        # Extract evidence
        evidence = RelationshipEvidence(
            evidence_id=f"{entity1.entity_id}_{entity2.entity_id}_evidence",
            document_id=document.id,
            text_span=f"{entity1.entity_name} ... {entity2.entity_name}",
            context=self._extract_relationship_context(entity1, entity2, text),
            confidence=0.6
        )
        
        relationship = Relationship(
            relationship_id=f"{entity1.entity_id}_{entity2.entity_id}_cooccur",
            source_entity_id=entity1.entity_id,
            target_entity_id=entity2.entity_id,
            relationship_type=relationship_type,
            strength=1.0,
            confidence=0.6,
            evidence=[evidence],
            source_documents=[document.id],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        return relationship
    
    def _determine_relationship_type(self, entity1: Entity, entity2: Entity, text: str) -> str:
        """Determine relationship type based on entity types and context."""
        # Domain-specific relationship inference
        if self.domain_config and self.domain_config.domain_name == "biomedical":
            if entity1.entity_type == "PROTEIN" and entity2.entity_type == "PROTEIN":
                return "INTERACTS_WITH"
            elif entity1.entity_type == "DRUG" and entity2.entity_type == "DISEASE":
                return "TREATS"
            elif entity1.entity_type == "GENE" and entity2.entity_type == "DISEASE":
                return "ASSOCIATED_WITH"
        
        # Default to co-occurrence
        return "CO_OCCURS"
    
    def _extract_relationship_context(self, entity1: Entity, entity2: Entity, text: str) -> str:
        """Extract context around relationship."""
        # Find sentence containing both entities
        sentences = text.split('.')
        for sentence in sentences:
            if entity1.entity_name in sentence and entity2.entity_name in sentence:
                return sentence.strip()
        return f"Co-occurrence of {entity1.entity_name} and {entity2.entity_name}"
```

### Testing Strategy for Phase 1

#### Unit Tests
- Entity extraction accuracy
- Relationship extraction logic
- Schema manager integration
- Domain configuration application

#### Integration Tests
- End-to-end graph building
- Vector store integration
- Schema creation and migration

## Phase 2: Enhanced Extraction (Weeks 3-4)

### Objectives
- Improve entity and relationship extraction accuracy
- Add domain-specific extraction capabilities
- Implement entity linking and disambiguation

### Deliverables

#### 2.1 Advanced Entity Extraction
- NER model integration (spaCy, transformers)
- Biomedical entity recognition (BioBERT, SciBERT)
- Custom pattern-based extractors

#### 2.2 Enhanced Relationship Extraction
- Dependency parsing for relationships
- Pattern-based relationship extraction
- Domain-specific relationship rules

#### 2.3 Entity Linking Service
- Entity disambiguation algorithms
- Canonical name resolution
- Cross-document entity linking

## Phase 3: Query Engine & Optimization (Weeks 5-6)

### Objectives
- Implement graph query and traversal engine
- Add performance optimization
- Create subgraph extraction algorithms

### Deliverables

#### 3.1 Graph Query Engine
**File**: `iris_rag/storage/knowledge_graph/query_engine.py`

#### 3.2 Performance Optimization
- Query result caching
- Index optimization
- Query plan optimization

#### 3.3 Subgraph Extraction
- Relevance-based subgraph extraction
- Multi-hop traversal algorithms
- Context-aware pruning

## Phase 4: IRIS Globals Optimization (Future Enhancement)

### Objectives
- Implement high-performance graph traversal using IRIS globals
- Create ObjectScript integration layer
- Optimize for complex path queries

### Deliverables

#### 4.1 ObjectScript Graph Functions
**File**: `objectscript/GraphTraversal.cls`

#### 4.2 Python-ObjectScript Bridge
**File**: `iris_rag/storage/knowledge_graph/iris_optimizer.py`

#### 4.3 Adaptive Query Planning
- Performance threshold detection
- Automatic optimization strategy selection
- Fallback mechanisms

## Phase 5: Integration & Testing (Weeks 7-8)

### Objectives
- Complete integration with existing RAG pipeline
- Comprehensive testing suite
- Performance benchmarking
- Production readiness

### Deliverables

#### 5.1 Enhanced GraphRAG Pipeline
**File**: `iris_rag/pipelines/enhanced_graphrag.py`

#### 5.2 Comprehensive Test Suite
- Unit tests for all components
- Integration tests
- Performance tests
- End-to-end tests with real data

#### 5.3 Documentation and Examples
- API documentation
- Usage examples
- Performance benchmarks
- Migration guides

## Configuration Management

### Domain Configuration Example

```yaml
# config/graphrag_biomedical.yaml
graphrag:
  domain: "biomedical"
  entity_extraction:
    mode: "advanced"  # basic, pattern, advanced
    models:
      - "en_core_sci_sm"  # spaCy biomedical model
      - "dmis-lab/biobert-base-cased-v1.1"
    patterns:
      gene_patterns:
        - r'\b[A-Z][A-Z0-9]+\b'  # Gene symbols
      protein_patterns:
        - r'\b\w+ase\b'  # Enzymes
  relationship_extraction:
    mode: "pattern"  # cooccurrence, pattern, advanced
    window_size: 50
    patterns:
      interaction_patterns:
        - "interacts with"
        - "binds to"
        - "regulates"
  performance:
    use_globals_optimization: false  # Enable in Phase 4
    cache_size: 1000
    max_traversal_depth: 3
```

## Success Metrics

### Phase 1 Metrics
- **Entity Extraction**: >70% precision/recall on test dataset
- **Relationship Extraction**: >60% accuracy on co-occurrence relationships
- **Integration**: 100% compatibility with existing RAG pipeline
- **Performance**: <30 seconds graph construction for 100 documents

### Phase 2 Metrics
- **Entity Extraction**: >85% precision/recall with NER models
- **Relationship Extraction**: >75% accuracy with pattern matching
- **Entity Linking**: >80% accuracy on entity disambiguation

### Phase 3 Metrics
- **Query Performance**: <3 seconds for complex graph queries
- **Subgraph Quality**: >85% relevance score for extracted subgraphs
- **Scalability**: Handle 10K+ entities efficiently

### Phase 4 Metrics
- **Globals Optimization**: >50% performance improvement for complex queries
- **Deep Traversal**: <1 second for 5-hop traversals
- **Large Scale**: Handle 100K+ entities with sub-second queries

## Risk Mitigation

### Technical Risks
1. **Performance Bottlenecks**: Mitigated by incremental optimization and IRIS globals strategy
2. **Entity Extraction Accuracy**: Addressed through domain-specific models and patterns
3. **Integration Complexity**: Minimized by maintaining interface compatibility

### Implementation Risks
1. **Scope Creep**: Controlled through phase-based delivery
2. **Resource Constraints**: Managed through prioritized feature delivery
3. **Testing Complexity**: Addressed through comprehensive test strategy

## Conclusion

This implementation plan provides a clear roadmap for delivering comprehensive GraphRAG knowledge graph infrastructure. The phase-based approach ensures incremental value delivery while building toward advanced capabilities that leverage IRIS's unique performance characteristics.

The design maintains full compatibility with existing RAG templates architecture while providing extensible, domain-configurable knowledge graph capabilities suitable for biomedical and other specialized applications.