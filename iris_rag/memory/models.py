#!/usr/bin/env python3
"""
Generic memory models for RAG applications.

These models provide extensible data structures that applications can adapt
for their specific memory requirements.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from enum import Enum


class MemoryType(Enum):
    """Types of memory items."""
    KNOWLEDGE_PATTERN = "knowledge_pattern"
    TEMPORAL_CONTEXT = "temporal_context"
    ENTITY = "entity"
    RELATIONSHIP = "relationship"
    QUERY_CONTEXT = "query_context"
    LEARNED_PATTERN = "learned_pattern"


class TemporalWindow(Enum):
    """Temporal window configurations."""
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"
    PERMANENT = "permanent"


@dataclass
class GenericMemoryItem:
    """
    Generic memory item that applications can extend.
    
    This base structure provides common memory patterns while allowing
    applications to add domain-specific data.
    """
    memory_id: str
    content: Dict[str, Any]
    memory_type: MemoryType
    confidence_score: float = 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Applications can extend this base structure
    application_data: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if memory item has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def update_confidence(self, new_confidence: float) -> None:
        """Update confidence score and timestamp."""
        self.confidence_score = new_confidence
        self.updated_at = datetime.utcnow()


@dataclass
class Entity:
    """Generic entity representation."""
    entity_id: str
    name: str
    entity_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 1.0
    source_documents: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    canonical_form: Optional[str] = None
    
    def __post_init__(self):
        """Set canonical form if not provided."""
        if self.canonical_form is None:
            self.canonical_form = self.name.lower().strip()


@dataclass
class Relationship:
    """Generic relationship representation."""
    relationship_id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 1.0
    source_documents: List[str] = field(default_factory=list)
    bidirectional: bool = False


@dataclass
class KnowledgePattern:
    """
    Generic knowledge pattern extracted from RAG responses.
    
    Represents reusable knowledge that can be applied across queries.
    """
    pattern_id: str
    pattern_type: str  # "entity_cluster", "relationship_pattern", "concept_pattern"
    source_rag_technique: str
    extraction_confidence: float = 1.0
    entities: List[Entity] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    context_summary: str = ""
    query_context: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_memory_item(self) -> GenericMemoryItem:
        """Convert to generic memory item for storage."""
        return GenericMemoryItem(
            memory_id=self.pattern_id,
            content={
                "pattern_type": self.pattern_type,
                "entities": [entity.__dict__ for entity in self.entities],
                "relationships": [rel.__dict__ for rel in self.relationships],
                "context_summary": self.context_summary,
                "query_context": self.query_context
            },
            memory_type=MemoryType.KNOWLEDGE_PATTERN,
            confidence_score=self.extraction_confidence,
            metadata={
                "source_rag_technique": self.source_rag_technique,
                "entity_count": len(self.entities),
                "relationship_count": len(self.relationships)
            }
        )


@dataclass
class TemporalWindowConfig:
    """Configuration for temporal windows."""
    name: str
    duration_days: int
    cleanup_frequency: str = "daily"  # daily, weekly, monthly
    retention_policy: str = "expire"  # expire, archive, keep
    
    def get_expiration_time(self, from_time: datetime = None) -> datetime:
        """Get expiration time for this window."""
        if from_time is None:
            from_time = datetime.utcnow()
        return from_time + timedelta(days=self.duration_days)


@dataclass
class KnowledgeExtractionConfig:
    """Configuration for knowledge extraction."""
    entity_extraction: Dict[str, Any] = field(default_factory=lambda: {
        "method": "spacy",
        "confidence_threshold": 0.8,
        "max_entities_per_document": 50
    })
    relationship_extraction: Dict[str, Any] = field(default_factory=lambda: {
        "method": "dependency_parsing",
        "max_distance": 3,
        "confidence_threshold": 0.7
    })
    pattern_clustering: Dict[str, Any] = field(default_factory=lambda: {
        "similarity_threshold": 0.85,
        "max_cluster_size": 20
    })


@dataclass
class PersistenceConfig:
    """Configuration for memory persistence."""
    storage_backend: str = "iris"  # iris, postgres, memory
    connection_config: Dict[str, Any] = field(default_factory=dict)
    batch_size: int = 100
    enable_compression: bool = True
    index_config: Dict[str, Any] = field(default_factory=lambda: {
        "entity_index": True,
        "temporal_index": True,
        "similarity_index": True
    })


@dataclass
class PerformanceConfig:
    """Configuration for memory performance."""
    cache_config: Dict[str, Any] = field(default_factory=lambda: {
        "l1_size": 1000,
        "l2_size": 10000,
        "ttl_seconds": 3600
    })
    parallel_processing: Dict[str, Any] = field(default_factory=lambda: {
        "max_workers": 4,
        "batch_size": 50
    })
    performance_targets: Dict[str, float] = field(default_factory=lambda: {
        "extraction_time_ms": 50.0,
        "retrieval_time_ms": 100.0,
        "storage_time_ms": 20.0
    })


@dataclass
class MemoryConfig:
    """
    Configuration for memory behavior.
    
    Applications can extend this configuration with custom settings.
    """
    temporal_windows: List[TemporalWindowConfig] = field(default_factory=lambda: [
        TemporalWindowConfig("short_term", 7),
        TemporalWindowConfig("medium_term", 30),
        TemporalWindowConfig("long_term", 90)
    ])
    knowledge_extraction: KnowledgeExtractionConfig = field(default_factory=KnowledgeExtractionConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # Allow applications to add custom configuration
    custom_config: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MemoryConfig':
        """Create MemoryConfig from dictionary."""
        # Extract known configuration sections
        temporal_windows = []
        if 'temporal_windows' in config_dict:
            for tw_config in config_dict['temporal_windows']:
                temporal_windows.append(TemporalWindowConfig(**tw_config))
        
        knowledge_extraction = KnowledgeExtractionConfig()
        if 'knowledge_extraction' in config_dict:
            knowledge_extraction = KnowledgeExtractionConfig(**config_dict['knowledge_extraction'])
        
        persistence = PersistenceConfig()
        if 'persistence' in config_dict:
            persistence = PersistenceConfig(**config_dict['persistence'])
        
        performance = PerformanceConfig()
        if 'performance' in config_dict:
            performance = PerformanceConfig(**config_dict['performance'])
        
        custom_config = config_dict.get('custom_config', {})
        
        return cls(
            temporal_windows=temporal_windows or cls().temporal_windows,
            knowledge_extraction=knowledge_extraction,
            persistence=persistence,
            performance=performance,
            custom_config=custom_config
        )


@dataclass
class MemoryItem:
    """Item stored in temporal memory."""
    item_id: str
    content: str
    context: Dict[str, Any]
    temporal_window: TemporalWindow
    relevance_score: float = 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    source_query: str = ""
    embeddings: Optional[List[float]] = None


@dataclass
class LearningResult:
    """Result of incremental learning operation."""
    new_patterns: List[KnowledgePattern] = field(default_factory=list)
    updated_patterns: List[KnowledgePattern] = field(default_factory=list)
    merged_entities: int = 0
    merged_relationships: int = 0
    processing_time_ms: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    
    @property
    def total_changes(self) -> int:
        """Total number of changes made."""
        return len(self.new_patterns) + len(self.updated_patterns)


@dataclass
class EnrichedRAGResponse:
    """RAG response enriched with memory context."""
    base_response: Any  # Original RAG response
    memory_context: List[MemoryItem] = field(default_factory=list)
    extracted_patterns: List[KnowledgePattern] = field(default_factory=list)
    enrichment_metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def has_memory_enhancement(self) -> bool:
        """Check if response was enhanced with memory."""
        return len(self.memory_context) > 0 or len(self.extracted_patterns) > 0