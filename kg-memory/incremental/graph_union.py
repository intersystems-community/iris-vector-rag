#!/usr/bin/env python3
"""
Graph Union Operations for LightRAG-inspired incremental indexing.

This module implements V̂ ∪ V̂' (entity union) and Ê ∪ Ê' (relationship union) operations
for efficient knowledge graph updates with atomic transactions and rollback capability.
"""

import logging
import time
import json
from dataclasses import dataclass, field
from typing import Set, List, Dict, Any, Optional, Tuple
from datetime import datetime

from iris_rag.core.connection import ConnectionManager
from iris_rag.storage.schema_manager import SchemaManager

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents a knowledge graph entity."""
    entity_id: str
    entity_name: str
    entity_type: str
    canonical_name: Optional[str] = None
    description: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    confidence_score: float = 1.0
    source_documents: List[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.canonical_name is None:
            self.canonical_name = self.entity_name
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()


@dataclass
class Relationship:
    """Represents a knowledge graph relationship."""
    relationship_id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 1.0
    source_documents: List[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()


@dataclass
class EntityUnionResult:
    """Result of entity union operation."""
    new_entities: List[Entity] = field(default_factory=list)
    updated_entities: List[Entity] = field(default_factory=list)
    unchanged_entities: List[Entity] = field(default_factory=list)
    total_processed: int = 0
    processing_time_ms: float = 0.0
    conflicts_resolved: int = 0
    
    @property
    def total_changes(self) -> int:
        return len(self.new_entities) + len(self.updated_entities)


@dataclass
class RelationshipUnionResult:
    """Result of relationship union operation."""
    new_relationships: List[Relationship] = field(default_factory=list)
    updated_relationships: List[Relationship] = field(default_factory=list)
    unchanged_relationships: List[Relationship] = field(default_factory=list)
    total_processed: int = 0
    processing_time_ms: float = 0.0
    conflicts_resolved: int = 0
    
    @property
    def total_changes(self) -> int:
        return len(self.new_relationships) + len(self.updated_relationships)


class GraphUnionOperator:
    """
    Graph Union Operator for V̂ ∪ V̂' and Ê ∪ Ê' operations.
    
    Features:
    - Entity union operations preserving node properties and relationships
    - Relationship union operations preserving edge weights and metadata
    - Atomic transactions with rollback capability
    - Integration with existing RAG.Entities and RAG.EntityRelationships schemas
    - Performance target: <30s for 1K entities
    """
    
    def __init__(self, connection_manager: ConnectionManager, schema_manager: SchemaManager):
        """
        Initialize the Graph Union Operator.
        
        Args:
            connection_manager: Database connection manager
            schema_manager: Schema manager for table operations
        """
        self.connection_manager = connection_manager
        self.schema_manager = schema_manager
        
        # Performance tracking
        self._entity_union_times: List[float] = []
        self._relationship_union_times: List[float] = []
        
        # Ensure required tables exist
        self._ensure_graph_tables()
        
        logger.info("Graph Union Operator initialized")
    
    def union_entities(self, existing_entities: Set[Entity], new_entities: Set[Entity]) -> EntityUnionResult:
        """
        Perform V̂ ∪ V̂' entity union operation.
        
        Args:
            existing_entities: Set of existing entities
            new_entities: Set of new entities to union
            
        Returns:
            EntityUnionResult with operation details
        """
        start_time = time.perf_counter()
        
        try:
            result = EntityUnionResult()
            
            # Convert sets to dicts for efficient lookup by entity_id
            existing_dict = {entity.entity_id: entity for entity in existing_entities}
            new_dict = {entity.entity_id: entity for entity in new_entities}
            
            all_entity_ids = set(existing_dict.keys()) | set(new_dict.keys())
            result.total_processed = len(all_entity_ids)
            
            for entity_id in all_entity_ids:
                existing_entity = existing_dict.get(entity_id)
                new_entity = new_dict.get(entity_id)
                
                if existing_entity is None and new_entity is not None:
                    # New entity
                    result.new_entities.append(new_entity)
                    
                elif existing_entity is not None and new_entity is None:
                    # Existing entity remains unchanged
                    result.unchanged_entities.append(existing_entity)
                    
                elif existing_entity is not None and new_entity is not None:
                    # Potential update - merge entities
                    merged_entity = self._merge_entities(existing_entity, new_entity)
                    
                    if self._entities_differ(existing_entity, merged_entity):
                        result.updated_entities.append(merged_entity)
                        if self._has_entity_conflict(existing_entity, new_entity):
                            result.conflicts_resolved += 1
                    else:
                        result.unchanged_entities.append(existing_entity)
            
            result.processing_time_ms = (time.perf_counter() - start_time) * 1000
            self._entity_union_times.append(result.processing_time_ms)
            
            logger.info(f"Entity union completed: {result.total_changes} changes, "
                       f"{result.conflicts_resolved} conflicts resolved in {result.processing_time_ms:.2f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in entity union operation: {e}")
            raise
    
    def union_relationships(self, existing_rels: Set[Relationship], new_rels: Set[Relationship]) -> RelationshipUnionResult:
        """
        Perform Ê ∪ Ê' relationship union operation.
        
        Args:
            existing_rels: Set of existing relationships
            new_rels: Set of new relationships to union
            
        Returns:
            RelationshipUnionResult with operation details
        """
        start_time = time.perf_counter()
        
        try:
            result = RelationshipUnionResult()
            
            # Convert sets to dicts for efficient lookup by relationship_id
            existing_dict = {rel.relationship_id: rel for rel in existing_rels}
            new_dict = {rel.relationship_id: rel for rel in new_rels}
            
            all_rel_ids = set(existing_dict.keys()) | set(new_dict.keys())
            result.total_processed = len(all_rel_ids)
            
            for rel_id in all_rel_ids:
                existing_rel = existing_dict.get(rel_id)
                new_rel = new_dict.get(rel_id)
                
                if existing_rel is None and new_rel is not None:
                    # New relationship
                    result.new_relationships.append(new_rel)
                    
                elif existing_rel is not None and new_rel is None:
                    # Existing relationship remains unchanged
                    result.unchanged_relationships.append(existing_rel)
                    
                elif existing_rel is not None and new_rel is not None:
                    # Potential update - merge relationships
                    merged_rel = self._merge_relationships(existing_rel, new_rel)
                    
                    if self._relationships_differ(existing_rel, merged_rel):
                        result.updated_relationships.append(merged_rel)
                        if self._has_relationship_conflict(existing_rel, new_rel):
                            result.conflicts_resolved += 1
                    else:
                        result.unchanged_relationships.append(existing_rel)
            
            result.processing_time_ms = (time.perf_counter() - start_time) * 1000
            self._relationship_union_times.append(result.processing_time_ms)
            
            logger.info(f"Relationship union completed: {result.total_changes} changes, "
                       f"{result.conflicts_resolved} conflicts resolved in {result.processing_time_ms:.2f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in relationship union operation: {e}")
            raise
    
    def atomic_graph_update(self, entity_updates: EntityUnionResult, 
                           relationship_updates: RelationshipUnionResult) -> bool:
        """
        Perform atomic graph update with rollback capability.
        
        Args:
            entity_updates: Entity union results to apply
            relationship_updates: Relationship union results to apply
            
        Returns:
            True if update successful, False otherwise
        """
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()
        
        try:
            # Start transaction
            connection.autocommit = False
            
            # Apply entity updates
            entity_success = self._apply_entity_updates(cursor, entity_updates)
            if not entity_success:
                raise Exception("Entity updates failed")
            
            # Apply relationship updates
            relationship_success = self._apply_relationship_updates(cursor, relationship_updates)
            if not relationship_success:
                raise Exception("Relationship updates failed")
            
            # Commit transaction
            connection.commit()
            
            logger.info(f"Atomic graph update successful: {entity_updates.total_changes} entity changes, "
                       f"{relationship_updates.total_changes} relationship changes")
            
            return True
            
        except Exception as e:
            logger.error(f"Atomic graph update failed: {e}")
            connection.rollback()
            return False
        finally:
            connection.autocommit = True
            cursor.close()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for graph union operations."""
        return {
            "entity_union_performance": {
                "avg_time_ms": sum(self._entity_union_times) / len(self._entity_union_times) if self._entity_union_times else 0,
                "max_time_ms": max(self._entity_union_times) if self._entity_union_times else 0,
                "total_operations": len(self._entity_union_times)
            },
            "relationship_union_performance": {
                "avg_time_ms": sum(self._relationship_union_times) / len(self._relationship_union_times) if self._relationship_union_times else 0,
                "max_time_ms": max(self._relationship_union_times) if self._relationship_union_times else 0,
                "total_operations": len(self._relationship_union_times)
            }
        }
    
    def _merge_entities(self, existing: Entity, new: Entity) -> Entity:
        """Merge two entities, with new entity taking precedence for conflicts."""
        merged = Entity(
            entity_id=existing.entity_id,
            entity_name=new.entity_name if new.entity_name != existing.entity_name else existing.entity_name,
            entity_type=new.entity_type if new.entity_type != existing.entity_type else existing.entity_type,
            canonical_name=new.canonical_name if new.canonical_name != existing.canonical_name else existing.canonical_name,
            description=new.description if new.description else existing.description,
            properties={**existing.properties, **new.properties},  # Merge properties
            embedding=new.embedding if new.embedding else existing.embedding,
            confidence_score=max(existing.confidence_score, new.confidence_score),
            source_documents=list(set(existing.source_documents + new.source_documents)),
            created_at=existing.created_at,
            updated_at=new.updated_at
        )
        return merged
    
    def _merge_relationships(self, existing: Relationship, new: Relationship) -> Relationship:
        """Merge two relationships, with new relationship taking precedence for conflicts."""
        merged = Relationship(
            relationship_id=existing.relationship_id,
            source_entity_id=existing.source_entity_id,
            target_entity_id=existing.target_entity_id,
            relationship_type=new.relationship_type if new.relationship_type != existing.relationship_type else existing.relationship_type,
            weight=max(existing.weight, new.weight),  # Take higher weight
            metadata={**existing.metadata, **new.metadata},  # Merge metadata
            confidence_score=max(existing.confidence_score, new.confidence_score),
            source_documents=list(set(existing.source_documents + new.source_documents)),
            created_at=existing.created_at,
            updated_at=new.updated_at
        )
        return merged
    
    def _entities_differ(self, entity1: Entity, entity2: Entity) -> bool:
        """Check if two entities differ in any significant way."""
        return (
            entity1.entity_name != entity2.entity_name or
            entity1.entity_type != entity2.entity_type or
            entity1.canonical_name != entity2.canonical_name or
            entity1.description != entity2.description or
            entity1.properties != entity2.properties or
            entity1.confidence_score != entity2.confidence_score or
            set(entity1.source_documents) != set(entity2.source_documents)
        )
    
    def _relationships_differ(self, rel1: Relationship, rel2: Relationship) -> bool:
        """Check if two relationships differ in any significant way."""
        return (
            rel1.relationship_type != rel2.relationship_type or
            rel1.weight != rel2.weight or
            rel1.metadata != rel2.metadata or
            rel1.confidence_score != rel2.confidence_score or
            set(rel1.source_documents) != set(rel2.source_documents)
        )
    
    def _has_entity_conflict(self, existing: Entity, new: Entity) -> bool:
        """Check if there's a conflict between existing and new entity."""
        return (
            existing.entity_name != new.entity_name or
            existing.entity_type != new.entity_type or
            existing.canonical_name != new.canonical_name
        )
    
    def _has_relationship_conflict(self, existing: Relationship, new: Relationship) -> bool:
        """Check if there's a conflict between existing and new relationship."""
        return (
            existing.relationship_type != new.relationship_type or
            abs(existing.weight - new.weight) > 0.1
        )
    
    def _apply_entity_updates(self, cursor, entity_updates: EntityUnionResult) -> bool:
        """Apply entity updates to the database."""
        try:
            # Insert new entities
            for entity in entity_updates.new_entities:
                cursor.execute("""
                    INSERT INTO RAG.Entities 
                    (entity_id, entity_name, canonical_name, entity_type, description, 
                     properties, confidence_score, source_documents, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    entity.entity_id,
                    entity.entity_name,
                    entity.canonical_name,
                    entity.entity_type,
                    entity.description,
                    json.dumps(entity.properties) if entity.properties else None,
                    entity.confidence_score,
                    json.dumps(entity.source_documents) if entity.source_documents else None,
                    entity.created_at,
                    entity.updated_at
                ])
            
            # Update existing entities
            for entity in entity_updates.updated_entities:
                cursor.execute("""
                    UPDATE RAG.Entities 
                    SET entity_name = ?, canonical_name = ?, entity_type = ?, 
                        description = ?, properties = ?, confidence_score = ?,
                        source_documents = ?, updated_at = ?
                    WHERE entity_id = ?
                """, [
                    entity.entity_name,
                    entity.canonical_name,
                    entity.entity_type,
                    entity.description,
                    json.dumps(entity.properties) if entity.properties else None,
                    entity.confidence_score,
                    json.dumps(entity.source_documents) if entity.source_documents else None,
                    entity.updated_at,
                    entity.entity_id
                ])
            
            logger.debug(f"Applied {len(entity_updates.new_entities)} new and "
                        f"{len(entity_updates.updated_entities)} updated entities")
            return True
            
        except Exception as e:
            logger.error(f"Error applying entity updates: {e}")
            return False
    
    def _apply_relationship_updates(self, cursor, relationship_updates: RelationshipUnionResult) -> bool:
        """Apply relationship updates to the database."""
        try:
            # Insert new relationships
            for rel in relationship_updates.new_relationships:
                cursor.execute("""
                    INSERT INTO RAG.EntityRelationships 
                    (relationship_id, source_entity_id, target_entity_id, relationship_type,
                     weight, metadata, confidence_score, source_documents, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    rel.relationship_id,
                    rel.source_entity_id,
                    rel.target_entity_id,
                    rel.relationship_type,
                    rel.weight,
                    json.dumps(rel.metadata) if rel.metadata else None,
                    rel.confidence_score,
                    json.dumps(rel.source_documents) if rel.source_documents else None,
                    rel.created_at,
                    rel.updated_at
                ])
            
            # Update existing relationships
            for rel in relationship_updates.updated_relationships:
                cursor.execute("""
                    UPDATE RAG.EntityRelationships 
                    SET relationship_type = ?, weight = ?, metadata = ?,
                        confidence_score = ?, source_documents = ?, updated_at = ?
                    WHERE relationship_id = ?
                """, [
                    rel.relationship_type,
                    rel.weight,
                    json.dumps(rel.metadata) if rel.metadata else None,
                    rel.confidence_score,
                    json.dumps(rel.source_documents) if rel.source_documents else None,
                    rel.updated_at,
                    rel.relationship_id
                ])
            
            logger.debug(f"Applied {len(relationship_updates.new_relationships)} new and "
                        f"{len(relationship_updates.updated_relationships)} updated relationships")
            return True
            
        except Exception as e:
            logger.error(f"Error applying relationship updates: {e}")
            return False
    
    def _ensure_graph_tables(self):
        """Ensure required graph tables exist."""
        try:
            # This will be handled by schema extensions
            self.schema_manager.ensure_table_schema("Entities")
            self.schema_manager.ensure_table_schema("EntityRelationships")
            logger.debug("Graph tables ensured")
            
        except Exception as e:
            logger.warning(f"Could not ensure graph tables: {e}")
            # Tables will be created by schema extensions