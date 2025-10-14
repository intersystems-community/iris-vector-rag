#!/usr/bin/env python3
"""
Knowledge Pattern Extractor for RAG Memory Components.

Extracts reusable knowledge patterns from any RAG technique response.
Provides configurable entity and relationship extraction methods.
"""

import hashlib
import logging
import re
import time
from typing import Any, Dict, List, Optional

from iris_rag.memory.models import (
    Entity,
    KnowledgeExtractionConfig,
    KnowledgePattern,
    Relationship,
)

logger = logging.getLogger(__name__)


class KnowledgePatternExtractor:
    """
    Extract knowledge patterns from RAG responses for memory storage.

    Features:
    - Configurable entity extraction (spacy, nltk, regex)
    - Relationship extraction via dependency parsing
    - Pattern clustering and deduplication
    - Performance target: <50ms per RAG response
    """

    def __init__(self, config: KnowledgeExtractionConfig):
        """
        Initialize knowledge pattern extractor.

        Args:
            config: Configuration for extraction methods and thresholds
        """
        self.config = config
        self.entity_config = config.entity_extraction
        self.relationship_config = config.relationship_extraction
        self.pattern_config = config.pattern_clustering

        # Initialize NLP components based on configuration
        self.nlp_processor = self._initialize_nlp_processor()

        # Performance tracking
        self._extraction_times: List[float] = []

        logger.info(
            f"Knowledge extractor initialized with method: {self.entity_config['method']}"
        )

    def extract_patterns(self, rag_response: Any) -> List[KnowledgePattern]:
        """
        Extract knowledge patterns from RAG response.

        Performance target: <50ms per response

        Args:
            rag_response: RAG response object (adaptable to any format)

        Returns:
            List of extracted knowledge patterns
        """
        start_time = time.perf_counter()

        try:
            patterns = []

            # Extract content from RAG response (adaptable)
            content_text = self._extract_content_from_response(rag_response)
            query_text = self._extract_query_from_response(rag_response)
            technique = self._extract_technique_from_response(rag_response)

            if not content_text:
                return patterns

            # Extract entities
            entities = self.extract_entities(content_text)

            # Extract relationships
            relationships = self.extract_relationships(content_text, entities)

            # Create knowledge pattern
            if entities or relationships:
                pattern = KnowledgePattern(
                    pattern_id=self._generate_pattern_id(content_text, query_text),
                    pattern_type=self._determine_pattern_type(entities, relationships),
                    source_rag_technique=technique,
                    entities=entities,
                    relationships=relationships,
                    context_summary=content_text[:200],  # First 200 chars
                    query_context=query_text,
                )
                patterns.append(pattern)

            # Track performance
            processing_time = (time.perf_counter() - start_time) * 1000
            self._extraction_times.append(processing_time)

            # Log performance warning if target exceeded
            if processing_time > 50.0:  # 50ms target
                logger.warning(
                    f"Knowledge extraction took {processing_time:.2f}ms (target: <50ms)"
                )

            logger.debug(
                f"Extracted {len(patterns)} patterns with {len(entities)} entities, "
                f"{len(relationships)} relationships in {processing_time:.2f}ms"
            )

            return patterns

        except Exception as e:
            logger.error(f"Error extracting knowledge patterns: {e}")
            return []

    def extract_entities(self, content: str) -> List[Entity]:
        """
        Extract entities using configurable NER approaches.

        Args:
            content: Text content to extract entities from

        Returns:
            List of extracted entities
        """
        try:
            method = self.entity_config.get("method", "spacy")
            confidence_threshold = self.entity_config.get("confidence_threshold", 0.8)
            max_entities = self.entity_config.get("max_entities_per_document", 50)

            if method == "spacy":
                entities = self._extract_entities_spacy(content, confidence_threshold)
            elif method == "regex":
                entities = self._extract_entities_regex(content)
            else:
                # Fallback to simple regex
                entities = self._extract_entities_regex(content)

            # Limit number of entities and deduplicate
            entities = self._deduplicate_entities(entities)
            return entities[:max_entities]

        except Exception as e:
            logger.warning(f"Error extracting entities: {e}")
            return []

    def extract_relationships(
        self, content: str, entities: List[Entity]
    ) -> List[Relationship]:
        """
        Extract relationships between entities.

        Args:
            content: Text content to analyze
            entities: List of entities to find relationships between

        Returns:
            List of extracted relationships
        """
        try:
            method = self.relationship_config.get("method", "dependency_parsing")
            max_distance = self.relationship_config.get("max_distance", 3)
            confidence_threshold = self.relationship_config.get(
                "confidence_threshold", 0.7
            )

            if method == "dependency_parsing" and self.nlp_processor:
                relationships = self._extract_relationships_dependency(
                    content, entities, max_distance
                )
            else:
                # Fallback to pattern-based extraction
                relationships = self._extract_relationships_pattern(content, entities)

            # Filter by confidence
            relationships = [
                r for r in relationships if r.confidence_score >= confidence_threshold
            ]

            return relationships

        except Exception as e:
            logger.warning(f"Error extracting relationships: {e}")
            return []

    def cluster_similar_patterns(
        self, patterns: List[KnowledgePattern]
    ) -> List[List[KnowledgePattern]]:
        """
        Cluster similar knowledge patterns for deduplication.

        Args:
            patterns: List of patterns to cluster

        Returns:
            List of pattern clusters
        """
        try:
            similarity_threshold = self.pattern_config.get("similarity_threshold", 0.85)
            max_cluster_size = self.pattern_config.get("max_cluster_size", 20)

            clusters = []
            processed = set()

            for i, pattern in enumerate(patterns):
                if i in processed:
                    continue

                cluster = [pattern]
                processed.add(i)

                for j, other_pattern in enumerate(patterns[i + 1 :], i + 1):
                    if j in processed:
                        continue

                    if (
                        self._calculate_pattern_similarity(pattern, other_pattern)
                        >= similarity_threshold
                    ):
                        cluster.append(other_pattern)
                        processed.add(j)

                        if len(cluster) >= max_cluster_size:
                            break

                clusters.append(cluster)

            return clusters

        except Exception as e:
            logger.warning(f"Error clustering patterns: {e}")
            return [[p] for p in patterns]  # Return individual patterns

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for knowledge extraction."""
        if not self._extraction_times:
            return {"status": "no_metrics_available"}

        return {
            "avg_extraction_time_ms": sum(self._extraction_times)
            / len(self._extraction_times),
            "max_extraction_time_ms": max(self._extraction_times),
            "min_extraction_time_ms": min(self._extraction_times),
            "total_extractions": len(self._extraction_times),
            "target_met_percentage": len(
                [t for t in self._extraction_times if t <= 50.0]
            )
            / len(self._extraction_times)
            * 100,
        }

    def _initialize_nlp_processor(self) -> Optional[Any]:
        """Initialize NLP processor based on configuration."""
        try:
            method = self.entity_config.get("method", "spacy")

            if method == "spacy":
                try:
                    import spacy

                    # Try to load English model, fallback to basic if not available
                    try:
                        nlp = spacy.load("en_core_web_sm")
                    except OSError:
                        logger.warning(
                            "spaCy English model not found, using blank model"
                        )
                        nlp = spacy.blank("en")
                    return nlp
                except ImportError:
                    logger.warning(
                        "spaCy not available, falling back to regex extraction"
                    )
                    return None

            return None

        except Exception as e:
            logger.warning(f"Error initializing NLP processor: {e}")
            return None

    def _extract_content_from_response(self, rag_response: Any) -> str:
        """Extract content text from RAG response (adaptable to any format)."""
        try:
            # Handle different RAG response formats
            if hasattr(rag_response, "response_text"):
                return str(rag_response.response_text)
            elif hasattr(rag_response, "answer"):
                return str(rag_response.answer)
            elif (
                hasattr(rag_response, "retrieved_docs") and rag_response.retrieved_docs
            ):
                # Combine retrieved document content
                content_parts = []
                for doc in rag_response.retrieved_docs[:3]:  # Limit to top 3
                    if isinstance(doc, dict):
                        content_parts.append(doc.get("content", ""))
                    else:
                        content_parts.append(str(doc))
                return " ".join(content_parts)
            elif isinstance(rag_response, str):
                return rag_response
            else:
                return str(rag_response)
        except Exception:
            return ""

    def _extract_query_from_response(self, rag_response: Any) -> str:
        """Extract query text from RAG response."""
        try:
            if hasattr(rag_response, "query"):
                return str(rag_response.query)
            elif hasattr(rag_response, "question"):
                return str(rag_response.question)
            return ""
        except Exception:
            return ""

    def _extract_technique_from_response(self, rag_response: Any) -> str:
        """Extract RAG technique from response."""
        try:
            if hasattr(rag_response, "technique_used"):
                return str(rag_response.technique_used)
            elif hasattr(rag_response, "method"):
                return str(rag_response.method)
            return "unknown"
        except Exception:
            return "unknown"

    def _extract_entities_spacy(
        self, content: str, confidence_threshold: float
    ) -> List[Entity]:
        """Extract entities using spaCy NER."""
        entities = []

        if not self.nlp_processor:
            return self._extract_entities_regex(content)

        try:
            doc = self.nlp_processor(content)

            for ent in doc.ents:
                # Simple confidence based on entity length and type
                confidence = min(1.0, len(ent.text) / 20.0 + 0.5)

                if confidence >= confidence_threshold:
                    entity = Entity(
                        entity_id=self._generate_entity_id(ent.text, ent.label_),
                        name=ent.text,
                        entity_type=ent.label_,
                        confidence_score=confidence,
                        properties={"start": ent.start_char, "end": ent.end_char},
                    )
                    entities.append(entity)

        except Exception as e:
            logger.warning(f"Error in spaCy entity extraction: {e}")
            return self._extract_entities_regex(content)

        return entities

    def _extract_entities_regex(self, content: str) -> List[Entity]:
        """Extract entities using regex patterns (fallback method)."""
        entities = []

        # Simple patterns for common entity types
        patterns = {
            "PERSON": r"\b[A-Z][a-z]+ [A-Z][a-z]+\b",
            "DATE": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b",
            "MONEY": r"\$\d+(?:,\d{3})*(?:\.\d{2})?",
            "ORG": r"\b[A-Z][A-Za-z]+ (?:Inc|Corp|LLC|Ltd|Company|Organization)\b",
            "PERCENT": r"\d+(?:\.\d+)?%",
        }

        for entity_type, pattern in patterns.items():
            matches = re.finditer(pattern, content)
            for match in matches:
                entity_text = match.group().strip()
                if len(entity_text) > 2:  # Filter very short matches
                    entity = Entity(
                        entity_id=self._generate_entity_id(entity_text, entity_type),
                        name=entity_text,
                        entity_type=entity_type,
                        confidence_score=0.7,  # Default confidence for regex
                        properties={"start": match.start(), "end": match.end()},
                    )
                    entities.append(entity)

        return entities

    def _extract_relationships_dependency(
        self, content: str, entities: List[Entity], max_distance: int
    ) -> List[Relationship]:
        """Extract relationships using dependency parsing."""
        relationships = []

        if not self.nlp_processor:
            return self._extract_relationships_pattern(content, entities)

        try:
            doc = self.nlp_processor(content)
            entity_spans = {ent.name: ent for ent in entities}

            for token in doc:
                if (
                    token.dep_ in ["nsubj", "dobj", "pobj"]
                    and token.head.pos_ == "VERB"
                ):
                    # Find entities near this token
                    source_entity = None
                    target_entity = None

                    for ent_name, entity in entity_spans.items():
                        if (
                            abs(token.i - entity.properties.get("start", 0))
                            <= max_distance
                        ):
                            if source_entity is None:
                                source_entity = entity
                            elif target_entity is None:
                                target_entity = entity
                                break

                    if (
                        source_entity
                        and target_entity
                        and source_entity != target_entity
                    ):
                        relationship = Relationship(
                            relationship_id=self._generate_relationship_id(
                                source_entity.entity_id,
                                target_entity.entity_id,
                                token.head.lemma_,
                            ),
                            source_entity_id=source_entity.entity_id,
                            target_entity_id=target_entity.entity_id,
                            relationship_type=token.head.lemma_,
                            confidence_score=0.8,
                        )
                        relationships.append(relationship)

        except Exception as e:
            logger.warning(f"Error in dependency parsing: {e}")
            return self._extract_relationships_pattern(content, entities)

        return relationships

    def _extract_relationships_pattern(
        self, content: str, entities: List[Entity]
    ) -> List[Relationship]:
        """Extract relationships using pattern matching (fallback)."""
        relationships = []

        # Simple pattern: Entity1 [verb] Entity2
        verb_patterns = [
            "is",
            "was",
            "has",
            "contains",
            "includes",
            "causes",
            "affects",
        ]

        for i, entity1 in enumerate(entities):
            for entity2 in entities[i + 1 :]:
                for verb in verb_patterns:
                    # Look for patterns like "Entity1 verb Entity2"
                    pattern = rf"\b{re.escape(entity1.name)}\s+{verb}\s+{re.escape(entity2.name)}\b"
                    if re.search(pattern, content, re.IGNORECASE):
                        relationship = Relationship(
                            relationship_id=self._generate_relationship_id(
                                entity1.entity_id, entity2.entity_id, verb
                            ),
                            source_entity_id=entity1.entity_id,
                            target_entity_id=entity2.entity_id,
                            relationship_type=verb,
                            confidence_score=0.6,
                        )
                        relationships.append(relationship)
                        break

        return relationships

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities based on canonical form."""
        seen_canonical = set()
        deduplicated = []

        for entity in entities:
            canonical = entity.canonical_form.lower()
            if canonical not in seen_canonical:
                seen_canonical.add(canonical)
                deduplicated.append(entity)

        return deduplicated

    def _calculate_pattern_similarity(
        self, pattern1: KnowledgePattern, pattern2: KnowledgePattern
    ) -> float:
        """Calculate similarity between two knowledge patterns."""
        # Simple similarity based on entity and relationship overlap
        entities1 = set(e.canonical_form for e in pattern1.entities)
        entities2 = set(e.canonical_form for e in pattern2.entities)

        if not entities1 and not entities2:
            return 1.0

        entity_similarity = (
            len(entities1 & entities2) / len(entities1 | entities2)
            if entities1 or entities2
            else 0.0
        )

        # Factor in context similarity (simple word overlap)
        context1_words = set(pattern1.context_summary.lower().split())
        context2_words = set(pattern2.context_summary.lower().split())
        context_similarity = (
            len(context1_words & context2_words) / len(context1_words | context2_words)
            if context1_words or context2_words
            else 0.0
        )

        return (entity_similarity + context_similarity) / 2.0

    def _generate_pattern_id(self, content: str, query: str) -> str:
        """Generate deterministic pattern ID."""
        content_hash = hashlib.sha256(f"{content}:{query}".encode()).hexdigest()
        return f"pattern_{content_hash[:16]}"

    def _generate_entity_id(self, name: str, entity_type: str) -> str:
        """Generate deterministic entity ID."""
        entity_hash = hashlib.sha256(
            f"{name.lower()}:{entity_type}".encode()
        ).hexdigest()
        return f"entity_{entity_hash[:16]}"

    def _generate_relationship_id(
        self, source_id: str, target_id: str, rel_type: str
    ) -> str:
        """Generate deterministic relationship ID."""
        rel_hash = hashlib.sha256(
            f"{source_id}:{target_id}:{rel_type}".encode()
        ).hexdigest()
        return f"rel_{rel_hash[:16]}"

    def _determine_pattern_type(
        self, entities: List[Entity], relationships: List[Relationship]
    ) -> str:
        """Determine pattern type based on content."""
        if len(relationships) > len(entities):
            return "relationship_pattern"
        elif len(entities) > 3:
            return "entity_cluster"
        else:
            return "concept_pattern"
