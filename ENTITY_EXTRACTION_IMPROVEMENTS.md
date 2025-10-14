# GraphRAG Entity Extraction Improvement Plan

Based on research into LightRAG, Microsoft GraphRAG, and SOTA medical NER systems, this document outlines specific improvements needed for our entity extraction to reach production quality.

## Current Status Analysis

### What's Working
- ✅ Basic pattern-based extraction for DRUG, DISEASE, GENE entities
- ✅ Simple co-occurrence relationship detection
- ✅ IRIS database storage integration
- ✅ Configuration-driven entity types
- ✅ GraphRAG query pipeline integration

### Critical Gaps Identified
- ❌ **No LLM-based extraction** despite configuration options
- ❌ **Basic regex patterns** insufficient for medical domain
- ❌ **No iterative refinement** (gleaning loop pattern)
- ❌ **Limited relationship types** (only co-occurrence)
- ❌ **No entity disambiguation** across documents
- ❌ **No community detection** for graph structure
- ❌ **No temporal or factual claims**

## Research Findings Summary

### LightRAG Key Innovations
1. **Gleaning Loop**: Iterative entity extraction with retry mechanism
2. **Structured Deduplication**: Merges entities with description consolidation
3. **Key-Value Representation**: Entities become knowledge pairs with rich summaries
4. **Dual-Level Retrieval**: Handles both specific and global queries

### Microsoft GraphRAG Best Practices
1. **Parallel Processing**: Simultaneous entity/relationship extraction
2. **LLM Summarization**: Consolidates multiple entity descriptions
3. **Community Detection**: Groups related entities for retrieval
4. **Claim Extraction**: Identifies factual statements with boundaries

### Medical NER SOTA Patterns
1. **Multi-Task Learning**: Combines multiple NLP tasks for better accuracy
2. **BERT Integration**: Transformer-based contextual understanding
3. **Domain Adaptation**: Fine-tuned on medical corpora (UMLS, SNOMED)
4. **Seven Entity Categories**: From unigrams to complex tabular entities

## Implementation Roadmap

### Phase 1: Enhanced Pattern-Based Extraction (2-3 days)

#### 1.1 Expand Medical Entity Patterns
Replace current basic patterns with comprehensive medical regex:

```python
# Enhanced medical patterns based on UMLS/SNOMED
ENHANCED_PATTERNS = {
    "DRUG": [
        # Generic drug name patterns
        r'\b[A-Z][a-z]+(?:ine|ide|ate|cin|pam|zole|pril|sartan|mab|nib|tinib)\b',
        # Brand names and common drugs
        r'\b(?:[A-Z][a-z]{2,}(?:®|™)?)\s*(?:\d+\s*mg|\d+/\d+)\b',
        # Drug combinations
        r'\b[A-Z][a-z]+/[A-Z][a-z]+\b',
        # Common medications
        r'\b(?:aspirin|ibuprofen|acetaminophen|morphine|insulin|metformin|lisinopril|atorvastatin)\b'
    ],
    "DISEASE": [
        # Medical conditions with standard suffixes
        r'\b[A-Z][a-z]+(?:itis|osis|emia|pathy|syndrome|disease|disorder)\b',
        # Specific conditions
        r'\b(?:diabetes|cancer|hypertension|pneumonia|arthritis|COVID-19|influenza)\b',
        # ICD-10 patterns
        r'\b[A-Z]\d{2}(?:\.\d{1,2})?\b'
    ],
    "GENE": [
        # Gene symbols (HUGO nomenclature)
        r'\b[A-Z]{2,}[0-9]+(?:[A-Z]?)?\b',  # TP53, BRCA1, KRAS
        # Protein names
        r'\b(?:p53|BRCA1|EGFR|HER2|BCR-ABL)\b'
    ],
    "ANATOMY": [
        # Body parts and organs
        r'\b(?:heart|lung|liver|kidney|brain|spine|bone|muscle|artery|vein)\b',
        # Anatomical regions
        r'\b(?:thoracic|abdominal|pelvic|cranial|cervical|lumbar)\b'
    ],
    "PROCEDURE": [
        # Medical procedures
        r'\b(?:surgery|biopsy|chemotherapy|radiation|transplant|dialysis)\b',
        # Diagnostic procedures
        r'\b(?:MRI|CT|X-ray|ultrasound|endoscopy|colonoscopy)\b'
    ]
}
```

#### 1.2 Implement Context-Aware Extraction
Add context windows around matches for better accuracy:

```python
def extract_with_context(self, text: str, entity_type: str, pattern: str) -> List[Entity]:
    """Extract entities with surrounding context for better classification."""
    entities = []
    for match in re.finditer(pattern, text, re.IGNORECASE):
        # Get context window
        start_ctx = max(0, match.start() - 100)
        end_ctx = min(len(text), match.end() + 100)
        context = text[start_ctx:end_ctx]

        # Calculate confidence based on context
        confidence = self._calculate_context_confidence(match.group(0), context, entity_type)

        entity = Entity(
            text=match.group(0),
            entity_type=entity_type,
            confidence=confidence,
            start_offset=match.start(),
            end_offset=match.end(),
            source_document_id=document.id if document else None,
            metadata={
                "method": "enhanced_pattern",
                "context": context,
                "pattern_confidence": confidence
            }
        )
        entities.append(entity)

    return entities
```

### Phase 2: LLM-Based Entity Extraction (3-4 days)

#### 2.1 Implement Actual LLM Integration
Replace mock LLM calls with real OpenAI/Anthropic integration:

```python
def _extract_llm_entities(self, text: str, document: Optional[Document] = None) -> List[Entity]:
    """Extract entities using LLM with structured prompts."""
    try:
        from common.utils import get_llm_func

        llm_func = get_llm_func('openai', 'gpt-4o-mini')

        prompt = f"""
        Extract medical entities from the following text. Return a JSON array with this exact format:

        [
          {{
            "text": "exact text span",
            "type": "DRUG|DISEASE|GENE|ANATOMY|PROCEDURE|PERSON|ORGANIZATION",
            "confidence": 0.8,
            "start": 0,
            "end": 10,
            "context": "surrounding context"
          }}
        ]

        Text: {text[:2000]}

        Focus on medical terminology, drug names, diseases, genes, anatomical terms, and medical procedures.
        Be precise with text spans and confidence scores.
        """

        response = llm_func(prompt)
        return self._parse_structured_llm_response(response, document)

    except Exception as e:
        logger.error(f"LLM entity extraction failed: {e}")
        return []

def _parse_structured_llm_response(self, response: str, document: Optional[Document]) -> List[Entity]:
    """Parse structured LLM response with error handling."""
    entities = []
    try:
        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response.strip()

        raw_entities = json.loads(json_str)

        for raw_entity in raw_entities:
            entity = Entity(
                text=raw_entity.get("text", ""),
                entity_type=raw_entity.get("type", "UNKNOWN"),
                confidence=raw_entity.get("confidence", 0.5),
                start_offset=raw_entity.get("start", 0),
                end_offset=raw_entity.get("end", 0),
                source_document_id=document.id if document else None,
                metadata={
                    "method": "llm_structured",
                    "context": raw_entity.get("context", ""),
                    "llm_confidence": raw_entity.get("confidence", 0.5)
                }
            )
            entities.append(entity)

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.error(f"Failed to parse LLM response: {e}")
        logger.debug(f"Response was: {response[:500]}")

    return entities
```

#### 2.2 Implement Gleaning Loop Pattern
Add iterative refinement inspired by LightRAG:

```python
def extract_with_gleaning(self, text: str, document: Optional[Document] = None, max_iterations: int = 2) -> List[Entity]:
    """Extract entities with gleaning loop for improved recall."""
    all_entities = []

    # Initial extraction
    initial_entities = self._extract_llm_entities(text, document)
    all_entities.extend(initial_entities)

    # Gleaning iterations
    for iteration in range(max_iterations):
        # Find text regions not covered by existing entities
        uncovered_regions = self._find_uncovered_regions(text, all_entities)

        if not uncovered_regions:
            break

        # Extract from uncovered regions with more aggressive prompts
        gleaned_entities = []
        for region_start, region_end in uncovered_regions:
            region_text = text[region_start:region_end]
            if len(region_text.strip()) > 50:  # Only process substantial regions
                region_entities = self._extract_llm_entities_aggressive(region_text, document, region_start)
                gleaned_entities.extend(region_entities)

        if not gleaned_entities:
            break

        all_entities.extend(gleaned_entities)
        logger.debug(f"Gleaning iteration {iteration + 1}: found {len(gleaned_entities)} additional entities")

    # Deduplicate and merge
    return self._deduplicate_entities(all_entities)
```

### Phase 3: Advanced Relationship Extraction (2-3 days)

#### 3.1 Implement Medical Relationship Types
Add domain-specific relationship patterns:

```python
MEDICAL_RELATIONSHIP_PATTERNS = {
    "treats": [
        r'(\w+)\s+(?:treats|therapy for|treatment of)\s+(\w+)',
        r'(\w+)\s+(?:is used to treat|for treating)\s+(\w+)'
    ],
    "causes": [
        r'(\w+)\s+(?:causes|leads to|results in)\s+(\w+)',
        r'(\w+)\s+(?:is caused by|due to)\s+(\w+)'
    ],
    "prevents": [
        r'(\w+)\s+(?:prevents|prevents|reduces risk of)\s+(\w+)'
    ],
    "interacts_with": [
        r'(\w+)\s+(?:interacts with|combined with)\s+(\w+)'
    ]
}

def extract_semantic_relationships(self, entities: List[Entity], document: Document) -> List[Relationship]:
    """Extract relationships using semantic patterns."""
    relationships = []
    text = document.page_content

    # Create entity lookup by position
    entity_by_position = {(e.start_offset, e.end_offset): e for e in entities}

    for rel_type, patterns in MEDICAL_RELATIONSHIP_PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                source_text = match.group(1)
                target_text = match.group(2)

                # Find corresponding entities
                source_entity = self._find_entity_by_text(source_text, entities)
                target_entity = self._find_entity_by_text(target_text, entities)

                if source_entity and target_entity:
                    relationship = Relationship(
                        source_entity_id=source_entity.id,
                        target_entity_id=target_entity.id,
                        relationship_type=rel_type,
                        confidence=0.8,
                        source_document_id=document.id,
                        metadata={
                            "method": "semantic_pattern",
                            "pattern": pattern,
                            "context": text[match.start()-50:match.end()+50]
                        }
                    )
                    relationships.append(relationship)

    return relationships
```

### Phase 4: Entity Disambiguation and Consolidation (3-4 days)

#### 4.1 Implement Cross-Document Entity Merging
Add entity consolidation inspired by LightRAG:

```python
def consolidate_entities_across_documents(self, documents: List[Document]) -> Dict[str, List[Entity]]:
    """Consolidate entities across multiple documents."""
    entity_clusters = {}

    for document in documents:
        doc_entities = self.extract_entities(document)

        for entity in doc_entities:
            # Find similar entities in existing clusters
            cluster_key = self._find_or_create_cluster(entity, entity_clusters)

            if cluster_key not in entity_clusters:
                entity_clusters[cluster_key] = []

            entity_clusters[cluster_key].append(entity)

    # Merge entities in each cluster
    consolidated_entities = {}
    for cluster_key, cluster_entities in entity_clusters.items():
        merged_entity = self._merge_entity_cluster(cluster_entities)
        consolidated_entities[cluster_key] = merged_entity

    return consolidated_entities

def _merge_entity_cluster(self, entities: List[Entity]) -> Entity:
    """Merge multiple entities into a single consolidated entity."""
    if len(entities) == 1:
        return entities[0]

    # Select canonical text (most frequent or longest)
    text_counts = {}
    for entity in entities:
        text_counts[entity.text] = text_counts.get(entity.text, 0) + 1

    canonical_text = max(text_counts.items(), key=lambda x: (x[1], len(x[0])))[0]

    # Select highest confidence
    max_confidence = max(entity.confidence for entity in entities)

    # Merge descriptions
    descriptions = [entity.metadata.get("description", "") for entity in entities if entity.metadata.get("description")]
    merged_description = " | ".join(set(desc for desc in descriptions if desc))

    # Create merged entity
    base_entity = next(e for e in entities if e.text == canonical_text)
    merged_entity = Entity(
        text=canonical_text,
        entity_type=base_entity.entity_type,
        confidence=max_confidence,
        start_offset=base_entity.start_offset,
        end_offset=base_entity.end_offset,
        source_document_id=base_entity.source_document_id,
        metadata={
            "method": "consolidated",
            "source_count": len(entities),
            "descriptions": merged_description,
            "source_documents": [e.source_document_id for e in entities]
        }
    )

    return merged_entity
```

## Implementation Priority

### Immediate (Week 1)
1. ✅ Fix current GraphRAG keyword processing bug (COMPLETED)
2. 🔄 Implement enhanced medical patterns (Phase 1.1)
3. 🔄 Add context-aware extraction (Phase 1.2)

### Short-term (Weeks 2-3)
4. 📋 Real LLM integration for entity extraction (Phase 2.1)
5. 📋 Gleaning loop implementation (Phase 2.2)
6. 📋 Enhanced relationship extraction (Phase 3.1)

### Medium-term (Weeks 4-6)
7. 📋 Cross-document entity consolidation (Phase 4.1)
8. 📋 Community detection for graph structure
9. 📋 Temporal and factual claim extraction

## Success Metrics

### Quantitative Targets
- **Entity Recall**: >80% of relevant medical entities identified
- **Entity Precision**: >85% of extracted entities are accurate
- **Relationship Accuracy**: >75% of relationships are semantically correct
- **RAGAS Scores**: GraphRAG pipeline achieves >60% performance

### Qualitative Improvements
- Entities include rich contextual descriptions
- Relationships capture semantic meaning, not just co-occurrence
- Knowledge graph supports both local and global queries
- Medical domain knowledge is properly represented

## Validation Strategy

### Test Datasets
1. **Medical Literature**: PMC articles with known entity ground truth
2. **Clinical Notes**: De-identified clinical text with expert annotations
3. **Drug Information**: Pharmaceutical documentation with known relationships

### Evaluation Framework
```python
def evaluate_extraction_quality(extracted_entities: List[Entity], ground_truth: List[Entity]) -> Dict[str, float]:
    """Evaluate entity extraction quality against ground truth."""
    return {
        "precision": calculate_precision(extracted_entities, ground_truth),
        "recall": calculate_recall(extracted_entities, ground_truth),
        "f1_score": calculate_f1(extracted_entities, ground_truth),
        "type_accuracy": calculate_type_accuracy(extracted_entities, ground_truth)
    }
```

This improvement plan transforms our "bare-bones" entity extraction into a production-ready system that matches SOTA GraphRAG implementations while maintaining compatibility with our IRIS database and existing pipeline architecture.