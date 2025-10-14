# GraphRAG Entity Extraction Integration Analysis

## Executive Summary

Analysis reveals the merged GraphRAG implementation needs to integrate with existing **spaCy/NumPy NER pipelines** and **local LLM-based entity extraction systems** already present in the codebase. The current dual strategy (service + local) should be enhanced to leverage these proven extraction mechanisms.

## Current Entity Extraction Infrastructure

### 1. SpaCy/NumPy Biomedical NER Pipeline

The codebase has sophisticated spaCy integration with biomedical models:

**Primary Models (in order of preference):**
```python
# From scripts/test_biomedical_entity_extraction.py
try:
    self.nlp = spacy.load("en_core_sci_sm")  # ScispaCy biomedical
    logger.info("Loaded scispacy model: en_core_sci_sm")
except:
    try:
        self.nlp = spacy.load("en_ner_bc5cdr_md")  # BioBERT
        logger.info("Loaded biomedical model: en_ner_bc5cdr_md")
    except:
        self.nlp = spacy.load("en_core_web_sm")  # General fallback
```

**Key Features:**
- **Biomedical Specialization**: ScispaCy models trained on scientific literature
- **Entity Type Support**: DISEASE, CHEMICAL, DRUG, PROCEDURE entities
- **Performance**: Optimized for large-scale document processing
- **NumPy Integration**: Uses NumPy arrays for embeddings and vector operations

### 2. Knowledge Extractor Integration

The `iris_rag/memory/knowledge_extractor.py` provides configurable extraction:

```python
def _extract_entities_spacy(self, content: str, confidence_threshold: float) -> List[Entity]:
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
                    properties={"start": ent.start_char, "end": ent.end_char}
                )
                entities.append(entity)
```

### 3. Local LLM Entity Extraction System

The codebase includes LLM-based extraction capabilities:

**Service Architecture:**
```python
# From docs/architecture/graphrag_service_interfaces.md
class LLMEntityExtractor(IEntityExtractor):
    """LLM-based entity extraction using structured prompts."""
    
    def __init__(self, llm_func: Callable[[str], str], config: Dict[str, Any]):
        self.llm_func = llm_func
        self.prompt_template = config.get('prompt_template', self._default_prompt())
        self.max_retries = config.get('max_retries', 3)
        self.rate_limit_delay = config.get('rate_limit_delay', 1.0)
```

**Configuration Support:**
```yaml
# From docs/architecture/graphrag_framework_integration.md
entity_extraction:
  default_strategy: "hybrid"  # nlp, llm, pattern, hybrid
  
  extractors:
    nlp:
      enabled: true
      model: "en_core_sci_sm"
      confidence_threshold: 0.7
    
    llm:
      enabled: false  # Enable if LLM is available
      model: "gpt-4"
      max_retries: 3
      batch_size: 3
```

## Integration Strategy for Merged GraphRAG

### 1. Enhanced Local Extraction Method

The merged implementation's `_extract_locally()` method should integrate with existing infrastructure:

```python
def _extract_locally(self, documents: List[Document]) -> Tuple[int, int]:
    """Extract entities using local extraction methods with spaCy/LLM integration."""
    total_entities = 0
    total_relationships = 0

    # Initialize spaCy pipeline if available
    nlp_processor = self._initialize_spacy_pipeline()
    
    # Check for LLM availability
    llm_available = self.llm_func is not None
    
    for doc in documents:
        try:
            # Primary: Use spaCy biomedical NER
            if nlp_processor:
                entities = self._extract_entities_spacy(doc, nlp_processor)
            # Fallback: Use LLM if available
            elif llm_available:
                entities = self._extract_entities_llm(doc)
            # Last resort: Pattern matching
            else:
                entities = self._extract_entities_pattern(doc)
            
            relationships = self._extract_relationships(doc, entities)
            
            self._store_entities(doc.id, entities)
            self._store_relationships(doc.id, relationships)
            
            total_entities += len(entities)
            total_relationships += len(relationships)
            
        except Exception as e:
            logger.error(f"Local entity extraction failed for document {doc.id}: {e}")
            raise EntityExtractionFailedException(f"Local extraction failed for document {doc.id}: {e}")
    
    return total_entities, total_relationships
```

### 2. SpaCy Pipeline Integration

```python
def _initialize_spacy_pipeline(self):
    """Initialize spaCy pipeline with biomedical models."""
    try:
        # Prefer biomedical models
        models_to_try = [
            "en_core_sci_sm",      # ScispaCy biomedical
            "en_ner_bc5cdr_md",    # BioBERT
            "en_core_web_sm"       # General fallback
        ]
        
        for model_name in models_to_try:
            try:
                import spacy
                nlp = spacy.load(model_name)
                logger.info(f"Loaded spaCy model: {model_name}")
                return nlp
            except OSError:
                continue
        
        logger.warning("No spaCy model available")
        return None
        
    except Exception as e:
        logger.warning(f"Error initializing spaCy pipeline: {e}")
        return None

def _extract_entities_spacy(self, document: Document, nlp_processor) -> List[Dict[str, Any]]:
    """Extract entities using spaCy NER with biomedical optimization."""
    entities = []
    text = document.page_content
    
    try:
        # Process text with spaCy
        doc = nlp_processor(text[:1000000])  # Limit for memory
        
        for ent in doc.ents:
            # Enhanced confidence scoring for biomedical entities
            confidence = self._calculate_biomedical_confidence(ent)
            
            if confidence >= 0.7:  # Configurable threshold
                entity = {
                    "entity_id": f"{document.id}_entity_{len(entities)}",
                    "entity_text": ent.text,
                    "entity_type": self._map_spacy_to_standard_type(ent.label_),
                    "position": ent.start_char,
                    "confidence": confidence,
                    "embedding": self._get_entity_embedding(ent.text)
                }
                entities.append(entity)
    
    except Exception as e:
        logger.warning(f"spaCy extraction failed: {e}")
        return []
    
    return entities[:self.max_entities]
```

### 3. LLM Integration Enhancement

```python
def _extract_entities_llm(self, document: Document) -> List[Dict[str, Any]]:
    """Extract entities using local LLM with biomedical prompts."""
    if not self.llm_func:
        return []
    
    # Use biomedical-specific prompt
    prompt = self._build_biomedical_prompt(document.page_content)
    
    try:
        response = self.llm_func(prompt)
        entities = self._parse_llm_entities(response, document)
        
        # Validate and score entities
        validated_entities = []
        for entity in entities:
            if self._validate_biomedical_entity(entity):
                validated_entities.append(entity)
        
        return validated_entities[:self.max_entities]
        
    except Exception as e:
        logger.error(f"LLM entity extraction failed: {e}")
        return []

def _build_biomedical_prompt(self, text: str) -> str:
    """Build biomedical-specific extraction prompt."""
    return f"""
    Extract biomedical entities from the following text. Focus on:
    - Diseases and medical conditions
    - Drugs and treatments  
    - Genes and proteins
    - Medical procedures
    - Anatomical structures
    
    Return entities in JSON format with: name, type, confidence, start_char, end_char
    
    Entity types: DISEASE, DRUG, TREATMENT, GENE, PROTEIN, ANATOMY, PROCEDURE
    
    Text: {text[:2000]}
    
    Entities:
    """
```

## Configuration Integration

### Enhanced Pipeline Configuration

```python
# In merged GraphRAG __init__
self.extraction_config = self.pipeline_config.get("entity_extraction", {})
self.extraction_strategy = self.extraction_config.get("strategy", "spacy_primary")

# Strategy options:
# - "spacy_primary": SpaCy with LLM fallback
# - "llm_primary": LLM with spaCy fallback  
# - "hybrid": Both methods with confidence weighting
# - "service_only": Use EntityExtractionService only
```

### Model Loading Configuration

```yaml
# Pipeline configuration
pipelines:
  graphrag:
    entity_extraction:
      strategy: "spacy_primary"
      spacy:
        models: ["en_core_sci_sm", "en_ner_bc5cdr_md", "en_core_web_sm"]
        confidence_threshold: 0.7
        max_entities: 50
      llm:
        enabled: true
        prompt_type: "biomedical"
        max_retries: 3
        batch_size: 5
      performance:
        cache_embeddings: true
        parallel_processing: true
```

## Performance Optimization

### 1. Caching Strategy
- **Model Caching**: Cache loaded spaCy models across documents
- **Embedding Caching**: Store entity embeddings to avoid recomputation
- **Result Caching**: Cache extraction results for duplicate text

### 2. Batch Processing
- **Document Batching**: Process multiple documents in spaCy pipeline
- **Entity Batching**: Batch entity storage operations
- **Embedding Batching**: Generate embeddings in batches

### 3. Memory Management
- **Text Chunking**: Process large documents in chunks
- **Model Sharing**: Share spaCy models across extraction instances
- **Resource Cleanup**: Proper cleanup of spaCy resources

## Integration Benefits

### 1. Proven Performance
- **spaCy Integration**: Leverages existing optimized biomedical NER
- **LLM Flexibility**: Uses local LLM capabilities when available
- **Fallback Robustness**: Multiple extraction strategies ensure reliability

### 2. Domain Specialization
- **Biomedical Focus**: Optimized for medical/scientific content
- **Entity Type Mapping**: Consistent mapping to standard ontologies
- **Confidence Scoring**: Domain-aware confidence calculation

### 3. Operational Excellence
- **Resource Efficiency**: Optimal use of computational resources
- **Error Handling**: Graceful degradation across extraction methods
- **Monitoring**: Performance tracking for all extraction strategies

## Recommended Implementation

### Phase 1: SpaCy Integration
1. Enhance `_extract_locally()` with spaCy pipeline
2. Implement biomedical model loading hierarchy
3. Add entity type mapping and confidence scoring

### Phase 2: LLM Integration  
1. Add LLM fallback extraction method
2. Implement biomedical prompt templates
3. Create entity validation logic

### Phase 3: Hybrid Strategy
1. Implement confidence-weighted hybrid extraction
2. Add performance monitoring and comparison
3. Optimize batch processing and caching

This integration strategy leverages the existing proven infrastructure while maintaining the production-quality characteristics of the merged GraphRAG implementation.