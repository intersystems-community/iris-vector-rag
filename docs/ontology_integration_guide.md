# General-Purpose Ontology Integration Guide for IRIS RAG

## Overview

The IRIS RAG framework includes advanced general-purpose ontology support that works with **ANY domain**. This system automatically adapts to different ontology formats and domains without requiring hardcoded domain-specific implementations.

Key capabilities:

- **Universal ontology support** - Works with ANY domain (medical, legal, financial, scientific, etc.)
- **Auto-detection of domains** from ontology content and metadata
- **Multi-format ontology loading** (OWL, RDF, SKOS, TTL, N3, XML)
- **Ontology-aware entity extraction** with semantic enrichment
- **Query expansion** using concept hierarchies and synonyms
- **Dynamic entity mapping** generation from ontology concepts
- **Reasoning-based relationship inference** across any domain

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Configuration](#configuration)
4. [Supported Ontology Formats](#supported-ontology-formats)
5. [Entity Extraction](#entity-extraction)
6. [GraphRAG Integration](#graphrag-integration)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Usage](#advanced-usage)
10. [API Reference](#api-reference)

## Quick Start

### 1. Enable General Ontology Support

Update your configuration file (`iris_rag/config/default_config.yaml`):

```yaml
ontology:
  enabled: true
  type: "general"                    # Single general-purpose type
  auto_detect_domain: true           # Auto-detect domain from ontology
  sources:
    - type: "owl"
      path: "path/to/your/ontology.owl"    # Works with ANY ontology
  reasoning:
    enable_inference: true
    max_inference_depth: 3
```

### 2. Basic Usage Example - Medical Domain

```python
from iris_rag.pipelines.graphrag_merged import GraphRAGPipeline
from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.models import Document

# Initialize with general ontology support
config_manager = ConfigurationManager()
config_manager._config['ontology'] = {
    'enabled': True,
    'type': 'general',
    'auto_detect_domain': True,
    'sources': [
        {
            'type': 'owl',
            'path': 'medical_ontology.owl'  # Your medical ontology
        }
    ]
}

pipeline = GraphRAGPipeline(config_manager=config_manager)

# Load medical documents - system auto-detects medical domain
documents = [Document(
    id="medical_doc",
    page_content="Patient has type 2 diabetes and takes metformin daily.",
    metadata={"source": "clinical_notes"}
)]

pipeline.load_documents("", documents=documents)

# Query with ontology reasoning - works with any domain
result = pipeline.query("What medications treat diabetes?")

print(f"Found {len(result['retrieved_documents'])} documents")
print(f"Auto-detected domain: {result.get('ontology_insights', {}).get('detected_domain')}")
```

### 3. Basic Usage Example - Legal Domain

```python
# Same system, different domain - just change the ontology file
config_manager._config['ontology']['sources'] = [
    {
        'type': 'owl',
        'path': 'legal_ontology.owl'  # Your legal ontology
    }
]

# Load legal documents - system auto-detects legal domain
documents = [Document(
    id="legal_doc", 
    page_content="The contract includes liability clauses and confidentiality agreements.",
    metadata={"source": "contracts"}
)]

pipeline.load_documents("", documents=documents)

# Same query interface, different domain
result = pipeline.query("What are the liability requirements?")
```

### 4. Run the Demo

```bash
cd /path/to/iris-rag
python scripts/demo_ontology_support.py --ontology /path/to/your/ontology.owl --enable-reasoning
```

## Architecture Overview

### Core Components

```
iris_rag/ontology/
├── __init__.py                     # Module exports and plugin registry
├── models.py                      # Core data models (Concept, Relationship, etc.)
├── loader.py                      # Multi-format ontology loading (OWL/RDF/SKOS/TTL/N3/XML)
├── reasoner.py                    # Reasoning engine and query expansion
└── plugins/
    ├── __init__.py               # General plugin functions
    └── general_ontology.py       # Universal ontology plugin
```

### Data Flow

1. **Ontology Loading**: Universal plugin loads any ontology format
2. **Domain Detection**: Auto-detects domain from ontology content/metadata
3. **Entity Mapping**: Dynamically generates entity mappings from concepts
4. **Entity Extraction**: Domain-agnostic extraction with ontology enhancement
5. **Reasoning**: Universal hierarchical navigation and relationship inference
6. **Query Enhancement**: Expansion using synonyms and concept hierarchies
7. **Graph Integration**: Enriched entities enhance GraphRAG traversal

## Configuration

### Complete Configuration Example

```yaml
# Entity extraction with general ontology support
entity_extraction:
  enabled: true
  method: "ontology_hybrid"           # ontology_hybrid, llm_basic, pattern_only
  confidence_threshold: 0.7
  entity_types: []                    # Auto-populated from ontology

# General ontology configuration
ontology:
  enabled: true
  type: "general"                     # Universal type
  auto_detect_domain: true            # Auto-detect from ontology
  
  # Ontology data sources - works with ANY domain
  sources:
    - type: "owl"
      path: "ontologies/medical.owl"           # Medical domain example
    - type: "rdf"  
      path: "ontologies/legal.rdf"             # Legal domain example
    - type: "skos"
      path: "ontologies/financial.skos"       # Financial domain example
    - type: "ttl"
      path: "ontologies/scientific.ttl"       # Scientific domain example
  
  # Custom domain definitions (optional)
  custom_domains:
    enabled: false
    definition_path: null
  
  # Reasoning configuration
  reasoning:
    enable_inference: true
    max_inference_depth: 3
    confidence_threshold: 0.6
    reasoning_strategies:
      - "subsumption"                # Hierarchical reasoning
      - "property_based"             # Property-based inference
      - "rule_based"                 # Custom rule application
  
  # Entity mapping (auto-generated from ontology)
  entity_mapping:
    auto_generate: true
    auto_domain_detection: true
    expansion_strategy: "synonyms"    # synonyms, hierarchical, semantic
  
  # Performance settings
  performance:
    max_concepts: 10000
    cache_queries: true
    lazy_load: true

# GraphRAG with general ontology integration
pipelines:
  graphrag:
    ontology_integration:
      query_expansion: true
      entity_enrichment: true
      relationship_inference: true
      semantic_similarity: true
```

## Supported Ontology Formats

The general-purpose system supports all standard ontology formats:

### OWL (Web Ontology Language)
```yaml
sources:
  - type: "owl"
    path: "medical_ontology.owl"
    # Auto-detects: medical, biomedical, clinical domains
```

### RDF (Resource Description Framework)
```yaml
sources:
  - type: "rdf"
    path: "legal_concepts.rdf"
    # Auto-detects: legal, regulatory, compliance domains
```

### SKOS (Simple Knowledge Organization System)
```yaml
sources:
  - type: "skos"
    path: "financial_terms.skos"
    # Auto-detects: financial, banking, economic domains
```

### TTL (Turtle)
```yaml
sources:
  - type: "ttl"
    path: "scientific_vocabulary.ttl"
    # Auto-detects: scientific, research, academic domains
```

### N3 (Notation3)
```yaml
sources:
  - type: "n3"
    path: "technology_concepts.n3"
    # Auto-detects: technology, engineering domains
```

### XML (with ontology structure)
```yaml
sources:
  - type: "xml"
    path: "business_processes.xml"
    # Auto-detects: business, process, organizational domains
```

## Entity Extraction

### General Ontology-Aware Extraction

The system uses a universal extraction approach that adapts to any domain:

```python
from iris_rag.services.entity_extraction import OntologyAwareEntityExtractor

# Initialize with general ontology (works with any domain)
extractor = OntologyAwareEntityExtractor(config_manager=config_manager)

# Medical text example
medical_text = "Patient diagnosed with hypertension, prescribed ACE inhibitor medication"
medical_entities = extractor.extract_with_ontology(medical_text)

# Legal text example  
legal_text = "Contract breach results in liability for damages and penalties"
legal_entities = extractor.extract_with_ontology(legal_text)

# Financial text example
financial_text = "Investment portfolio shows capital gains and dividend income"
financial_entities = extractor.extract_with_ontology(financial_text)

# System auto-adapts to each domain
for entities, domain_type in [(medical_entities, "medical"), 
                             (legal_entities, "legal"), 
                             (financial_entities, "financial")]:
    print(f"\n{domain_type.upper()} ENTITIES:")
    for entity in entities:
        print(f"  - {entity.text} ({entity.entity_type})")
        print(f"    Auto-detected domain: {entity.metadata.get('auto_detected_domain')}")
        print(f"    Confidence: {entity.confidence:.2f}")
```

### Domain Auto-Detection

The system automatically detects domains from ontology content:

```python
from iris_rag.ontology.plugins import GeneralOntologyPlugin

plugin = GeneralOntologyPlugin()

# Load any ontology - system auto-detects domain
plugin.load_ontology_from_file("your_ontology.owl")

print(f"Auto-detected domain: {plugin.domain}")
print(f"Generated entity mappings: {plugin.entity_mappings}")
print(f"Loaded concepts: {len(plugin.hierarchy.concepts)}")
```

## GraphRAG Integration

### Universal Query Processing

The GraphRAG pipeline works with any ontology domain:

```python
# Works with medical ontology
result = pipeline.query("What are the treatment options for cardiovascular disease?")

# Works with legal ontology  
result = pipeline.query("What are the requirements for contract termination?")

# Works with financial ontology
result = pipeline.query("What are the risk factors for investment portfolios?")

# All queries use the same interface, system adapts automatically
print(f"Retrieved documents: {len(result['retrieved_documents'])}")

# Get ontology insights (domain-agnostic)
insights = result.get('ontology_insights', {})
print(f"Detected domain: {insights.get('detected_domain')}")
print(f"Inferred concepts: {[c['label'] for c in insights.get('inferred_concepts', [])]}")
```

### Document Loading with Any Domain

```python
# Mix documents from different domains
documents = [
    Document(id="med_1", page_content="Patient treatment protocol...", 
             metadata={"domain": "medical"}),
    Document(id="legal_1", page_content="Contract compliance requirements...", 
             metadata={"domain": "legal"}),
    Document(id="finance_1", page_content="Investment analysis report...", 
             metadata={"domain": "financial"})
]

# System handles all domains automatically
pipeline.load_documents("", documents=documents, generate_embeddings=True)
```

## Performance Optimization

### Configuration for Large Ontologies

```yaml
ontology:
  performance:
    max_concepts: 5000              # Limit total concepts
    cache_queries: true             # Cache frequent queries
    lazy_load: true                 # Load concepts on-demand
    similarity_threshold: 0.8       # Higher threshold = fewer matches
```

### Memory Management

```python
# For large ontologies, consider chunked loading
config_manager._config['ontology']['performance']['lazy_load'] = True

# Monitor memory usage
import psutil
process = psutil.Process()
memory_before = process.memory_info().rss / 1024 / 1024  # MB

# Load ontology
pipeline = GraphRAGPipeline(config_manager=config_manager)

memory_after = process.memory_info().rss / 1024 / 1024  # MB
print(f"Memory used: {memory_after - memory_before:.1f} MB")
```

## Troubleshooting

### Common Issues

#### 1. Ontology File Not Loading

**Problem**: Ontology file cannot be parsed

**Solution**:
```python
from iris_rag.ontology.plugins import GeneralOntologyPlugin

plugin = GeneralOntologyPlugin()

# Check file format detection
file_format = plugin._detect_format("your_ontology.owl")
print(f"Detected format: {file_format}")

# Check supported formats
supported = plugin.get_supported_formats()
print(f"Supported formats: {supported}")

# Validate file exists and is readable
import os
if os.path.exists("your_ontology.owl"):
    print("File exists")
    with open("your_ontology.owl", 'r') as f:
        print(f"First 200 chars: {f.read(200)}")
```

#### 2. Domain Not Auto-Detected

**Problem**: System cannot determine ontology domain

**Solution**:
```python
# Manually specify domain if auto-detection fails
custom_domain = {
    "name": "custom_domain",
    "description": "Custom domain description",
    "entity_types": ["CustomEntity", "CustomConcept"],
    "concepts": [
        {"id": "concept1", "label": "Concept 1", "type": "CustomEntity"}
    ]
}

plugin.load_custom_domain(custom_domain)
print(f"Manual domain set: {plugin.domain}")
```

#### 3. Poor Performance with Large Ontologies

**Problem**: System is slow with large ontology files

**Solution**:
```python
# Optimize for large ontologies
config_manager._config['ontology'].update({
    'performance': {
        'max_concepts': 2000,           # Reduce concept limit
        'cache_queries': True,          # Enable caching
        'lazy_load': True,             # Load on-demand
        'similarity_threshold': 0.9    # Higher threshold
    }
})

# Use concept filtering
plugin = GeneralOntologyPlugin(domain_config={
    'concept_filter': lambda concept: len(concept.label) < 50  # Filter long labels
})
```

## Advanced Usage

### Loading Multiple Ontologies

```python
# Load and merge multiple domain ontologies
config_manager._config['ontology']['sources'] = [
    {'type': 'owl', 'path': 'medical.owl'},
    {'type': 'rdf', 'path': 'legal.rdf'},
    {'type': 'skos', 'path': 'financial.skos'}
]

pipeline = GraphRAGPipeline(config_manager=config_manager)

# System automatically handles multi-domain content
mixed_text = "Medical malpractice insurance for healthcare investment portfolios"
entities = pipeline._extract_entities(mixed_text)

# Results span multiple domains
for entity in entities:
    print(f"{entity.text}: {entity.metadata.get('auto_detected_domain')}")
```

### Custom Domain Definition

```python
from iris_rag.ontology.plugins import GeneralOntologyPlugin

plugin = GeneralOntologyPlugin()

# Define custom domain for specialized use case
aerospace_domain = {
    "name": "aerospace",
    "description": "Aerospace engineering domain",
    "entity_types": ["Aircraft", "Engine", "System", "Component"],
    "concepts": [
        {
            "id": "turbofan",
            "label": "Turbofan Engine",
            "synonyms": ["jet engine", "turbofan"],
            "type": "Engine"
        },
        {
            "id": "avionics",
            "label": "Avionics System", 
            "synonyms": ["flight systems", "navigation"],
            "type": "System"
        }
    ],
    "relationships": [
        {
            "source": "turbofan",
            "target": "aircraft",
            "type": "part_of"
        }
    ]
}

plugin.load_custom_domain(aerospace_domain)
print(f"Custom domain loaded: {plugin.domain}")
```

### Real-time Ontology Updates

```python
# Monitor ontology file for changes
import watchdog.observers
import watchdog.events

class OntologyFileHandler(watchdog.events.FileSystemEventHandler):
    def __init__(self, pipeline):
        self.pipeline = pipeline
    
    def on_modified(self, event):
        if event.src_path.endswith('.owl'):
            print(f"Ontology file updated: {event.src_path}")
            # Reload ontology
            self.pipeline._init_ontology_support()

# Watch ontology directory
observer = watchdog.observers.Observer()
observer.schedule(OntologyFileHandler(pipeline), "ontologies/", recursive=True)
observer.start()
```

## API Reference

### Core Classes

#### `GeneralOntologyPlugin`

```python
class GeneralOntologyPlugin:
    def __init__(self, domain_config=None):
        """Initialize general-purpose ontology plugin."""
    
    def auto_detect_domain(self, ontology_data: Dict) -> str:
        """Auto-detect domain from ontology metadata."""
    
    def auto_generate_mappings(self, concepts: List[Concept]) -> Dict:
        """Automatically generate entity mappings from concepts."""
    
    def load_ontology_from_file(self, file_path: str):
        """Load ontology from any supported file format."""
    
    def load_custom_domain(self, domain_definition: Dict):
        """Load custom domain definition."""
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported ontology formats."""
```

#### `OntologyAwareEntityExtractor`

```python
class OntologyAwareEntityExtractor:
    def __init__(self, config_manager: ConfigurationManager):
        """Initialize with general ontology support."""
    
    def extract_with_ontology(self, text: str, document: Document = None) -> List[Entity]:
        """Extract entities using general ontology."""
```

### Plugin Functions

```python
# General plugin functions
def get_ontology_plugin() -> GeneralOntologyPlugin
def create_plugin_from_config(config: Dict) -> GeneralOntologyPlugin
def load_custom_domain_definition(definition_path: str) -> Dict
```

## Best Practices

### 1. Configuration Management
- Start with basic ontology configuration
- Enable auto-detection for unknown domains
- Use performance settings for large ontologies
- Cache frequently accessed concepts

### 2. Performance Optimization
- Monitor memory usage with large ontologies
- Use lazy loading for better startup times
- Set appropriate similarity thresholds
- Consider domain-specific optimizations

### 3. Quality Assurance
- Validate ontology files before deployment
- Test entity extraction across different text types
- Monitor auto-detection accuracy
- Review and tune confidence thresholds

### 4. Maintenance
- Keep ontology files updated with domain standards
- Monitor performance metrics
- Update entity mappings as domains evolve
- Document custom domain definitions

## Domain-Specific Examples

### Example 1: Scientific Research

```python
# Configure for scientific ontologies
config_manager._config['ontology']['sources'] = [
    {'type': 'owl', 'path': 'gene_ontology.owl'},
    {'type': 'rdf', 'path': 'chemical_entities.rdf'}
]

# Process scientific papers
documents = [Document(
    id="paper_1",
    page_content="The BRCA1 gene mutation increases breast cancer susceptibility...",
    metadata={"domain": "molecular_biology"}
)]

pipeline.load_documents("", documents=documents)
result = pipeline.query("What genes are associated with cancer risk?")
```

### Example 2: Business Processes

```python
# Configure for business domain
config_manager._config['ontology']['sources'] = [
    {'type': 'skos', 'path': 'business_processes.skos'},
    {'type': 'xml', 'path': 'organizational_structure.xml'}
]

# Process business documents  
documents = [Document(
    id="process_1",
    page_content="The procurement workflow requires approval from finance and legal departments...",
    metadata={"domain": "business_process"}
)]

pipeline.load_documents("", documents=documents)
result = pipeline.query("What are the approval requirements for procurement?")
```

This comprehensive guide demonstrates how the general-purpose ontology system works with ANY domain. The system automatically adapts to your specific ontology files and domain requirements without needing hardcoded domain-specific implementations.