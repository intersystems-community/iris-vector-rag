# Chunking System Architecture Diagram

## Overview

This document provides a visual representation of the chunking system architecture, showing service boundaries, data flow, and component interactions.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           RAG Pipeline Layer                                    │
├─────────────────┬─────────────────┬─────────────────┬─────────────────────────┤
│   BasicRAG      │     HyDE        │     CRAG        │    GraphRAG, etc.       │
│   Pipeline      │   Pipeline      │   Pipeline      │      Pipelines          │
└─────────┬───────┴─────────┬───────┴─────────┬───────┴─────────────────────────┘
          │                 │                 │
          │ add_documents() │                 │
          │                 │                 │
          ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        Storage Layer                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    IRISVectorStore                                      │   │
│  │                                                                         │   │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    │   │
│  │  │ chunking_config │    │   auto_chunk    │    │ chunking_service│    │   │
│  │  └─────────────────┘    └─────────────────┘    └─────────────────┘    │   │
│  │                                                                         │   │
│  │  add_documents(docs, auto_chunk=True, chunking_strategy="semantic")    │   │
│  │                                                                         │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │   │
│  │  │                 Chunking Decision Logic                         │   │   │
│  │  │                                                                 │   │   │
│  │  │  if auto_chunk and doc_size > threshold:                       │   │   │
│  │  │      chunks = chunking_service.chunk_document(doc, strategy)    │   │   │
│  │  │  else:                                                          │   │   │
│  │  │      chunks = [doc]                                             │   │   │
│  │  └─────────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      Chunking Service Layer                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                   DocumentChunkingService                               │   │
│  │                                                                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │   │
│  │  │ fixed_size  │  │  semantic   │  │   hybrid    │  │   custom    │   │   │
│  │  │  Strategy   │  │  Strategy   │  │  Strategy   │  │  Strategy   │   │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │   │
│  │                                                                         │   │
│  │  chunk_document(document, strategy) -> List[Document]                  │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      Configuration Layer                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    ConfigurationManager                                 │   │
│  │                                                                         │   │
│  │  storage:                                                               │   │
│  │    chunking:                                                            │   │
│  │      enabled: true                                                      │   │
│  │      default_strategy: "fixed_size"                                     │   │
│  │      auto_chunk_threshold: 1000                                         │   │
│  │      strategies:                                                        │   │
│  │        fixed_size: { chunk_size: 512, overlap: 50 }                    │   │
│  │        semantic: { model: "sentence-transformers/all-MiniLM-L6-v2" }   │   │
│  │        hybrid: { chunk_size: 512, overlap: 50 }                        │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Component Interactions

### 1. Pipeline → Storage Layer
```
Pipeline.load_documents(documents)
    ↓
IRISVectorStore.add_documents(documents, auto_chunk=True)
    ↓
Automatic chunking based on configuration
    ↓
Document storage in IRIS database
```

### 2. Configuration-Driven Behavior
```
ConfigurationManager.get("storage:chunking")
    ↓
IRISVectorStore initialization with chunking config
    ↓
DocumentChunkingService initialization with strategy
    ↓
Runtime chunking decisions based on document size
```

### 3. Chunking Strategy Selection
```
Document Size Check
    ↓
if size > threshold:
    ↓
DocumentChunkingService.chunk_document(doc, strategy)
    ↓
List[Document] chunks
else:
    ↓
Original document unchanged
```

## Service Boundaries

### **Pipeline Layer**
- **Responsibility**: Business logic for RAG techniques
- **Interface**: `load_documents(documents)`, `query(question)`
- **Dependencies**: IRISVectorStore, LLM functions, embedding functions
- **Abstraction**: Chunking-agnostic - delegates all chunking to storage layer

### **Storage Layer (IRISVectorStore)**
- **Responsibility**: Document storage, retrieval, and automatic chunking
- **Interface**: `add_documents(docs, auto_chunk=None, chunking_strategy=None)`
- **Dependencies**: DocumentChunkingService, ConfigurationManager, SchemaManager
- **Abstraction**: Provides unified document storage with optional chunking

### **Chunking Service Layer**
- **Responsibility**: Document chunking algorithms and strategies
- **Interface**: `chunk_document(document, strategy) -> List[Document]`
- **Dependencies**: Sentence transformers (for semantic chunking)
- **Abstraction**: Strategy pattern for different chunking approaches

### **Configuration Layer**
- **Responsibility**: System configuration and settings management
- **Interface**: `get(key, default=None)`
- **Dependencies**: YAML configuration files
- **Abstraction**: Centralized configuration with hierarchical key access

## Data Flow

### Document Ingestion Flow
```
1. Pipeline receives documents
2. Pipeline calls IRISVectorStore.add_documents()
3. IRISVectorStore checks chunking configuration
4. If auto_chunk enabled and document > threshold:
   a. IRISVectorStore calls DocumentChunkingService
   b. DocumentChunkingService applies strategy
   c. Returns chunked documents
5. IRISVectorStore stores documents (chunked or original)
6. Returns document IDs to pipeline
```

### Configuration Loading Flow
```
1. IRISVectorStore initialization
2. ConfigurationManager loads chunking config
3. DocumentChunkingService initialized with strategy
4. Runtime chunking decisions use loaded configuration
```

## Extensibility Points

### **New Chunking Strategies**
- Add strategy to `DocumentChunkingService.strategies`
- Add configuration parameters to `config/default.yaml`
- Strategy automatically available to all pipelines

### **Pipeline-Specific Overrides**
- Configure per-pipeline chunking behavior
- Override chunking strategy per document batch
- Disable chunking for specific pipelines (e.g., ColBERT)

### **Custom Chunking Logic**
- Extend `DocumentChunkingService` with new strategies
- Add custom threshold logic in `IRISVectorStore`
- Implement document-type-specific chunking

## Performance Characteristics

### **Chunking Decision Overhead**
- O(1) configuration lookup
- O(1) document size check
- Minimal overhead for documents below threshold

### **Chunking Processing**
- O(n) where n = document length
- Strategy-dependent complexity
- Configurable chunk size and overlap

### **Storage Efficiency**
- Chunked documents stored with metadata
- Original document relationships preserved
- Efficient retrieval through vector similarity

## Error Handling

### **Configuration Errors**
- Invalid strategy names → fallback to default
- Missing configuration → chunking disabled
- Invalid parameters → use strategy defaults

### **Chunking Errors**
- Chunking service failure → store original document
- Strategy not found → fallback to fixed_size
- Document too small → store without chunking

### **Storage Errors**
- Database connection issues → propagate to pipeline
- Schema validation errors → detailed error messages
- Vector insertion failures → transaction rollback

This architecture provides a clean separation of concerns while maintaining flexibility and extensibility for future enhancements.