# GraphRAG System Architecture Diagram

## Complete System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    IRIS RAG Framework                                              │
│                                 GraphRAG Integration Architecture                                  │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                      USER INTERFACE LAYER                                          │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   create_rag    │    │ setup_pipeline  │    │get_pipeline_    │    │  Direct Factory │         │
│  │  (pipeline_type │    │ (requirements)  │    │    status       │    │     Access      │         │
│  │   = "graphrag") │    │                 │    │                 │    │                 │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│           │                       │                       │                       │                │
└───────────┼───────────────────────┼───────────────────────┼───────────────────────┼────────────────┘
            │                       │                       │                       │
            ▼                       ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 VALIDATED PIPELINE FACTORY                                         │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                            create_pipeline()                                                  │ │
│  │                                                                                               │ │
│  │  1. Load Requirements     2. Validate Prerequisites    3. Auto-Setup (optional)              │ │
│  │     ↓                        ↓                           ↓                                    │ │
│  │  GraphRAGRequirements → PreConditionValidator → SetupOrchestrator                           │ │
│  │                                                                                               │ │
│  │  4. Create Framework Dependencies    5. Create Pipeline Instance                              │ │
│  │     ↓                                  ↓                                                      │ │
│  │  EntityExtractionService → GraphRAGPipeline(connection, config, entity_service)            │ │
│  └─────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
            │                       │                       │                       │
            ▼                       ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    VALIDATION LAYER                                                │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │GraphRAGRequire- │    │ PreCondition    │    │ SetupOrchestra- │    │ Validation      │         │
│  │    ments        │───▶│   Validator     │───▶│      tor        │───▶│   Report        │         │
│  │                 │    │                 │    │                 │    │                 │         │
│  │• Required Tables│    │• Table Checks   │    │• Auto-create    │    │• Overall Valid  │         │
│  │• Required Embeds│    │• Embedding      │    │  Tables         │    │• Table Status   │         │
│  │• Min Row Counts │    │  Validation     │    │• Generate       │    │• Embedding      │         │
│  │• Graph Rules    │    │• Graph Rules    │    │  Embeddings     │    │  Status         │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│                                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 CONFIGURATION LAYER                                                │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │ Configuration   │    │ Pipeline Config │    │ YAML Config     │    │ Requirements    │         │
│  │   Manager       │───▶│    Service      │───▶│    Files        │───▶│   Registry      │         │
│  │                 │    │                 │    │                 │    │                 │         │
│  │• default_config │    │• pipelines.yaml │    │• default_config │    │• "graphrag" →   │         │
│  │• get("pipelines │    │  definitions    │    │  .yaml          │    │  GraphRAGReq    │         │
│  │  :graphrag")    │    │• Validation     │    │• entity_extract │    │• Pipeline       │         │
│  │• Entity Extract │    │• Module Loading │    │  ion config     │    │  Lookup         │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│                                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 GRAPHRAG PIPELINE LAYER                                            │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                GraphRAGPipeline                                              │ │
│  │                                                                                               │ │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    │ │
│  │  │  load_documents │    │      query      │    │   retrieve      │    │      ask        │    │ │
│  │  │                 │    │                 │    │                 │    │                 │    │ │
│  │  │1. Store Docs    │    │1. Find Seed     │    │• Graph Traversal│    │• Full Query     │    │ │
│  │  │2. Extract       │    │   Entities      │    │• Return Docs    │    │• Return Answer  │    │ │
│  │  │   Entities      │    │2. Traverse      │    │• Fallback Vector│    │• Include Sources│    │ │
│  │  │3. Store Graph   │    │   Graph         │    │  Search         │    │                 │    │ │
│  │  │4. Generate      │    │3. Get Documents │    │                 │    │                 │    │ │
│  │  │   Embeddings    │    │4. Generate      │    │                 │    │                 │    │ │
│  │  │                 │    │   Answer        │    │                 │    │                 │    │ │
│  │  └─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘    │ │
│  └─────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
        │                   │                   │                   │                   │
        ▼                   ▼                   ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              ENTITY EXTRACTION SERVICE LAYER                                       │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                           EntityExtractionService                                            │ │
│  │                                                                                               │ │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    │ │
│  │  │ Strategy Router │───▶│ Extractor Pool  │───▶│ Result Merger   │───▶│ Quality Filter  │    │ │
│  │  │                 │    │                 │    │                 │    │                 │    │ │
│  │  │• Config-driven  │    │• NLP Extractor  │    │• Confidence     │    │• Min Confidence │    │ │
│  │  │• Fallback Logic │    │• LLM Extractor  │    │  Weighting      │    │• Type Validation│    │ │
│  │  │• Hybrid Mode    │    │• Pattern Extract│    │• Duplicate      │    │• Canonical      │    │ │
│  │  │• Batch Support  │    │• Domain Rules   │    │  Elimination    │    │  Linking        │    │ │
│  │  └─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘    │ │
│  │                                                                                               │ │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                           │ │
│  │  │ Entity Linker   │    │Relationship     │    │Performance      │                           │ │
│  │  │                 │    │  Extractor      │    │  Monitor        │                           │ │
│  │  │• Similarity     │    │                 │    │                 │                           │ │
│  │  │  Matching       │    │• Dependency     │    │• Timing Metrics │                           │ │
│  │  │• Canonical      │    │  Parsing        │    │• Error Tracking │                           │ │
│  │  │  Resolution     │    │• Co-occurrence  │    │• Resource Usage │                           │ │
│  │  │• Alias Handling │    │• Semantic Rules │    │• Circuit Breaker│                           │ │
│  │  └─────────────────┘    └─────────────────┘    └─────────────────┘                           │ │
│  └─────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   STORAGE LAYER                                                    │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │ SchemaManager   │    │ IRISVectorStore │    │ Knowledge Graph │    │ Connection      │         │
│  │                 │    │                 │    │   Storage       │    │   Manager       │         │
│  │• Table DDL      │───▶│• Document       │    │                 │───▶│                 │         │
│  │• Graph Tables   │    │  Storage        │    │• Entity Storage │    │• Connection     │         │
│  │• Migration      │    │• Vector Search  │    │• Relationship   │    │  Pooling        │         │
│  │• Index Creation │    │• Embedding Mgmt │    │  Storage        │    │• Transaction    │         │
│  │• Validation     │    │• Fallback Logic │    │• Graph Traversal│    │  Management     │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│                                                                                                     │
│                                          │                                                          │
│                                          ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                  IRIS Database                                               │ │
│  │                                                                                               │ │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    │ │
│  │  │RAG.SourceDocs   │    │  RAG.Entities   │    │RAG.EntityRela-  │    │RAG.EntityEmbed- │    │ │
│  │  │                 │    │                 │    │   tionships     │    │    dings        │    │ │
│  │  │• doc_id (PK)    │    │• entity_id (PK) │    │• relation_id(PK)│    │• entity_id (PK) │    │ │
│  │  │• title          │    │• entity_name    │    │• source_entity  │    │• embedding      │    │ │
│  │  │• text_content   │    │• entity_type    │    │• target_entity  │    │• model_name     │    │ │
│  │  │• embedding      │    │• source_doc_id  │    │• relation_type  │    │• created_at     │    │ │
│  │  │• created_at     │    │• confidence     │    │• confidence     │    │                 │    │ │
│  │  │                 │    │• char_start     │    │• source_doc_id  │    │                 │    │ │
│  │  │                 │    │• char_end       │    │• created_at     │    │                 │    │ │
│  │  │                 │    │• embedding      │    │                 │    │                 │    │ │
│  │  │                 │    │• created_at     │    │                 │    │                 │    │ │
│  │  └─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘    │ │
│  │                           │                      │                      │                     │ │
│  │                           │                      │                      │                     │ │
│  │  Indexes:                 │ Indexes:             │ Indexes:             │ Indexes:            │ │
│  │  • idx_docs_created       │ • idx_entities_name  │ • idx_rel_source     │ • idx_embed_entity  │ │
│  │  • idx_docs_title         │ • idx_entities_type  │ • idx_rel_target     │                     │ │
│  │                           │ • idx_entities_doc   │ • idx_rel_type       │                     │ │
│  │                           │ • idx_entities_conf  │ • idx_rel_bidir      │                     │ │
│  └─────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow Architecture

```
                     USER REQUEST: "What treatments are available for diabetes?"
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              1. QUERY PROCESSING FLOW                                              │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                │
                    ┌───────────────────────────┼───────────────────────────┐
                    ▼                           ▼                           ▼
      ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
      │ Parse Query Terms   │    │  Find Seed Entities │    │ Validate Graph      │
      │                     │    │                     │    │   Connectivity      │
      │• "treatments"       │───▶│ Query: RAG.Entities │───▶│                     │
      │• "diabetes"         │    │ WHERE entity_name   │    │ Check: Min entities │
      │• Extract keywords   │    │ LIKE '%diabetes%'   │    │ Check: Min relations│
      └─────────────────────┘    └─────────────────────┘    └─────────────────────┘
                                                │                           │
                                                ▼                           │
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│                                2. GRAPH TRAVERSAL FLOW                                           │   │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘   │
                                                │                                                       │
                    ┌───────────────────────────┼───────────────────────────┐                           │
                    ▼                           ▼                           ▼                           │
      ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐                     │
      │   Seed Entities     │    │  Traverse Graph     │    │ Collect Related     │                     │
      │                     │    │                     │    │    Entities         │                     │
      │• entity_id: "D001" │───▶│ Query: EntityRela-  │───▶│                     │                     │
      │• entity_name:       │    │  tionships WHERE   │    │• Follow "TREATS"    │                     │
      │  "diabetes"         │    │  source = "D001"   │    │• Follow "CAUSED_BY" │                     │
      │• entity_type:       │    │  OR target = "D001"│    │• Max depth: 2       │                     │
      │  "DISEASE"          │    │                     │    │• Max entities: 50   │                     │
      └─────────────────────┘    └─────────────────────┘    └─────────────────────┘                     │
                                                │                           │                           │
                                                ▼                           ▼                           │
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│                              3. DOCUMENT RETRIEVAL FLOW                                          │   │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘   │
                                                │                                                       │
                    ┌───────────────────────────┼───────────────────────────┐                           │
                    ▼                           ▼                           ▼                           │
      ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐                     │
      │ Related Entities    │    │ Get Source Docs     │    │ Rank by Relevance   │                     │
      │                     │    │                     │    │                     │                     │
      │• "insulin"          │───▶│ Query: SourceDocs   │───▶│• Entity confidence  │                     │
      │• "metformin"        │    │ JOIN Entities ON    │    │• Relationship conf  │                     │
      │• "diet"             │    │ source_doc_id       │    │• Graph distance     │                     │
      │• "exercise"         │    │ WHERE entity_id IN  │    │• Document relevance │                     │
      │                     │    │ (related_entities)  │    │                     │                     │
      └─────────────────────┘    └─────────────────────┘    └─────────────────────┘                     │
                                                │                                                       │
                                                ▼                                                       │
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│                               4. FALLBACK MECHANISM                                              │ ◀─┘
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                │
                                   IF graph_results < threshold
                                                │
                                                ▼
                    ┌─────────────────────┐         ┌─────────────────────┐
                    │ Vector Search       │         │ Combine Results     │
                    │   Fallback          │────────▶│                     │
                    │                     │         │• Graph results +    │
                    │• Embedding query    │         │• Vector results     │
                    │• Similarity search  │         │• Deduplicate docs   │
                    │• Top-k documents    │         │• Re-rank by score   │
                    └─────────────────────┘         └─────────────────────┘
                                                                │
                                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              5. ANSWER GENERATION FLOW                                           │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                │
                    ┌───────────────────────────┼───────────────────────────┐
                    ▼                           ▼                           ▼
      ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
      │ Context Assembly    │    │  LLM Generation     │    │ Response Packaging  │
      │                     │    │                     │    │                     │
      │• Top documents      │───▶│ Prompt: "Based on  │───▶│• Query              │
      │• Entity context     │    │ knowledge graph     │    │• Answer             │
      │• Relationship info  │    │ context, answer:    │    │• Sources            │
      │• Source attribution │    │ What treatments..." │    │• Retrieved docs     │
      │                     │    │                     │    │• Execution time     │
      └─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

## Component Interaction Matrix

| Component | SchemaManager | EntityExtraction | KnowledgeGraph | VectorStore | ConfigManager | Validation |
|-----------|---------------|------------------|----------------|-------------|---------------|------------|
| **GraphRAGPipeline** | Creates tables | Extracts entities | Queries graph | Fallback search | Gets config | Validates prereqs |
| **EntityExtractionService** | - | Core function | Stores results | - | Strategy config | Type validation |
| **SchemaManager** | Core function | - | Manages tables | Vector dims | Table config | Schema validation |
| **ValidatedFactory** | Uses for setup | Injects service | - | Creates instance | Uses for config | Core validation |
| **ValidationOrchestrator** | Triggers setup | - | Validates data | Checks embeddings | Uses requirements | Reports status |

## Error Handling & Circuit Breaker Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                ERROR HANDLING ARCHITECTURE                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘

GraphRAG Query Request
         │
         ▼
┌─────────────────┐      SUCCESS     ┌─────────────────┐      SUCCESS     ┌─────────────────┐
│ Find Seed       │─────────────────▶│ Traverse Graph  │─────────────────▶│ Get Documents   │
│ Entities        │                  │                 │                  │                 │
└─────────────────┘                  └─────────────────┘                  └─────────────────┘
         │                                    │                                    │
         │ FAILURE                            │ FAILURE                            │ FAILURE
         ▼                                    ▼                                    ▼
┌─────────────────┐                  ┌─────────────────┐                  ┌─────────────────┐
│ • Log warning   │                  │ • Log warning   │                  │ • Log warning   │
│ • Increment     │                  │ • Increment     │                  │ • Increment     │
│   error count   │                  │   error count   │                  │   error count   │
│ • Check circuit │                  │ • Check circuit │                  │ • Check circuit │
│   breaker       │                  │   breaker       │                  │   breaker       │
└─────────────────┘                  └─────────────────┘                  └─────────────────┘
         │                                    │                                    │
         └────────────────────────────────────┼────────────────────────────────────┘
                                              │
                                              ▼
                                   ┌─────────────────┐
                                   │ Circuit Breaker │
                                   │   Evaluation    │
                                   │                 │
                                   │• Error count    │
                                   │• Failure rate   │
                                   │• Time window    │
                                   └─────────────────┘
                                              │
                        ┌─────────────────────┼─────────────────────┐
                        │                     │                     │
                        ▼                     │                     ▼
              ┌─────────────────┐             │           ┌─────────────────┐
              │ CIRCUIT OPEN    │             │           │ CIRCUIT CLOSED  │
              │                 │             │           │                 │
              │• Skip graph     │             │           │• Continue with  │
              │• Direct vector  │             │           │  graph queries  │
              │  search         │             │           │• Reset counters │
              │• Log circuit    │             │           │                 │
              │  open event     │             │           │                 │
              └─────────────────┘             │           └─────────────────┘
                        │                     │                     │
                        └─────────────────────┼─────────────────────┘
                                              │
                                              ▼
                                   ┌─────────────────┐
                                   │ Vector Search   │
                                   │   Fallback      │
                                   │                 │
                                   │• Always         │
                                   │  available      │
                                   │• Performance    │
                                   │  guarantee      │
                                   └─────────────────┘
```

This architecture ensures GraphRAG integrates seamlessly with the existing IRIS RAG framework while providing robust error handling, performance monitoring, and graceful degradation to vector search when the knowledge graph is unavailable.