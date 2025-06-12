# Self-Healing Data State Management Architecture

## Overview

This document defines the architecture for a declarative, `make`-driven self-healing data state management system that ensures datasets reach desired states (e.g., "1000 documents ready for RAGAS evaluation") using the `SetupOrchestrator`.

## System Architecture

### 1. Service Boundaries

```
┌─────────────────────────────────────────────────────────────────┐
│                    Make Target Layer                            │
├─────────────────────────────────────────────────────────────────┤
│  make ensure-ragas-1000-ready                                   │
│  make populate-token-embeddings-1000                            │
│  make validate-dataset-state                                    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              Dataset State Management Script                    │
├─────────────────────────────────────────────────────────────────┤
│  scripts/ensure_dataset_state.py                               │
│  - Target state parsing                                        │
│  - State validation                                            │
│  - Orchestrator coordination                                   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              Enhanced SetupOrchestrator                        │
├─────────────────────────────────────────────────────────────────┤
│  iris_rag/validation/orchestrator.py                          │
│  - Granular healing methods                                    │
│  - Idempotent operations                                       │
│  - Progress tracking                                           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Data State Validators                          │
├─────────────────────────────────────────────────────────────────┤
│  - Document count validation                                   │
│  - Token embedding completeness                               │
│  - Schema integrity checks                                    │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Component Design

#### 2.1 Make Target Layer

**Parameterized Targets with Document Count:**

**Primary Target:** `make ensure-ragas-ready DOCS=1000`
- **Command:** `python scripts/ensure_dataset_state.py --target-state ragas-eval --doc-count $(DOCS) --auto-fix`
- **Default:** `DOCS=1000` if not specified

**Specialized Targets:**
- `make populate-token-embeddings DOCS=1000`: Focus specifically on token embeddings
- `make validate-dataset-state DOCS=1000`: Validation only, no healing
- `make ensure-ragas-ready DOCS=5000`: Scale up to 5000 documents
- `make ensure-ragas-ready DOCS=10000`: Scale up to 10000 documents

**Makefile Implementation:**
```makefile
# Default document count
DOCS ?= 1000

ensure-ragas-ready:
	python scripts/ensure_dataset_state.py --target-state ragas-eval --doc-count $(DOCS) --auto-fix

populate-token-embeddings:
	python scripts/ensure_dataset_state.py --target-state token-embeddings --doc-count $(DOCS) --auto-fix

validate-dataset-state:
	python scripts/ensure_dataset_state.py --target-state ragas-eval --doc-count $(DOCS) --validate-only

# Convenience targets for common scales
ensure-ragas-1k: DOCS=1000
ensure-ragas-1k: ensure-ragas-ready

ensure-ragas-5k: DOCS=5000
ensure-ragas-5k: ensure-ragas-ready

ensure-ragas-10k: DOCS=10000
ensure-ragas-10k: ensure-ragas-ready
```

#### 2.2 Dataset State Management Script

**File:** [`scripts/ensure_dataset_state.py`](scripts/ensure_dataset_state.py)

**Responsibilities:**
- Parse target state specifications
- Initialize configuration and connection managers
- Coordinate with `SetupOrchestrator` for healing operations
- Provide detailed progress reporting
- Support dry-run mode for validation

**Interface:**
```python
class DatasetStateManager:
    def __init__(self, config_manager, connection_manager, setup_orchestrator)
    def ensure_target_state(self, target_state: str, doc_count: int = 1000, auto_fix: bool = False) -> StateReport
    def validate_current_state(self, target_state: str, doc_count: int = 1000) -> StateValidation
    def get_healing_plan(self, target_state: str, doc_count: int = 1000) -> HealingPlan
```

**Command Line Interface:**
```bash
# Basic usage with default 1000 documents
python scripts/ensure_dataset_state.py --target-state ragas-eval --auto-fix

# Scale to different document counts
python scripts/ensure_dataset_state.py --target-state ragas-eval --doc-count 5000 --auto-fix
python scripts/ensure_dataset_state.py --target-state ragas-eval --doc-count 10000 --auto-fix

# Validation only (no healing)
python scripts/ensure_dataset_state.py --target-state ragas-eval --doc-count 1000 --validate-only

# Dry run mode
python scripts/ensure_dataset_state.py --target-state ragas-eval --doc-count 1000 --dry-run
```

#### 2.3 Enhanced SetupOrchestrator

**Current Issue Analysis:**
The existing [`_generate_token_embeddings()`](iris_rag/validation/orchestrator.py:490) method has an all-or-nothing check:

```python
# Current problematic logic
cursor.execute("SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings")
existing_count = cursor.fetchone()[0]
if existing_count > 0:
    self.logger.info(f"Token embeddings already exist ({existing_count} tokens)")
    return
```

**Proposed Enhancement:**
Replace with granular healing methods that identify and process only missing embeddings.

### 3. Target State Specifications

#### 3.1 RAGAS-Eval State (Parameterized)

**Requirements:**
- Exactly `doc_count` documents in [`RAG.SourceDocuments`](RAG.SourceDocuments)
- All documents have valid abstracts
- All documents have token embeddings in [`RAG.DocumentTokenEmbeddings`](RAG.DocumentTokenEmbeddings)
- Token embeddings use correct dimensionality (768)
- Schema integrity validated

#### 3.2 State Validation Logic

```python
def validate_ragas_state(doc_count: int = 1000) -> StateValidation:
    checks = {
        "document_count": validate_document_count(target_count=doc_count),
        "token_embeddings_complete": validate_token_embeddings_completeness(doc_count=doc_count),
        "embedding_dimensions": validate_embedding_dimensions(expected_dim=768),
        "schema_integrity": validate_schema_integrity()
    }
    return StateValidation(checks)

def validate_document_count(target_count: int) -> ValidationCheck:
    """Validate that we have at least target_count documents."""
    query = "SELECT COUNT(*) FROM RAG.SourceDocuments WHERE abstract IS NOT NULL"
    actual_count = execute_query(query)[0][0]
    
    return ValidationCheck(
        name="document_count",
        passed=actual_count >= target_count,
        message=f"Found {actual_count} documents, target: {target_count}",
        details={"actual": actual_count, "target": target_count}
    )

def validate_token_embeddings_completeness(doc_count: int) -> ValidationCheck:
    """Validate token embeddings for top doc_count documents."""
    query = f"""
    SELECT COUNT(DISTINCT sd.doc_id) as docs_with_embeddings,
           COUNT(DISTINCT top_docs.doc_id) as total_target_docs
    FROM (
        SELECT TOP {doc_count} doc_id
        FROM RAG.SourceDocuments
        WHERE abstract IS NOT NULL
        ORDER BY doc_id
    ) top_docs
    LEFT JOIN RAG.DocumentTokenEmbeddings dte ON top_docs.doc_id = dte.doc_id
    """
    result = execute_query(query)[0]
    docs_with_embeddings, total_target_docs = result
    
    return ValidationCheck(
        name="token_embeddings_complete",
        passed=docs_with_embeddings == total_target_docs,
        message=f"Token embeddings: {docs_with_embeddings}/{total_target_docs} documents",
        details={
            "docs_with_embeddings": docs_with_embeddings,
            "total_target_docs": total_target_docs,
            "missing_count": total_target_docs - docs_with_embeddings
        }
    )
```

## 4. Idempotent Healing Architecture

### 4.1 Enhanced SetupOrchestrator Methods

**New Public Method:**
```python
def heal_token_embeddings(self, target_doc_count: int = 1000, force_regenerate: bool = False) -> HealingResult
```

**Granular Healing Logic:**
```python
def _identify_missing_token_embeddings(self, target_doc_count: int = 1000) -> List[str]:
    """Identify documents missing token embeddings within target count."""
    query = f"""
    SELECT top_docs.doc_id
    FROM (
        SELECT TOP {target_doc_count} doc_id
        FROM RAG.SourceDocuments
        WHERE abstract IS NOT NULL
        ORDER BY doc_id
    ) top_docs
    LEFT JOIN RAG.DocumentTokenEmbeddings dte ON top_docs.doc_id = dte.doc_id
    WHERE dte.doc_id IS NULL
    """
    
    return [row[0] for row in cursor.fetchall()]

def _generate_token_embeddings_for_documents(self, doc_ids: List[str]) -> HealingResult:
    """Generate token embeddings only for specified documents."""
    # Efficient batch processing for missing embeddings only
    
def get_target_document_set(self, doc_count: int = 1000) -> List[str]:
    """Get the target set of doc_ids for the specified count."""
    query = f"""
    SELECT TOP {doc_count} doc_id
    FROM RAG.SourceDocuments
    WHERE abstract IS NOT NULL
    ORDER BY doc_id
    """
    return [row[0] for row in cursor.fetchall()]
```

### 4.2 Healing Strategy

**Progressive Healing:**
1. **Validate Current State** - Identify what's missing
2. **Generate Healing Plan** - Determine required operations
3. **Execute Healing** - Process only missing components
4. **Verify Completion** - Confirm target state achieved

**Efficiency Optimizations:**
- Process only documents missing token embeddings
- Batch processing for performance
- Progress tracking and resumability
- Rollback capability for failed operations

## 5. Implementation Plan

### 5.1 Phase 1: Enhanced SetupOrchestrator (Priority: High)

**Deliverables:**
- Add [`heal_token_embeddings()`](iris_rag/validation/orchestrator.py) public method
- Implement [`_identify_missing_token_embeddings()`](iris_rag/validation/orchestrator.py) 
- Replace all-or-nothing logic with granular healing
- Add progress tracking and logging

### 5.2 Phase 2: Dataset State Management Script (Priority: High)

**Deliverables:**
- Create [`scripts/ensure_dataset_state.py`](scripts/ensure_dataset_state.py)
- Implement target state parsing
- Add state validation logic
- Integrate with enhanced `SetupOrchestrator`

### 5.3 Phase 3: Make Target Integration (Priority: Medium)

**Deliverables:**
- Add `make ensure-ragas-1000-ready` target
- Add `make populate-token-embeddings-1000` target
- Add `make validate-dataset-state` target
- Update documentation

### 5.4 Phase 4: Advanced Features (Priority: Low)

**Deliverables:**
- Dry-run mode support
- Healing plan visualization
- Performance metrics collection
- Rollback capabilities

## 6. API Design

### 6.1 SetupOrchestrator Enhancement

```python
class SetupOrchestrator:
    # New public methods for granular healing with parameterized document counts
    def heal_token_embeddings(self, target_doc_count: int = 1000,
                            force_regenerate: bool = False) -> HealingResult:
        """Heal missing token embeddings for specified document count."""
        
    def validate_token_embeddings_state(self, target_doc_count: int = 1000) -> ValidationResult:
        """Validate token embeddings completeness without healing."""
        
    def get_token_embeddings_healing_plan(self, target_doc_count: int = 1000) -> HealingPlan:
        """Generate plan for healing token embeddings without execution."""
        
    def get_target_document_set(self, doc_count: int = 1000) -> List[str]:
        """Get the target set of doc_ids for the specified count."""
        
    def scale_dataset_to_count(self, target_count: int) -> ScalingResult:
        """Ensure dataset has exactly target_count documents ready for evaluation."""
```

### 6.2 Dataset State Manager

```python
class DatasetStateManager:
    def ensure_target_state(self, target_state: str, auto_fix: bool = False) -> StateReport:
        """Ensure dataset reaches specified target state."""
        
    def validate_current_state(self, target_state: str) -> StateValidation:
        """Validate current dataset state against target."""
        
    def get_healing_plan(self, target_state: str) -> HealingPlan:
        """Generate healing plan without execution."""
```

## 7. Error Handling & Recovery

### 7.1 Failure Scenarios

- **Partial Token Embedding Generation:** Resume from last successful document
- **Database Connection Issues:** Retry with exponential backoff
- **Insufficient Resources:** Graceful degradation with smaller batches
- **Schema Inconsistencies:** Detailed error reporting with suggested fixes

### 7.2 Recovery Mechanisms

- **Progress Checkpointing:** Save progress after each batch
- **Rollback Support:** Undo partial changes on failure
- **Retry Logic:** Configurable retry attempts with backoff
- **Detailed Logging:** Comprehensive audit trail for debugging

## 8. Performance Considerations

### 8.1 Optimization Strategies

- **Batch Processing:** Process documents in configurable batches
- **Parallel Processing:** Multi-threaded token embedding generation
- **Memory Management:** Stream processing for large datasets
- **Database Optimization:** Efficient queries with proper indexing

### 8.2 Scalability

- **Horizontal Scaling:** Support for distributed processing
- **Resource Monitoring:** Track memory and CPU usage
- **Adaptive Batching:** Adjust batch sizes based on system performance
- **Progress Reporting:** Real-time status updates

## 9. Testing Strategy

### 9.1 Unit Tests

- Test granular healing methods in isolation
- Validate state detection logic
- Test error handling scenarios

### 9.2 Integration Tests

- End-to-end healing workflows
- Make target execution
- Cross-component interaction validation

### 9.3 Performance Tests

- Large dataset healing (1000+ documents)
- Memory usage validation
- Concurrent operation testing

## 10. Documentation Requirements

### 10.1 User Documentation

- Make target usage guide
- Target state specifications
- Troubleshooting guide

### 10.2 Developer Documentation

- API reference for enhanced components
- Architecture decision records
- Performance tuning guide
## 11. Scaling and Usage Examples

### 11.1 Common Usage Patterns

**Development (1K documents):**
```bash
make ensure-ragas-ready DOCS=1000
# or
make ensure-ragas-1k
```

**Testing (5K documents):**
```bash
make ensure-ragas-ready DOCS=5000
# or  
make ensure-ragas-5k
```

**Production Benchmarking (10K+ documents):**
```bash
make ensure-ragas-ready DOCS=10000
make ensure-ragas-ready DOCS=50000
```

**Validation Only:**
```bash
make validate-dataset-state DOCS=1000
make validate-dataset-state DOCS=5000
```

**Token Embeddings Only:**
```bash
make populate-token-embeddings DOCS=1000
make populate-token-embeddings DOCS=5000
```

### 11.2 Scaling Architecture

**Document Selection Strategy:**
- Use deterministic ordering (`ORDER BY doc_id`) for consistent target sets
- Top N documents approach ensures reproducible results
- Support for different selection criteria (newest, random, etc.)

**Performance Scaling:**
```python
# Adaptive batch sizing based on document count
def get_optimal_batch_size(doc_count: int) -> int:
    if doc_count <= 1000:
        return 16
    elif doc_count <= 5000:
        return 32
    elif doc_count <= 10000:
        return 64
    else:
        return 128

# Memory-aware processing
def process_with_memory_management(doc_count: int):
    batch_size = get_optimal_batch_size(doc_count)
    max_memory_mb = get_available_memory() * 0.8
    
    if estimated_memory_usage(batch_size) > max_memory_mb:
        batch_size = calculate_safe_batch_size(max_memory_mb)
```

### 11.3 Resource Planning

**Memory Requirements by Scale:**
- 1K documents: ~2GB RAM
- 5K documents: ~8GB RAM  
- 10K documents: ~16GB RAM
- 50K+ documents: Consider distributed processing

**Processing Time Estimates:**
- 1K documents: ~5-10 minutes
- 5K documents: ~25-50 minutes
- 10K documents: ~1-2 hours
- 50K+ documents: ~5-10 hours

### 11.4 Monitoring and Progress Tracking

**Progress Reporting:**
```python
class ProgressTracker:
    def __init__(self, total_docs: int):
        self.total_docs = total_docs
        self.processed_docs = 0
        self.start_time = time.time()
    
    def update(self, docs_processed: int):
        self.processed_docs += docs_processed
        progress_pct = (self.processed_docs / self.total_docs) * 100
        elapsed = time.time() - self.start_time
        eta = (elapsed / self.processed_docs) * (self.total_docs - self.processed_docs)
        
        print(f"Progress: {progress_pct:.1f}% ({self.processed_docs}/{self.total_docs}) ETA: {eta:.0f}s")
```

This architecture ensures clean separation of concerns, extensible design, and efficient self-healing capabilities while maintaining the existing system's integrity. The parameterized approach allows seamless scaling from development (1K docs) to production benchmarking (50K+ docs) with consistent interfaces and predictable resource usage.

This architecture ensures clean separation of concerns, extensible design, and efficient self-healing capabilities while maintaining the existing system's integrity.