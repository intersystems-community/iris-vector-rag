# Declarative State Management for RAG Templates

## Vision

Instead of imperatively managing documents (`add_documents`, `delete_documents`), declare the desired state and let the system reconcile reality with the specification.

```python
# Instead of this (imperative):
rag.add_documents(["doc1", "doc2"])
rag.delete_document("doc3")

# Do this (declarative):
rag.sync_state({
    "documents": [
        {"id": "doc1", "content": "...", "version": "1.0"},
        {"id": "doc2", "content": "...", "version": "1.0"}
    ],
    "expected_count": 2,
    "validation": "strict"
})
```

## Core Concepts

### 1. State Specification

```yaml
# rag_state.yaml
state:
  documents:
    source: "data/pmc_oas_downloaded"
    count: 1000
    selection:
      strategy: "latest"  # or "random", "specific"
      criteria:
        - has_abstract: true
        - min_length: 500
    
  embeddings:
    model: "all-MiniLM-L6-v2"
    dimension: 384
    
  chunks:
    strategy: "semantic"
    size: 512
    overlap: 50
    
  validation:
    mode: "strict"  # fail if can't achieve state
    tolerance: 0.95  # accept 95% of target
```

### 2. Drift Detection

```python
class StateManager:
    """Manages declarative state for RAG system."""
    
    def detect_drift(self, desired_state: Dict) -> DriftReport:
        """Detect differences between current and desired state."""
        current = self.get_current_state()
        
        drift = DriftReport()
        
        # Document drift
        drift.document_drift = self._compare_documents(
            current.documents, 
            desired_state["documents"]
        )
        
        # Embedding drift
        drift.embedding_drift = self._compare_embeddings(
            current.embeddings,
            desired_state["embeddings"]
        )
        
        # Chunk drift
        drift.chunk_drift = self._compare_chunks(
            current.chunks,
            desired_state["chunks"]
        )
        
        return drift
    
    def reconcile(self, drift: DriftReport) -> ReconciliationPlan:
        """Create plan to reconcile drift."""
        plan = ReconciliationPlan()
        
        # Documents to add
        plan.add_documents = drift.missing_documents
        
        # Documents to update
        plan.update_documents = drift.outdated_documents
        
        # Documents to remove
        plan.remove_documents = drift.extra_documents
        
        # Re-embedding needed
        plan.reembed = drift.embedding_model_changed
        
        return plan
```

### 3. State Reconciliation

```python
class DeclarativeRAG(RAG):
    """RAG system with declarative state management."""
    
    def __init__(self, state_spec: Union[str, Dict]):
        super().__init__()
        self.state_manager = StateManager()
        self.desired_state = self._load_state_spec(state_spec)
    
    async def sync_state(self, 
                        mode: str = "auto",
                        dry_run: bool = False) -> SyncReport:
        """Sync to desired state."""
        
        # Detect drift
        drift = self.state_manager.detect_drift(self.desired_state)
        
        if not drift.has_drift():
            return SyncReport(status="in_sync")
        
        # Create reconciliation plan
        plan = self.state_manager.reconcile(drift)
        
        if dry_run:
            return SyncReport(
                status="would_change",
                plan=plan
            )
        
        # Execute plan
        if mode == "auto":
            return await self._execute_plan(plan)
        elif mode == "interactive":
            return await self._interactive_sync(plan)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    async def _execute_plan(self, plan: ReconciliationPlan) -> SyncReport:
        """Execute reconciliation plan."""
        report = SyncReport()
        
        # Add missing documents
        if plan.add_documents:
            added = await self._add_documents_batch(plan.add_documents)
            report.documents_added = len(added)
        
        # Update outdated documents
        if plan.update_documents:
            updated = await self._update_documents_batch(plan.update_documents)
            report.documents_updated = len(updated)
        
        # Remove extra documents
        if plan.remove_documents:
            removed = await self._remove_documents_batch(plan.remove_documents)
            report.documents_removed = len(removed)
        
        # Re-embed if needed
        if plan.reembed:
            reembedded = await self._reembed_all_documents()
            report.documents_reembedded = len(reembedded)
        
        return report
```

## Integration with Test Isolation

### 1. Declarative Test States

```python
@pytest.fixture
def declarative_test_state():
    """Provides declarative state management for tests."""
    
    def _create_state(spec: Dict) -> DeclarativeTestEnvironment:
        env = DeclarativeTestEnvironment()
        
        # Define desired state
        env.declare_state({
            "documents": spec.get("documents", []),
            "expected_counts": {
                "documents": spec.get("doc_count", 0),
                "chunks": spec.get("chunk_count", 0),
                "embeddings": spec.get("embedding_count", 0)
            },
            "validation": spec.get("validation", "strict")
        })
        
        # Sync to desired state
        env.sync()
        
        return env
    
    return _create_state

class TestWithDeclarativeState:
    
    def test_exact_document_count(self, declarative_test_state):
        """Test with exact document count."""
        
        # Declare desired state
        env = declarative_test_state({
            "doc_count": 100,
            "documents": generate_test_documents(100)
        })
        
        # System automatically achieves this state
        assert env.get_document_count() == 100
        
        # Even if documents exist from other tests
        # the system ensures exactly 100
    
    def test_drift_correction(self, declarative_test_state):
        """Test drift detection and correction."""
        
        # Initial state
        env = declarative_test_state({
            "doc_count": 50,
            "validation": "strict"
        })
        
        # Manually cause drift
        env.connection.execute("DELETE FROM Documents WHERE id < 10")
        
        # Re-sync detects and fixes drift
        report = env.sync()
        
        assert report.documents_added == 10
        assert env.get_document_count() == 50
```

### 2. MCP Integration with Declarative State

```typescript
// MCP server with declarative state
class DeclarativeMCPServer {
  private stateManager: StateManager;
  
  async initialize(stateSpec: StateSpecification) {
    this.stateManager = new StateManager(stateSpec);
    
    // Ensure initial state
    await this.stateManager.sync();
    
    // Monitor for drift
    this.startDriftMonitor();
  }
  
  async handleQuery(query: string) {
    // Check state before query
    const drift = await this.stateManager.checkDrift();
    
    if (drift.isSignificant()) {
      // Auto-heal before query
      await this.stateManager.sync();
    }
    
    return this.ragEngine.query(query);
  }
  
  private startDriftMonitor() {
    setInterval(async () => {
      const drift = await this.stateManager.checkDrift();
      
      if (drift.exists()) {
        console.log(`Drift detected: ${drift.summary()}`);
        
        if (this.config.autoHeal) {
          await this.stateManager.sync();
        }
      }
    }, this.config.driftCheckInterval);
  }
}
```

### 3. State Versioning and Migration

```python
class VersionedStateManager(StateManager):
    """State manager with version support."""
    
    def __init__(self):
        super().__init__()
        self.migrations = {}
    
    def register_migration(self, 
                          from_version: str, 
                          to_version: str, 
                          migration_func: Callable):
        """Register a state migration."""
        key = f"{from_version}->{to_version}"
        self.migrations[key] = migration_func
    
    async def migrate_state(self, 
                           current_version: str,
                           target_version: str) -> MigrationReport:
        """Migrate state between versions."""
        
        # Find migration path
        path = self._find_migration_path(current_version, target_version)
        
        if not path:
            raise ValueError(f"No migration path from {current_version} to {target_version}")
        
        # Execute migrations in sequence
        report = MigrationReport()
        
        for step in path:
            migration = self.migrations[step]
            step_report = await migration()
            report.add_step(step, step_report)
        
        return report

# Example migration
async def migrate_v1_to_v2():
    """Migrate from schema v1 to v2."""
    # Add new metadata fields
    await db.execute("""
        ALTER TABLE Documents 
        ADD COLUMN version VARCHAR(50),
        ADD COLUMN checksum VARCHAR(64)
    """)
    
    # Backfill data
    await db.execute("""
        UPDATE Documents 
        SET version = '1.0',
            checksum = HASH(content)
        WHERE version IS NULL
    """)
    
    return {"documents_migrated": count}
```

## Implementation Plan

### Phase 1: Core Drift Detection
```python
# 1. Implement state inspection
def get_current_state() -> SystemState:
    return SystemState(
        document_count=count_documents(),
        chunk_count=count_chunks(),
        embedding_model=get_embedding_model(),
        # ... etc
    )

# 2. Implement state comparison
def compare_states(current: SystemState, 
                   desired: SystemState) -> DriftReport:
    # Compare all aspects
    pass

# 3. Basic reconciliation
def create_reconciliation_plan(drift: DriftReport) -> Plan:
    # Generate steps to fix drift
    pass
```

### Phase 2: Declarative API
```python
# 1. State specification parser
def parse_state_spec(spec: Union[str, Dict]) -> StateSpec:
    # Handle YAML, JSON, Python dict
    pass

# 2. Declarative RAG class
class DeclarativeRAG(RAG):
    def sync_state(self, spec: StateSpec):
        # Main sync logic
        pass

# 3. Progress reporting
def sync_with_progress(spec: StateSpec) -> Generator:
    # Yield progress updates
    pass
```

### Phase 3: Test Integration
```python
# 1. Test fixtures
@pytest.fixture
def declared_state():
    # Declarative state for tests
    pass

# 2. Test utilities
def assert_state_matches(expected: StateSpec):
    # Verify state matches spec
    pass

# 3. MCP test helpers
async def sync_mcp_state(spec: StateSpec):
    # Sync across Python and Node.js
    pass
```

## Benefits

1. **Reproducible Tests**: Declare exactly what state you want
2. **Self-Healing**: System detects and fixes drift automatically
3. **MCP Friendly**: Node.js and Python stay in sync
4. **Version Control**: State specs can be versioned with code
5. **Debugging**: Clear view of expected vs actual state
6. **CI/CD**: Declarative specs work well in pipelines

## Example Usage

```python
# In tests
def test_with_exact_state():
    rag = DeclarativeRAG({
        "documents": {
            "count": 100,
            "source": "test_data/"
        },
        "embeddings": {
            "model": "all-MiniLM-L6-v2"
        }
    })
    
    # System ensures exactly 100 docs
    rag.sync_state()
    
    result = rag.query("test")
    assert len(result.documents) > 0

# In production
rag = DeclarativeRAG("config/production_state.yaml")

# Periodic sync
async def maintenance():
    while True:
        drift = rag.detect_drift()
        if drift.exists():
            logger.info(f"Fixing drift: {drift}")
            rag.sync_state()
        await asyncio.sleep(300)  # Check every 5 min

# In MCP server
const server = new MCPServer({
  stateSpec: {
    documents: { count: 1000 },
    autoHeal: true,
    healInterval: 60000  // 1 min
  }
});
```