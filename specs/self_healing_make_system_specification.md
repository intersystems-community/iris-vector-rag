# Self-Healing Make System Specification

## Executive Summary

This specification defines a comprehensive self-healing make system that automatically detects incomplete data and triggers population processes to achieve 100% table readiness without custom Python scripts. The system builds upon the existing [`DataPopulationOrchestrator`](rag_templates/validation/data_population_orchestrator.py:16) and integrates seamlessly with the current [`Makefile`](Makefile:1) infrastructure.

## Current State Analysis

### Database Status
- **RAG.SourceDocuments**: 1,006 docs ✅ (populated)
- **RAG.ColBERTTokenEmbeddings**: 0 records ❌ (empty)
- **RAG.ChunkedDocuments**: 0 records ❌ (empty)
- **RAG.GraphRAGEntities**: 0 records ❌ (empty)
- **RAG.GraphRAGRelationships**: 0 records ❌ (empty)
- **RAG.KnowledgeGraphNodes**: 0 records ❌ (empty)
- **RAG.DocumentEntities**: 0 records ❌ (empty)

### Existing Infrastructure
- ✅ [`DataPopulationOrchestrator`](rag_templates/validation/data_population_orchestrator.py:16) with dependency-aware table ordering
- ✅ [`Makefile`](Makefile:1) with comprehensive validation and auto-setup targets
- ✅ [`iris_rag`](iris_rag/__init__.py:1) package with working pipeline instantiation
- ✅ DBAPI connection infrastructure via [`get_iris_connection()`](common/iris_connection_manager.py:1)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Self-Healing Make System                     │
├─────────────────────────────────────────────────────────────────┤
│  1. Table Status Detection                                      │
│     ├── TableStatusDetector                                     │
│     ├── DependencyAnalyzer                                      │
│     └── ReadinessCalculator                                     │
│                                                                 │
│  2. Self-Healing Orchestration                                 │
│     ├── SelfHealingOrchestrator                                │
│     ├── PopulationTaskManager                                  │
│     └── ErrorRecoveryHandler                                   │
│                                                                 │
│  3. Make Integration Layer                                      │
│     ├── MakeTargetGenerator                                     │
│     ├── ProgressReporter                                       │
│     └── ValidationIntegrator                                   │
│                                                                 │
│  4. Enhanced DataPopulationOrchestrator                        │
│     ├── Existing population methods                            │
│     ├── Self-healing capabilities                              │
│     └── Progress tracking                                      │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Table Status Detection Module

#### 1.1 TableStatusDetector
```python
class TableStatusDetector:
    """
    Detects current population status of all RAG tables.
    """
    
    def __init__(self, db_connection):
        self.db_connection = db_connection
        self.required_tables = [
            "RAG.SourceDocuments",
            "RAG.ColBERTTokenEmbeddings", 
            "RAG.ChunkedDocuments",
            "RAG.GraphRAGEntities",
            "RAG.GraphRAGRelationships",
            "RAG.KnowledgeGraphNodes",
            "RAG.DocumentEntities"
        ]
    
    def detect_table_status(self) -> Dict[str, TableStatus]:
        """
        Returns comprehensive status for each table.
        
        Returns:
            Dict mapping table names to TableStatus objects containing:
            - record_count: int
            - is_populated: bool
            - last_updated: datetime
            - health_score: float (0.0-1.0)
            - dependencies_met: bool
        """
        pass
    
    def calculate_overall_readiness(self) -> ReadinessReport:
        """
        Calculates system-wide readiness percentage.
        
        Returns:
            ReadinessReport with:
            - overall_percentage: float
            - populated_tables: int
            - total_tables: int
            - missing_tables: List[str]
            - blocking_issues: List[str]
        """
        pass
```

#### 1.2 DependencyAnalyzer
```python
class DependencyAnalyzer:
    """
    Analyzes table dependencies and determines population order.
    """
    
    def __init__(self):
        self.dependency_graph = {
            "RAG.ChunkedDocuments": ["RAG.SourceDocuments"],
            "RAG.ColBERTTokenEmbeddings": ["RAG.SourceDocuments"],
            "RAG.GraphRAGEntities": ["RAG.SourceDocuments"],
            "RAG.GraphRAGRelationships": ["RAG.GraphRAGEntities"],
            "RAG.KnowledgeGraphNodes": ["RAG.GraphRAGEntities"],
            "RAG.DocumentEntities": ["RAG.SourceDocuments", "RAG.GraphRAGEntities"]
        }
    
    def get_population_order(self, missing_tables: List[str]) -> List[str]:
        """
        Returns optimal population order respecting dependencies.
        """
        pass
    
    def validate_dependencies(self, table_status: Dict[str, TableStatus]) -> List[str]:
        """
        Returns list of dependency violations.
        """
        pass
```

### 2. Self-Healing Orchestration Module

#### 2.1 SelfHealingOrchestrator
```python
class SelfHealingOrchestrator:
    """
    Main orchestrator for self-healing data population.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.detector = TableStatusDetector(get_iris_connection())
        self.analyzer = DependencyAnalyzer()
        self.population_orchestrator = DataPopulationOrchestrator(config, get_iris_connection())
        self.task_manager = PopulationTaskManager()
        self.error_handler = ErrorRecoveryHandler()
    
    def run_self_healing_cycle(self) -> SelfHealingResult:
        """
        Executes complete self-healing cycle.
        
        Process:
        1. Detect current table status
        2. Analyze dependencies and missing data
        3. Generate population plan
        4. Execute population tasks with error recovery
        5. Validate results and retry if needed
        6. Report final status
        
        Returns:
            SelfHealingResult with:
            - success: bool
            - initial_readiness: float
            - final_readiness: float
            - tables_populated: List[str]
            - errors_encountered: List[str]
            - execution_time: float
            - recommendations: List[str]
        """
        pass
    
    def detect_and_heal(self, target_readiness: float = 1.0) -> bool:
        """
        Simplified interface for make targets.
        
        Args:
            target_readiness: Desired readiness percentage (0.0-1.0)
            
        Returns:
            True if target readiness achieved, False otherwise
        """
        pass
```

#### 2.2 PopulationTaskManager
```python
class PopulationTaskManager:
    """
    Manages individual population tasks with progress tracking.
    """
    
    def __init__(self):
        self.active_tasks = {}
        self.completed_tasks = {}
        self.failed_tasks = {}
    
    def create_population_plan(self, missing_tables: List[str]) -> PopulationPlan:
        """
        Creates detailed execution plan for populating missing tables.
        
        Returns:
            PopulationPlan with:
            - tasks: List[PopulationTask]
            - estimated_duration: float
            - resource_requirements: Dict[str, Any]
            - parallel_groups: List[List[str]]
        """
        pass
    
    def execute_task(self, task: PopulationTask) -> TaskResult:
        """
        Executes single population task with progress tracking.
        """
        pass
    
    def get_progress_summary(self) -> ProgressSummary:
        """
        Returns current progress across all tasks.
        """
        pass
```

#### 2.3 ErrorRecoveryHandler
```python
class ErrorRecoveryHandler:
    """
    Handles errors and implements recovery strategies.
    """
    
    def __init__(self):
        self.recovery_strategies = {
            "connection_error": self._recover_connection,
            "data_corruption": self._recover_data_corruption,
            "dependency_violation": self._recover_dependency_violation,
            "resource_exhaustion": self._recover_resource_exhaustion
        }
    
    def handle_error(self, error: Exception, context: Dict[str, Any]) -> RecoveryResult:
        """
        Analyzes error and attempts recovery.
        
        Returns:
            RecoveryResult with:
            - recovery_attempted: bool
            - recovery_successful: bool
            - retry_recommended: bool
            - alternative_strategy: Optional[str]
        """
        pass
    
    def _recover_connection(self, context: Dict[str, Any]) -> bool:
        """Attempts to recover database connection."""
        pass
    
    def _recover_data_corruption(self, context: Dict[str, Any]) -> bool:
        """Attempts to recover from data corruption."""
        pass
```

### 3. Make Integration Layer

#### 3.1 Enhanced Makefile Targets

```makefile
# Self-Healing Data Population Targets
.PHONY: heal-data check-readiness populate-missing validate-healing

# Main self-healing target
heal-data:
	@echo "🔧 Running self-healing data population..."
	$(CONDA_RUN) python -c "from rag_templates.validation.self_healing_orchestrator import SelfHealingOrchestrator; orchestrator = SelfHealingOrchestrator(); result = orchestrator.detect_and_heal(); print('✅ Self-healing completed successfully' if result else '❌ Self-healing failed')"

# Check current readiness status
check-readiness:
	@echo "📊 Checking table readiness status..."
	$(CONDA_RUN) python -c "from rag_templates.validation.table_status_detector import TableStatusDetector; from common.iris_connection_manager import get_iris_connection; detector = TableStatusDetector(get_iris_connection()); report = detector.calculate_overall_readiness(); print(f'📈 Overall Readiness: {report.overall_percentage:.1f}% ({report.populated_tables}/{report.total_tables} tables)')"

# Populate only missing tables
populate-missing:
	@echo "🔄 Populating missing tables only..."
	$(CONDA_RUN) python -c "from rag_templates.validation.self_healing_orchestrator import SelfHealingOrchestrator; orchestrator = SelfHealingOrchestrator(); result = orchestrator.run_self_healing_cycle(); print(f'📊 Readiness improved from {result.initial_readiness:.1f}% to {result.final_readiness:.1f}%')"

# Validate healing effectiveness
validate-healing:
	@echo "✅ Validating self-healing effectiveness..."
	$(MAKE) check-readiness
	$(MAKE) validate-all-pipelines

# Complete self-healing workflow
auto-heal-all: check-readiness heal-data validate-healing
	@echo "🎉 Complete self-healing workflow finished!"

# Self-healing with specific target
heal-to-target:
	@if [ -z "$(TARGET)" ]; then \
		echo "Error: TARGET parameter required. Usage: make heal-to-target TARGET=0.8"; \
		echo "TARGET should be between 0.0 and 1.0 (e.g., 0.8 = 80% readiness)"; \
		exit 1; \
	fi
	@echo "🎯 Healing to $(TARGET) readiness target..."
	$(CONDA_RUN) python -c "from rag_templates.validation.self_healing_orchestrator import SelfHealingOrchestrator; orchestrator = SelfHealingOrchestrator(); result = orchestrator.detect_and_heal(target_readiness=$(TARGET)); print('✅ Target achieved' if result else '❌ Target not achieved')"

# Progressive healing (incremental approach)
heal-progressive:
	@echo "📈 Running progressive healing (incremental approach)..."
	@for target in 0.3 0.5 0.7 0.9 1.0; do \
		echo ""; \
		echo "=== Healing to $$target readiness ==="; \
		$(MAKE) heal-to-target TARGET=$$target || echo "⚠ Target $$target not achieved"; \
		sleep 2; \
	done
	@echo ""
	@echo "=== PROGRESSIVE HEALING COMPLETE ==="

# Emergency healing (force repopulation)
heal-emergency:
	@echo "🚨 Running emergency healing (force repopulation)..."
	$(CONDA_RUN) python -c "from rag_templates.validation.self_healing_orchestrator import SelfHealingOrchestrator; orchestrator = SelfHealingOrchestrator({'force_repopulation': True}); result = orchestrator.run_self_healing_cycle(); print('🚨 Emergency healing completed')"
```

#### 3.2 MakeTargetGenerator
```python
class MakeTargetGenerator:
    """
    Generates dynamic make targets based on current system state.
    """
    
    def __init__(self, orchestrator: SelfHealingOrchestrator):
        self.orchestrator = orchestrator
    
    def generate_healing_targets(self) -> List[str]:
        """
        Generates make targets for current healing needs.
        
        Returns:
            List of make target commands to execute
        """
        pass
    
    def generate_progress_targets(self) -> List[str]:
        """
        Generates targets for progress monitoring.
        """
        pass
```

#### 3.3 ProgressReporter
```python
class ProgressReporter:
    """
    Provides real-time progress reporting for make targets.
    """
    
    def __init__(self):
        self.start_time = None
        self.progress_callbacks = []
    
    def start_reporting(self, total_tasks: int):
        """Starts progress reporting session."""
        pass
    
    def update_progress(self, completed_tasks: int, current_task: str):
        """Updates progress display."""
        pass
    
    def finish_reporting(self, success: bool, summary: str):
        """Completes progress reporting."""
        pass
```

### 4. Enhanced DataPopulationOrchestrator

#### 4.1 Self-Healing Extensions
```python
# Extensions to existing DataPopulationOrchestrator class

def enable_self_healing(self, enabled: bool = True):
    """
    Enables self-healing capabilities.
    """
    self.self_healing_enabled = enabled
    self.max_retry_attempts = 3
    self.retry_delay = 5.0

def populate_with_healing(self, table_name: str) -> Tuple[bool, int, str]:
    """
    Populates table with self-healing retry logic.
    
    Args:
        table_name: Name of table to populate
        
    Returns:
        Tuple of (success, record_count, details)
    """
    for attempt in range(self.max_retry_attempts):
        try:
            success, count, details = self._populate_table(table_name)
            if success:
                return success, count, details
            
            # Attempt healing before retry
            if self.self_healing_enabled and attempt < self.max_retry_attempts - 1:
                self._attempt_table_healing(table_name)
                time.sleep(self.retry_delay)
                
        except Exception as e:
            if attempt == self.max_retry_attempts - 1:
                return False, 0, f"Failed after {self.max_retry_attempts} attempts: {str(e)}"
            
            logger.warning(f"Attempt {attempt + 1} failed for {table_name}: {e}")
            time.sleep(self.retry_delay)
    
    return False, 0, f"All {self.max_retry_attempts} attempts failed"

def get_population_progress(self) -> Dict[str, Any]:
    """
    Returns current population progress.
    
    Returns:
        Dict with:
        - total_tables: int
        - completed_tables: int
        - current_table: Optional[str]
        - estimated_completion: Optional[datetime]
        - errors: List[str]
    """
    pass
```

#### 4.2 Enhanced Population Methods

```python
def _populate_graphrag_entities(self) -> Tuple[bool, int, str]:
    """
    Enhanced GraphRAG entity extraction with self-healing.
    """
    try:
        # Import required modules
        from common.utils import get_llm_func
        llm_func = get_llm_func()
        
        if not llm_func:
            return False, 0, "Could not load LLM function for entity extraction"
        
        cursor = self.db_connection.cursor()
        
        # Get source documents for entity extraction
        cursor.execute("""
            SELECT doc_id, title, content 
            FROM RAG.SourceDocuments 
            WHERE doc_id NOT IN (
                SELECT DISTINCT source_doc_id 
                FROM RAG.GraphRAGEntities 
                WHERE source_doc_id IS NOT NULL
            )
            LIMIT 100
        """)
        
        source_docs = cursor.fetchall()
        
        if not source_docs:
            return False, 0, "No source documents found for entity extraction"
        
        total_entities = 0
        
        for doc in source_docs:
            doc_id, title, content = doc
            text = content or title
            
            if not text:
                continue
            
            # Extract entities using LLM
            entity_prompt = f"""
            Extract named entities from the following biomedical text. 
            Return entities as JSON list with fields: name, type, description.
            
            Text: {text[:2000]}
            """
            
            try:
                response = llm_func(entity_prompt)
                entities = self._parse_entity_response(response)
                
                for entity in entities:
                    entity_id = f"{doc_id}_{entity['name'].replace(' ', '_')}"
                    
                    insert_sql = """
                    INSERT INTO RAG.GraphRAGEntities 
                    (entity_id, entity_name, entity_type, description, source_doc_id)
                    VALUES (?, ?, ?, ?, ?)
                    """
                    
                    cursor.execute(insert_sql, [
                        entity_id,
                        entity['name'],
                        entity.get('type', 'UNKNOWN'),
                        entity.get('description', ''),
                        doc_id
                    ])
                    total_entities += 1
                    
            except Exception as e:
                logger.warning(f"Entity extraction failed for {doc_id}: {e}")
                continue
        
        self.db_connection.commit()
        cursor.close()
        
        return True, total_entities, f"Successfully extracted {total_entities} entities"
        
    except Exception as e:
        logger.error(f"Error populating GraphRAG entities: {e}")
        return False, 0, f"Error during entity extraction: {str(e)}"

def _parse_entity_response(self, response: str) -> List[Dict[str, str]]:
    """
    Parses LLM response to extract entity information.
    """
    try:
        import json
        import re
        
        # Try to extract JSON from response
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            entities = json.loads(json_match.group())
            return entities
        
        # Fallback: simple text parsing
        entities = []
        lines = response.split('\n')
        for line in lines:
            if ':' in line and any(keyword in line.lower() for keyword in ['protein', 'gene', 'disease', 'drug']):
                parts = line.split(':')
                if len(parts) >= 2:
                    entities.append({
                        'name': parts[0].strip(),
                        'type': 'BIOMEDICAL',
                        'description': parts[1].strip()
                    })
        
        return entities[:10]  # Limit to 10 entities per document
        
    except Exception as e:
        logger.warning(f"Failed to parse entity response: {e}")
        return []
```

## Integration with Existing Infrastructure

### 1. Makefile Integration

The self-healing system integrates with the existing [`Makefile`](Makefile:1) by:

1. **Extending existing targets**: Adding self-healing capabilities to [`auto-setup-all`](Makefile:191) and [`validate-all-pipelines`](Makefile:181)
2. **New healing targets**: Adding dedicated targets for different healing scenarios
3. **Progress reporting**: Integrating with existing validation and testing workflows
4. **Backward compatibility**: Maintaining all existing functionality

### 2. DataPopulationOrchestrator Enhancement

The existing [`DataPopulationOrchestrator`](rag_templates/validation/data_population_orchestrator.py:16) is enhanced with:

1. **Self-healing retry logic**: Automatic retry with exponential backoff
2. **Progress tracking**: Real-time progress reporting for make targets
3. **Error recovery**: Intelligent error handling and recovery strategies
4. **Dependency validation**: Enhanced dependency checking and resolution

### 3. Validation System Integration

Integration with the existing validation system via [`ComprehensiveValidationRunner`](rag_templates/validation/comprehensive_validation_runner.py:1):

1. **Pre-validation healing**: Automatic healing before validation runs
2. **Post-validation reporting**: Detailed readiness reports after healing
3. **Pipeline integration**: Seamless integration with pipeline validation

## Implementation Plan

### Phase 1: Core Detection (Week 1)
- [ ] Implement [`TableStatusDetector`](specs/self_healing_make_system_specification.md:67)
- [ ] Implement [`DependencyAnalyzer`](specs/self_healing_make_system_specification.md:95)
- [ ] Create basic readiness calculation
- [ ] Add unit tests for detection logic

### Phase 2: Self-Healing Orchestration (Week 2)
- [ ] Implement [`SelfHealingOrchestrator`](specs/self_healing_make_system_specification.md:125)
- [ ] Implement [`PopulationTaskManager`](specs/self_healing_make_system_specification.md:168)
- [ ] Implement [`ErrorRecoveryHandler`](specs/self_healing_make_system_specification.md:194)
- [ ] Add integration tests

### Phase 3: Make Integration (Week 3)
- [ ] Add new make targets for self-healing
- [ ] Implement [`ProgressReporter`](specs/self_healing_make_system_specification.md:290)
- [ ] Integrate with existing validation workflows
- [ ] Add end-to-end tests

### Phase 4: Enhanced Population Methods (Week 4)
- [ ] Enhance [`DataPopulationOrchestrator`](rag_templates/validation/data_population_orchestrator.py:16) with self-healing
- [ ] Implement enhanced GraphRAG entity extraction
- [ ] Implement enhanced relationship extraction
- [ ] Add comprehensive error handling

### Phase 5: Testing and Validation (Week 5)
- [ ] Comprehensive testing with 1000+ documents
- [ ] Performance optimization
- [ ] Documentation and examples
- [ ] Production readiness validation

## Testing Strategy

### 1. Unit Tests
```python
# tests/test_self_healing_system.py

class TestTableStatusDetector:
    def test_detect_empty_tables(self):
        """Test detection of empty tables."""
        pass
    
    def test_detect_populated_tables(self):
        """Test detection of populated tables."""
        pass
    
    def test_calculate_readiness_percentage(self):
        """Test readiness calculation."""
        pass

class TestSelfHealingOrchestrator:
    def test_healing_cycle_with_missing_data(self):
        """Test complete healing cycle."""
        pass
    
    def test_error_recovery(self):
        """Test error recovery mechanisms."""
        pass
    
    def test_progress_tracking(self):
        """Test progress tracking functionality."""
        pass
```

### 2. Integration Tests
```python
# tests/test_self_healing_integration.py

class TestMakeIntegration:
    def test_heal_data_target(self):
        """Test make heal-data target."""
        pass
    
    def test_check_readiness_target(self):
        """Test make check-readiness target."""
        pass
    
    def test_progressive_healing(self):
        """Test progressive healing workflow."""
        pass

class TestDataPopulationIntegration:
    def test_enhanced_population_methods(self):
        """Test enhanced population methods."""
        pass
    
    def test_self_healing_retry_logic(self):
        """Test retry logic with self-healing."""
        pass
```

### 3. End-to-End Tests
```python
# tests/test_self_healing_e2e.py

class TestSelfHealingE2E:
    def test_complete_healing_workflow(self):
        """Test complete self-healing workflow with 1000+ docs."""
        pass
    
    def test_healing_with_pipeline_validation(self):
        """Test healing integrated with pipeline validation."""
        pass
    
    def test_emergency_healing_scenario(self):
        """Test emergency healing with corrupted data."""
        pass
```

## Success Criteria

### 1. Functional Requirements
- ✅ **Automatic Detection**: System detects incomplete data without manual intervention
- ✅ **Self-Healing**: System automatically triggers population processes
- ✅ **100% Readiness**: System achieves 100% table readiness
- ✅ **Make Integration**: Seamless integration with existing Makefile
- ✅ **No Custom Scripts**: No need for custom Python scripts in production

### 2. Performance Requirements
- ✅ **Detection Speed**: Table status detection completes in < 5 seconds
- ✅ **Healing Speed**: Self-healing completes in < 10 minutes for 1000 docs
- ✅ **Memory Usage**: System uses < 2GB RAM during healing
- ✅ **Error Recovery**: System recovers from 95% of common errors

### 3. Quality Requirements
- ✅ **Test Coverage**: > 90% test coverage for all components
- ✅ **Documentation**: Comprehensive documentation and examples
- ✅ **Modularity**: All components < 500 lines and independently testable
- ✅ **Backward Compatibility**: No breaking changes to existing functionality

## Usage Examples

### 1. Basic Self-Healing
```bash
# Check current readiness
make check-readiness

# Run self-healing
make heal-data

# Validate results
make validate-healing
```

### 2. Progressive Healing
```bash
# Heal incrementally to 100%
make heal-progressive

# Heal to specific target (80%)
make heal-to-target TARGET=0.8
```

### 3. Emergency Healing
```bash
# Force repopulation of all tables
make heal-emergency

# Complete workflow with validation
make auto-heal-all
```

### 4. Integration with Existing Workflows
```bash
# Development setup with self-healing
make dev-setup

# Production check with auto-healing
make prod-check

# Testing with auto-healing
make test-with-auto-setup
```

## Monitoring and Observability

### 1. Progress Reporting
- Real-time progress bars for make targets
- Detailed logging of healing operations
- Performance metrics collection
- Error tracking and reporting

### 2. Health Metrics
- Table readiness percentage
- Population success rates
- Error recovery statistics
- Performance benchmarks

### 3. Alerting
- Automatic alerts for healing failures
- Threshold-based notifications
- Integration with existing monitoring

## Security Considerations

### 1. Data Protection
- No sensitive data in logs
- Secure handling of database connections
- Proper error message sanitization

### 2. Access Control
- Database connection security
- Proper privilege management
- Audit trail for healing operations

### 3. Error Handling
- Graceful degradation on failures
- Secure error reporting
- No information leakage in error messages

## Conclusion

This self-healing make system specification provides a comprehensive solution for achieving 100% table readiness through automatic detection and population processes. The system builds upon existing infrastructure while adding intelligent self-healing capabilities that integrate seamlessly with the current Makefile and validation workflows.

The modular design ensures testability and maintainability while the progressive implementation plan allows for incremental delivery and validation. The system addresses all requirements while maintaining backward compatibility and following established patterns in the codebase.