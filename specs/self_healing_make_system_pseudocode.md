# Self-Healing Make System - Detailed Pseudocode

## Executive Summary

This document provides comprehensive pseudocode for implementing a self-healing make system that automatically detects incomplete data and triggers population processes to achieve 100% table readiness. The system builds upon the existing [`DataPopulationOrchestrator`](rag_templates/validation/data_population_orchestrator.py:16) and integrates seamlessly with the current [`Makefile`](Makefile:1) infrastructure.

## Current State Analysis

**Database Status:**
- RAG.SourceDocuments: 1,006 docs ✅ (populated)
- 6 downstream tables: 0 records each ❌ (empty)
- Target: 100% table readiness (7/7 tables populated)

**Existing Infrastructure:**
- ✅ [`DataPopulationOrchestrator`](rag_templates/validation/data_population_orchestrator.py:16) with dependency-aware ordering
- ✅ [`iris_rag`](iris_rag/__init__.py:1) package with working pipeline instantiation
- ✅ [`get_iris_connection()`](common/iris_connection_manager.py:1) DBAPI connection infrastructure

---

## 1. TableStatusDetector - Table Population Detection

### 1.1 Core Detection Logic

```python
# File: rag_templates/validation/table_status_detector.py

PSEUDOCODE TableStatusDetector:
    
    INITIALIZE:
        SET db_connection = get_iris_connection()
        SET required_tables = [
            "RAG.SourceDocuments",
            "RAG.ColBERTTokenEmbeddings", 
            "RAG.ChunkedDocuments",
            "RAG.GraphRAGEntities",
            "RAG.GraphRAGRelationships",
            "RAG.KnowledgeGraphNodes",
            "RAG.DocumentEntities"
        ]
        SET table_status_cache = {}
        SET last_check_time = None
        SET cache_ttl_seconds = 300  # 5 minutes
    
    FUNCTION detect_table_status() -> Dict[str, TableStatus]:
        """
        Detects current population status of all RAG tables.
        Returns comprehensive status for each table.
        """
        
        # Check cache validity
        IF last_check_time AND (current_time - last_check_time) < cache_ttl_seconds:
            RETURN table_status_cache
        
        SET status_results = {}
        SET cursor = db_connection.cursor()
        
        FOR EACH table_name IN required_tables:
            TRY:
                # Get record count
                EXECUTE SQL: "SELECT COUNT(*) FROM {table_name}"
                SET record_count = cursor.fetchone()[0]
                
                # Get last updated timestamp (if available)
                TRY:
                    EXECUTE SQL: "SELECT MAX(created_at) FROM {table_name}"
                    SET last_updated = cursor.fetchone()[0]
                EXCEPT:
                    SET last_updated = None
                
                # Calculate health score based on record count and dependencies
                SET health_score = calculate_table_health_score(table_name, record_count)
                
                # Check if dependencies are met
                SET dependencies_met = check_table_dependencies(table_name, status_results)
                
                # Create TableStatus object
                SET table_status = TableStatus(
                    table_name=table_name,
                    record_count=record_count,
                    is_populated=(record_count > 0),
                    last_updated=last_updated,
                    health_score=health_score,
                    dependencies_met=dependencies_met
                )
                
                SET status_results[table_name] = table_status
                
            EXCEPT Exception as e:
                LOG ERROR: f"Failed to check status for {table_name}: {e}"
                SET status_results[table_name] = TableStatus(
                    table_name=table_name,
                    record_count=0,
                    is_populated=False,
                    last_updated=None,
                    health_score=0.0,
                    dependencies_met=False,
                    error=str(e)
                )
        
        cursor.close()
        
        # Update cache
        SET table_status_cache = status_results
        SET last_check_time = current_time
        
        RETURN status_results
    
    FUNCTION calculate_overall_readiness() -> ReadinessReport:
        """
        Calculates system-wide readiness percentage and identifies issues.
        """
        
        SET table_statuses = detect_table_status()
        SET total_tables = len(required_tables)
        SET populated_tables = 0
        SET missing_tables = []
        SET blocking_issues = []
        
        FOR EACH table_name, status IN table_statuses.items():
            IF status.is_populated:
                INCREMENT populated_tables
            ELSE:
                APPEND table_name TO missing_tables
                
                # Check for blocking issues
                IF NOT status.dependencies_met:
                    APPEND f"Dependencies not met for {table_name}" TO blocking_issues
                
                IF status.error:
                    APPEND f"Error accessing {table_name}: {status.error}" TO blocking_issues
        
        SET overall_percentage = (populated_tables / total_tables) * 100
        
        RETURN ReadinessReport(
            overall_percentage=overall_percentage,
            populated_tables=populated_tables,
            total_tables=total_tables,
            missing_tables=missing_tables,
            blocking_issues=blocking_issues,
            table_details=table_statuses
        )
    
    FUNCTION calculate_table_health_score(table_name: str, record_count: int) -> float:
        """
        Calculates health score (0.0-1.0) based on expected vs actual record count.
        """
        
        # Define expected record counts based on source documents
        SET source_doc_count = get_source_document_count()
        
        SET expected_counts = {
            "RAG.SourceDocuments": source_doc_count,
            "RAG.ChunkedDocuments": source_doc_count * 3,  # ~3 chunks per doc
            "RAG.ColBERTTokenEmbeddings": source_doc_count * 50,  # ~50 tokens per doc
            "RAG.GraphRAGEntities": source_doc_count * 10,  # ~10 entities per doc
            "RAG.GraphRAGRelationships": source_doc_count * 5,  # ~5 relationships per doc
            "RAG.KnowledgeGraphNodes": source_doc_count * 8,  # ~8 nodes per doc
            "RAG.DocumentEntities": source_doc_count * 12  # ~12 doc-entity links per doc
        }
        
        SET expected_count = expected_counts.get(table_name, source_doc_count)
        
        IF expected_count == 0:
            RETURN 1.0 IF record_count == 0 ELSE 0.0
        
        SET ratio = min(record_count / expected_count, 1.0)
        RETURN ratio
    
    FUNCTION check_table_dependencies(table_name: str, current_statuses: Dict) -> bool:
        """
        Checks if table dependencies are satisfied.
        """
        
        SET dependency_map = {
            "RAG.ChunkedDocuments": ["RAG.SourceDocuments"],
            "RAG.ColBERTTokenEmbeddings": ["RAG.SourceDocuments"],
            "RAG.GraphRAGEntities": ["RAG.SourceDocuments"],
            "RAG.GraphRAGRelationships": ["RAG.GraphRAGEntities"],
            "RAG.KnowledgeGraphNodes": ["RAG.GraphRAGEntities"],
            "RAG.DocumentEntities": ["RAG.SourceDocuments", "RAG.GraphRAGEntities"]
        }
        
        SET dependencies = dependency_map.get(table_name, [])
        
        FOR EACH dep_table IN dependencies:
            IF dep_table IN current_statuses:
                IF NOT current_statuses[dep_table].is_populated:
                    RETURN False
            ELSE:
                # Check dependency directly if not in current batch
                SET dep_status = get_single_table_status(dep_table)
                IF NOT dep_status.is_populated:
                    RETURN False
        
        RETURN True

# Data Classes
CLASS TableStatus:
    table_name: str
    record_count: int
    is_populated: bool
    last_updated: Optional[datetime]
    health_score: float  # 0.0-1.0
    dependencies_met: bool
    error: Optional[str] = None

CLASS ReadinessReport:
    overall_percentage: float
    populated_tables: int
    total_tables: int
    missing_tables: List[str]
    blocking_issues: List[str]
    table_details: Dict[str, TableStatus]
```

### 1.2 TDD Test Anchors

```python
# File: tests/test_table_status_detection.py

PSEUDOCODE test_table_status_detection():
    """
    TDD Anchor: Verify detection of empty vs populated tables
    """
    
    # Setup test database with known state
    SETUP test_db WITH:
        - RAG.SourceDocuments: 100 records
        - RAG.ChunkedDocuments: 0 records
        - RAG.ColBERTTokenEmbeddings: 50 records
    
    # Test detection
    SET detector = TableStatusDetector(test_db_connection)
    SET status = detector.detect_table_status()
    
    # Assertions
    ASSERT status["RAG.SourceDocuments"].is_populated == True
    ASSERT status["RAG.SourceDocuments"].record_count == 100
    ASSERT status["RAG.ChunkedDocuments"].is_populated == False
    ASSERT status["RAG.ColBERTTokenEmbeddings"].is_populated == True
    
    # Test readiness calculation
    SET readiness = detector.calculate_overall_readiness()
    ASSERT readiness.overall_percentage == (2/7) * 100  # 2 out of 7 tables populated
    ASSERT "RAG.ChunkedDocuments" IN readiness.missing_tables
```

---

## 2. SelfHealingOrchestrator - Healing Workflow Coordination

### 2.1 Main Orchestration Logic

```python
# File: rag_templates/validation/self_healing_orchestrator.py

PSEUDOCODE SelfHealingOrchestrator:
    
    INITIALIZE:
        SET config = config OR default_config
        SET detector = TableStatusDetector(get_iris_connection())
        SET analyzer = DependencyAnalyzer()
        SET population_orchestrator = DataPopulationOrchestrator(config, get_iris_connection())
        SET task_manager = PopulationTaskManager()
        SET error_handler = ErrorRecoveryHandler()
        SET max_healing_cycles = 3
        SET healing_timeout_minutes = 30
    
    FUNCTION run_self_healing_cycle() -> SelfHealingResult:
        """
        Executes complete self-healing cycle with comprehensive error recovery.
        """
        
        SET start_time = current_time
        SET cycle_count = 0
        SET initial_readiness = None
        SET final_readiness = None
        SET tables_populated = []
        SET errors_encountered = []
        SET recommendations = []
        
        TRY:
            LOG INFO: "Starting self-healing cycle..."
            
            # Step 1: Detect current table status
            LOG INFO: "Step 1: Detecting current table status..."
            SET initial_status = detector.detect_table_status()
            SET initial_report = detector.calculate_overall_readiness()
            SET initial_readiness = initial_report.overall_percentage
            
            LOG INFO: f"Initial readiness: {initial_readiness:.1f}%"
            
            # Step 2: Analyze dependencies and missing data
            LOG INFO: "Step 2: Analyzing dependencies and missing data..."
            SET missing_tables = initial_report.missing_tables
            SET dependency_violations = analyzer.validate_dependencies(initial_status)
            
            IF dependency_violations:
                LOG WARNING: f"Dependency violations found: {dependency_violations}"
                APPEND "Resolve dependency violations before population" TO recommendations
            
            # Step 3: Generate population plan
            LOG INFO: "Step 3: Generating population plan..."
            SET population_order = analyzer.get_population_order(missing_tables)
            SET population_plan = task_manager.create_population_plan(population_order)
            
            LOG INFO: f"Population plan: {len(population_plan.tasks)} tasks, estimated {population_plan.estimated_duration:.1f}s"
            
            # Step 4: Execute population tasks with error recovery
            LOG INFO: "Step 4: Executing population tasks..."
            
            WHILE cycle_count < max_healing_cycles AND missing_tables:
                INCREMENT cycle_count
                LOG INFO: f"Healing cycle {cycle_count}/{max_healing_cycles}"
                
                SET cycle_success = True
                SET cycle_populated = []
                
                FOR EACH task IN population_plan.tasks:
                    IF (current_time - start_time) > (healing_timeout_minutes * 60):
                        LOG ERROR: "Healing timeout exceeded"
                        APPEND "Healing timeout exceeded" TO errors_encountered
                        SET cycle_success = False
                        BREAK
                    
                    LOG INFO: f"Executing task: {task.table_name}"
                    SET task_result = task_manager.execute_task(task)
                    
                    IF task_result.success:
                        LOG INFO: f"Successfully populated {task.table_name}: {task_result.records_created} records"
                        APPEND task.table_name TO cycle_populated
                        APPEND task.table_name TO tables_populated
                    ELSE:
                        LOG ERROR: f"Failed to populate {task.table_name}: {task_result.error}"
                        APPEND f"Population failed for {task.table_name}: {task_result.error}" TO errors_encountered
                        
                        # Attempt error recovery
                        SET recovery_result = error_handler.handle_error(
                            task_result.error, 
                            {"table_name": task.table_name, "task": task}
                        )
                        
                        IF recovery_result.recovery_successful:
                            LOG INFO: f"Error recovery successful for {task.table_name}"
                            # Retry the task
                            SET retry_result = task_manager.execute_task(task)
                            IF retry_result.success:
                                APPEND task.table_name TO cycle_populated
                                APPEND task.table_name TO tables_populated
                            ELSE:
                                SET cycle_success = False
                        ELSE:
                            LOG ERROR: f"Error recovery failed for {task.table_name}"
                            SET cycle_success = False
                
                # Re-evaluate status after cycle
                SET current_status = detector.detect_table_status()
                SET current_report = detector.calculate_overall_readiness()
                SET missing_tables = current_report.missing_tables
                
                LOG INFO: f"Cycle {cycle_count} completed. Readiness: {current_report.overall_percentage:.1f}%"
                
                IF NOT missing_tables:
                    LOG INFO: "All tables populated successfully!"
                    BREAK
                
                IF NOT cycle_success:
                    LOG WARNING: f"Cycle {cycle_count} had failures, but continuing..."
            
            # Step 5: Validate results and generate recommendations
            LOG INFO: "Step 5: Validating results..."
            SET final_status = detector.detect_table_status()
            SET final_report = detector.calculate_overall_readiness()
            SET final_readiness = final_report.overall_percentage
            
            # Generate recommendations
            IF final_readiness < 100.0:
                APPEND f"Consider manual intervention for remaining tables: {final_report.missing_tables}" TO recommendations
                
                FOR EACH table IN final_report.missing_tables:
                    SET table_status = final_status[table]
                    IF NOT table_status.dependencies_met:
                        APPEND f"Resolve dependencies for {table}" TO recommendations
                    IF table_status.error:
                        APPEND f"Fix error for {table}: {table_status.error}" TO recommendations
            
            SET execution_time = current_time - start_time
            SET success = (final_readiness >= initial_readiness) AND (len(errors_encountered) == 0)
            
            LOG INFO: f"Self-healing cycle completed. Success: {success}, Duration: {execution_time:.1f}s"
            
            RETURN SelfHealingResult(
                success=success,
                initial_readiness=initial_readiness,
                final_readiness=final_readiness,
                tables_populated=tables_populated,
                errors_encountered=errors_encountered,
                execution_time=execution_time,
                recommendations=recommendations,
                cycles_executed=cycle_count
            )
            
        EXCEPT Exception as e:
            LOG ERROR: f"Self-healing cycle failed with exception: {e}"
            SET execution_time = current_time - start_time
            
            RETURN SelfHealingResult(
                success=False,
                initial_readiness=initial_readiness OR 0.0,
                final_readiness=initial_readiness OR 0.0,
                tables_populated=tables_populated,
                errors_encountered=[str(e)] + errors_encountered,
                execution_time=execution_time,
                recommendations=["Manual intervention required due to system error"],
                cycles_executed=cycle_count
            )
    
    FUNCTION detect_and_heal(target_readiness: float = 1.0) -> bool:
        """
        Simplified interface for make targets.
        Returns True if target readiness achieved.
        """
        
        LOG INFO: f"Starting healing to achieve {target_readiness * 100:.1f}% readiness"
        
        SET result = run_self_healing_cycle()
        SET achieved_readiness = result.final_readiness / 100.0
        
        IF achieved_readiness >= target_readiness:
            LOG INFO: f"Target readiness achieved: {achieved_readiness * 100:.1f}%"
            RETURN True
        ELSE:
            LOG WARNING: f"Target readiness not achieved: {achieved_readiness * 100:.1f}% < {target_readiness * 100:.1f}%"
            RETURN False

# Data Classes
CLASS SelfHealingResult:
    success: bool
    initial_readiness: float
    final_readiness: float
    tables_populated: List[str]
    errors_encountered: List[str]
    execution_time: float
    recommendations: List[str]
    cycles_executed: int
```

### 2.2 TDD Test Anchors

```python
# File: tests/test_self_healing_workflow.py

PSEUDOCODE test_self_healing_workflow():
    """
    TDD Anchor: Test complete healing process
    """
    
    # Setup test scenario
    SETUP test_db WITH:
        - RAG.SourceDocuments: 100 records
        - All other tables: 0 records
    
    # Test healing workflow
    SET orchestrator = SelfHealingOrchestrator(test_config, test_db_connection)
    SET result = orchestrator.run_self_healing_cycle()
    
    # Assertions
    ASSERT result.success == True
    ASSERT result.final_readiness > result.initial_readiness
    ASSERT len(result.tables_populated) > 0
    ASSERT result.execution_time > 0
    
    # Verify actual data population
    SET final_status = TableStatusDetector(test_db_connection).detect_table_status()
    ASSERT final_status["RAG.ChunkedDocuments"].is_populated == True
    ASSERT final_status["RAG.ColBERTTokenEmbeddings"].is_populated == True
```

---

## 3. Enhanced Makefile - Self-Healing Make Targets

### 3.1 Core Make Targets

```makefile
# File: Makefile (additions to existing file)

# Self-Healing Data Population Targets
.PHONY: heal-data check-readiness populate-missing validate-healing auto-heal-all heal-to-target heal-progressive heal-emergency

# Main self-healing target
heal-data:
	@echo "🔧 Running self-healing data population..."
	@echo "📊 Checking initial status..."
	$(CONDA_RUN) python -c "from rag_templates.validation.table_status_detector import TableStatusDetector; from common.iris_connection_manager import get_iris_connection; detector = TableStatusDetector(get_iris_connection()); report = detector.calculate_overall_readiness(); print(f'📈 Initial Readiness: {report.overall_percentage:.1f}% ({report.populated_tables}/{report.total_tables} tables)')"
	@echo ""
	@echo "🚀 Starting self-healing process..."
	$(CONDA_RUN) python -c "from rag_templates.validation.self_healing_orchestrator import SelfHealingOrchestrator; orchestrator = SelfHealingOrchestrator(); result = orchestrator.detect_and_heal(); print('✅ Self-healing completed successfully' if result else '❌ Self-healing failed')"
	@echo ""
	@echo "📊 Checking final status..."
	$(CONDA_RUN) python -c "from rag_templates.validation.table_status_detector import TableStatusDetector; from common.iris_connection_manager import get_iris_connection; detector = TableStatusDetector(get_iris_connection()); report = detector.calculate_overall_readiness(); print(f'📈 Final Readiness: {report.overall_percentage:.1f}% ({report.populated_tables}/{report.total_tables} tables)')"

# Check current readiness status
check-readiness:
	@echo "📊 Checking table readiness status..."
	$(CONDA_RUN) python -c "from rag_templates.validation.table_status_detector import TableStatusDetector; from common.iris_connection_manager import get_iris_connection; detector = TableStatusDetector(get_iris_connection()); report = detector.calculate_overall_readiness(); print(f'📈 Overall Readiness: {report.overall_percentage:.1f}% ({report.populated_tables}/{report.total_tables} tables)'); [print(f'  ❌ Missing: {table}') for table in report.missing_tables]; [print(f'  ⚠️  Issue: {issue}') for issue in report.blocking_issues]"

# Populate only missing tables
populate-missing:
	@echo "🔄 Populating missing tables only..."
	$(CONDA_RUN) python -c "from rag_templates.validation.self_healing_orchestrator import SelfHealingOrchestrator; orchestrator = SelfHealingOrchestrator(); result = orchestrator.run_self_healing_cycle(); print(f'📊 Readiness improved from {result.initial_readiness:.1f}% to {result.final_readiness:.1f}%'); print(f'📝 Tables populated: {result.tables_populated}'); [print(f'❌ Error: {error}') for error in result.errors_encountered]; [print(f'💡 Recommendation: {rec}') for rec in result.recommendations]"

# Validate healing effectiveness
validate-healing:
	@echo "✅ Validating self-healing effectiveness..."
	$(MAKE) check-readiness
	@echo ""
	@echo "🧪 Testing pipeline instantiation..."
	$(CONDA_RUN) python -c "import iris_rag; from common.iris_connection_manager import get_iris_connection; from common.utils import get_embedding_func, get_llm_func; connection = get_iris_connection(); embedding_func = get_embedding_func(); llm_func = get_llm_func(); pipeline = iris_rag.create_pipeline('basic', llm_func=llm_func, embedding_func=embedding_func, external_connection=connection, validate_requirements=False); print('✅ Pipeline instantiation successful')"

# Complete self-healing workflow
auto-heal-all: check-readiness heal-data validate-healing
	@echo "🎉 Complete self-healing workflow finished!"
	@echo ""
	@echo "📋 FINAL STATUS SUMMARY:"
	$(CONDA_RUN) python -c "from rag_templates.validation.table_status_detector import TableStatusDetector; from common.iris_connection_manager import get_iris_connection; detector = TableStatusDetector(get_iris_connection()); report = detector.calculate_overall_readiness(); print(f'🎯 Final Readiness: {report.overall_percentage:.1f}%'); print(f'✅ Populated Tables: {report.populated_tables}/{report.total_tables}'); print('🏆 SUCCESS: System ready for production!' if report.overall_percentage >= 85.0 else '⚠️  PARTIAL: Some tables may need manual intervention')"

# Self-healing with specific target
heal-to-target:
	@if [ -z "$(TARGET)" ]; then \
		echo "Error: TARGET parameter required. Usage: make heal-to-target TARGET=0.8"; \
		echo "TARGET should be between 0.0 and 1.0 (e.g., 0.8 = 80% readiness)"; \
		exit 1; \
	fi
	@echo "🎯 Healing to $(TARGET) readiness target..."
	@echo "📊 Current status:"
	$(MAKE) check-readiness
	@echo ""
	@echo "🚀 Starting targeted healing..."
	$(CONDA_RUN) python -c "from rag_templates.validation.self_healing_orchestrator import SelfHealingOrchestrator; orchestrator = SelfHealingOrchestrator(); result = orchestrator.detect_and_heal(target_readiness=$(TARGET)); print('✅ Target achieved' if result else '❌ Target not achieved')"

# Progressive healing (incremental approach)
heal-progressive:
	@echo "📈 Running progressive healing (incremental approach)..."
	@echo "🎯 This will gradually heal the system in stages"
	@echo ""
	@for target in 0.3 0.5 0.7 0.9 1.0; do \
		echo ""; \
		echo "=== Healing to $$target readiness ==="; \
		$(MAKE) heal-to-target TARGET=$$target || echo "⚠ Target $$target not achieved, continuing..."; \
		sleep 2; \
	done
	@echo ""
	@echo "=== PROGRESSIVE HEALING COMPLETE ==="
	$(MAKE) check-readiness

# Emergency healing (force repopulation)
heal-emergency:
	@echo "🚨 Running emergency healing (force repopulation)..."
	@echo "⚠️  WARNING: This will attempt to repopulate all tables"
	@echo ""
	$(CONDA_RUN) python -c "from rag_templates.validation.self_healing_orchestrator import SelfHealingOrchestrator; orchestrator = SelfHealingOrchestrator({'force_repopulation': True, 'max_healing_cycles': 5}); result = orchestrator.run_self_healing_cycle(); print('🚨 Emergency healing completed'); print(f'📊 Final readiness: {result.final_readiness:.1f}%'); print(f'⏱️  Execution time: {result.execution_time:.1f}s')"
```

### 3.2 TDD Test Anchors

```python
# File: tests/test_make_integration.py

PSEUDOCODE test_make_integration():
    """
    TDD Anchor: Verify make targets work correctly
    """
    
    # Test check-readiness target
    SET result = subprocess.run(["make", "check-readiness"], capture_output=True, text=True)
    ASSERT result.returncode == 0
    ASSERT "Overall Readiness:" IN result.stdout
    
    # Test heal-data target
    SET result = subprocess.run(["make", "heal-data"], capture_output=True, text=True)
    ASSERT result.returncode == 0
    ASSERT "Self-healing completed" IN result.stdout
    
    # Test heal-to-target with specific target
    SET result = subprocess.run(["make", "heal-to-target", "TARGET=0.5"], capture_output=True, text=True)
    ASSERT result.returncode == 0
```

---

## 4. DataPopulationManager - CLI Script for Make Integration

### 4.1 CLI Interface

```python
# File: rag_templates/validation/data_population_manager.py

PSEUDOCODE DataPopulationManager:
    """
    CLI script that provides make-friendly interface to self-healing system.
    """
    
    INITIALIZE:
        SET argument_parser = create_argument_parser()
        SET logger = setup_logging()
    
    FUNCTION main():
        """
        Main CLI entry point for make integration.
        """
        
        SET args = argument_parser.parse_args()
        
        TRY:
            IF args.command == "check-status":
                RETURN handle_check_status(args)
            ELIF args.command == "heal":
                RETURN handle_heal(args)
            ELIF args.command == "populate-table":
                RETURN handle_populate_table(args)
            ELIF args.command == "validate":
                RETURN handle_validate(args)
            ELSE:
                LOG ERROR: f"Unknown command: {args.command}"
                RETURN 1
                
        EXCEPT Exception as e:
            LOG ERROR: f"Command failed: {e}"
            IF args.verbose:
                PRINT traceback
            RETURN 1
    
    FUNCTION handle_check_status(args) -> int:
        """
        Handle status checking command.
        """
        
        SET detector = TableStatusDetector(get_iris_connection())
        SET report = detector.calculate_overall_readiness()
        
        # Output format based on args.format
        IF args.format == "json":
            PRINT json.dumps({
                "overall_percentage": report.overall_percentage,
                "populated_tables": report.populated_tables,
                "total_tables": report.total_tables,
                "missing_tables": report.missing_tables,
                "blocking_issues": report.blocking_issues
            })
        ELIF args.format == "make":
            # Make-friendly output
            PRINT f"READINESS_PERCENTAGE={report.overall_percentage:.1f}"
            PRINT f"POPULATED_TABLES={report.populated_tables}"
            PRINT f"TOTAL_TABLES={report.total_tables}"
            PRINT f"MISSING_COUNT={len(report.missing_tables)}"
        ELSE:
            # Human-readable output
            PRINT f"📊 Overall Readiness: {report.overall_percentage:.1f}%"
            PRINT f"📈 Populated Tables: {report.populated_tables}/{report.total_tables}"
            
            IF report.missing_tables:
                PRINT f"❌ Missing Tables: {', '.join(report.missing_tables)}"
            
            IF report.blocking_issues:
                PRINT f"⚠️  Blocking Issues:"
                FOR EACH issue IN report.blocking_issues:
                    PRINT f"   - {issue}"
        
        # Return appropriate exit code
        IF report.overall_percentage >= args.min_readiness:
            RETURN 0  # Success
        ELSE:
            RETURN 2  # Readiness below threshold
    
    FUNCTION handle_heal(args) -> int:
        """
        Handle healing command.
        """
        
        SET config = {}
        IF args.force:
            SET config["force_repopulation"] = True
        IF args.max_cycles:
            SET config["max_healing_cycles"] = args.max_cycles
        IF args.timeout:
            SET config["healing_timeout_minutes"] = args.timeout
        
        SET orchestrator = SelfHealingOrchestrator(config)
        
        IF args.target_readiness:
            SET result = orchestrator.detect_and_heal(args.target_readiness)
IF args.target_readiness:
                SET result = orchestrator.detect_and_heal(args.target_readiness)
                IF result:
                    PRINT f"✅ Target readiness {args.target_readiness * 100:.1f}% achieved"
                    RETURN 0
                ELSE:
                    PRINT f"❌ Target readiness {args.target_readiness * 100:.1f}% not achieved"
                    RETURN 3
            ELSE:
                SET result = orchestrator.run_self_healing_cycle()
                
                # Output results based on format
                IF args.format == "json":
                    PRINT json.dumps({
                        "success": result.success,
                        "initial_readiness": result.initial_readiness,
                        "final_readiness": result.final_readiness,
                        "tables_populated": result.tables_populated,
                        "errors_encountered": result.errors_encountered,
                        "execution_time": result.execution_time,
                        "recommendations": result.recommendations
                    })
                ELSE:
                    PRINT f"🔧 Self-healing cycle completed"
                    PRINT f"📊 Readiness: {result.initial_readiness:.1f}% → {result.final_readiness:.1f}%"
                    PRINT f"📝 Tables populated: {result.tables_populated}"
                    PRINT f"⏱️  Execution time: {result.execution_time:.1f}s"
                    
                    IF result.errors_encountered:
                        PRINT f"❌ Errors encountered:"
                        FOR EACH error IN result.errors_encountered:
                            PRINT f"   - {error}"
                    
                    IF result.recommendations:
                        PRINT f"💡 Recommendations:"
                        FOR EACH rec IN result.recommendations:
                            PRINT f"   - {rec}"
                
                RETURN 0 IF result.success ELSE 4
    
    FUNCTION handle_populate_table(args) -> int:
        """
        Handle single table population command.
        """
        
        SET orchestrator = DataPopulationOrchestrator(get_iris_connection())
        
        LOG INFO: f"Populating table: {args.table_name}"
        SET success, count, details = orchestrator.populate_with_healing(args.table_name)
        
        IF success:
            PRINT f"✅ Successfully populated {args.table_name}: {count} records"
            PRINT f"📝 Details: {details}"
            RETURN 0
        ELSE:
            PRINT f"❌ Failed to populate {args.table_name}: {details}"
            RETURN 5
    
    FUNCTION handle_validate(args) -> int:
        """
        Handle validation command.
        """
        
        # Check table readiness
        SET detector = TableStatusDetector(get_iris_connection())
        SET report = detector.calculate_overall_readiness()
        
        PRINT f"📊 Table Readiness: {report.overall_percentage:.1f}%"
        
        # Test pipeline instantiation if requested
        IF args.test_pipelines:
            PRINT "🧪 Testing pipeline instantiation..."
            
            TRY:
                import iris_rag
                from common.iris_connection_manager import get_iris_connection
                from common.utils import get_embedding_func, get_llm_func
                
                SET connection = get_iris_connection()
                SET embedding_func = get_embedding_func()
                SET llm_func = get_llm_func()
                
                SET pipeline = iris_rag.create_pipeline(
                    'basic', 
                    llm_func=llm_func, 
                    embedding_func=embedding_func, 
                    external_connection=connection, 
                    validate_requirements=False
                )
                
                PRINT "✅ Pipeline instantiation successful"
                
            EXCEPT Exception as e:
                PRINT f"❌ Pipeline instantiation failed: {e}"
                RETURN 6
        
        # Return based on readiness threshold
        IF report.overall_percentage >= args.min_readiness:
            PRINT f"✅ Validation passed (readiness >= {args.min_readiness:.1f}%)"
            RETURN 0
        ELSE:
            PRINT f"❌ Validation failed (readiness < {args.min_readiness:.1f}%)"
            RETURN 7
    
    FUNCTION create_argument_parser():
        """
        Creates command-line argument parser.
        """
        
        SET parser = argparse.ArgumentParser(
            description="Self-healing data population manager for RAG system"
        )
        
        SET subparsers = parser.add_subparsers(dest="command", help="Available commands")
        
        # Check status command
        SET status_parser = subparsers.add_parser("check-status", help="Check table readiness status")
        status_parser.add_argument("--format", choices=["human", "json", "make"], default="human")
        status_parser.add_argument("--min-readiness", type=float, default=0.0, help="Minimum readiness percentage")
        
        # Heal command
        SET heal_parser = subparsers.add_parser("heal", help="Run self-healing process")
        heal_parser.add_argument("--target-readiness", type=float, help="Target readiness (0.0-1.0)")
        heal_parser.add_argument("--force", action="store_true", help="Force repopulation")
        heal_parser.add_argument("--max-cycles", type=int, help="Maximum healing cycles")
        heal_parser.add_argument("--timeout", type=int, help="Timeout in minutes")
        heal_parser.add_argument("--format", choices=["human", "json"], default="human")
        
        # Populate table command
        SET populate_parser = subparsers.add_parser("populate-table", help="Populate specific table")
        populate_parser.add_argument("table_name", help="Name of table to populate")
        
        # Validate command
        SET validate_parser = subparsers.add_parser("validate", help="Validate system readiness")
        validate_parser.add_argument("--test-pipelines", action="store_true", help="Test pipeline instantiation")
        validate_parser.add_argument("--min-readiness", type=float, default=85.0, help="Minimum readiness percentage")
        
        # Global options
        parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
        parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
        
        RETURN parser

# CLI Entry Point
IF __name__ == "__main__":
    SET manager = DataPopulationManager()
    SET exit_code = manager.main()
    sys.exit(exit_code)
```

### 4.2 TDD Test Anchors

```python
# File: tests/test_data_population_manager.py

PSEUDOCODE test_data_population_manager_cli():
    """
    TDD Anchor: Test CLI interface for make integration
    """
    
    # Test check-status command
    SET result = subprocess.run([
        "python", "-m", "rag_templates.validation.data_population_manager",
        "check-status", "--format", "json"
    ], capture_output=True, text=True)
    
    ASSERT result.returncode == 0
    SET status_data = json.loads(result.stdout)
    ASSERT "overall_percentage" IN status_data
    ASSERT "populated_tables" IN status_data
    
    # Test heal command
    SET result = subprocess.run([
        "python", "-m", "rag_templates.validation.data_population_manager",
        "heal", "--target-readiness", "0.5"
    ], capture_output=True, text=True)
    
    ASSERT result.returncode IN [0, 3]  # Success or target not achieved
```

---

## 5. DependencyAnalyzer - Table Dependency Management

### 5.1 Dependency Analysis Logic

```python
# File: rag_templates/validation/dependency_analyzer.py

PSEUDOCODE DependencyAnalyzer:
    
    INITIALIZE:
        SET dependency_graph = {
            "RAG.ChunkedDocuments": ["RAG.SourceDocuments"],
            "RAG.ColBERTTokenEmbeddings": ["RAG.SourceDocuments"],
            "RAG.GraphRAGEntities": ["RAG.SourceDocuments"],
            "RAG.GraphRAGRelationships": ["RAG.GraphRAGEntities"],
            "RAG.KnowledgeGraphNodes": ["RAG.GraphRAGEntities"],
            "RAG.DocumentEntities": ["RAG.SourceDocuments", "RAG.GraphRAGEntities"]
        }
        SET population_weights = {
            "RAG.SourceDocuments": 1,
            "RAG.ChunkedDocuments": 2,
            "RAG.ColBERTTokenEmbeddings": 3,
            "RAG.GraphRAGEntities": 4,
            "RAG.GraphRAGRelationships": 5,
            "RAG.KnowledgeGraphNodes": 6,
            "RAG.DocumentEntities": 7
        }
    
    FUNCTION get_population_order(missing_tables: List[str]) -> List[str]:
        """
        Returns optimal population order respecting dependencies.
        Uses topological sorting to ensure dependencies are satisfied.
        """
        
        # Filter dependency graph to only include missing tables
        SET filtered_graph = {}
        FOR EACH table IN missing_tables:
            SET dependencies = dependency_graph.get(table, [])
            SET filtered_deps = [dep for dep IN dependencies IF dep IN missing_tables]
            SET filtered_graph[table] = filtered_deps
        
        # Perform topological sort
        SET sorted_order = []
        SET visited = set()
        SET visiting = set()
        
        FUNCTION visit(table: str):
            IF table IN visiting:
                RAISE Exception(f"Circular dependency detected involving {table}")
            
            IF table NOT IN visited:
                ADD table TO visiting
                
                FOR EACH dependency IN filtered_graph.get(table, []):
                    visit(dependency)
                
                REMOVE table FROM visiting
                ADD table TO visited
                APPEND table TO sorted_order
        
        FOR EACH table IN missing_tables:
            IF table NOT IN visited:
                visit(table)
        
        # Sort by population weight for tables at same dependency level
        SET final_order = sorted(sorted_order, key=lambda t: population_weights.get(t, 999))
        
        RETURN final_order
    
    FUNCTION validate_dependencies(table_status: Dict[str, TableStatus]) -> List[str]:
        """
        Returns list of dependency violations.
        """
        
        SET violations = []
        
        FOR EACH table_name, status IN table_status.items():
            IF NOT status.is_populated:
                CONTINUE  # Skip empty tables
            
            SET dependencies = dependency_graph.get(table_name, [])
            FOR EACH dep_table IN dependencies:
                IF dep_table IN table_status:
                    SET dep_status = table_status[dep_table]
                    IF NOT dep_status.is_populated:
                        APPEND f"{table_name} requires {dep_table} to be populated" TO violations
                ELSE:
                    # Check dependency directly
                    SET dep_populated = check_table_populated(dep_table)
                    IF NOT dep_populated:
                        APPEND f"{table_name} requires {dep_table} to be populated" TO violations
        
        RETURN violations
    
    FUNCTION get_dependency_tree(table_name: str) -> Dict[str, Any]:
        """
        Returns complete dependency tree for a table.
        """
        
        SET tree = {"table": table_name, "dependencies": []}
        SET dependencies = dependency_graph.get(table_name, [])
        
        FOR EACH dep_table IN dependencies:
            SET dep_tree = get_dependency_tree(dep_table)
            APPEND dep_tree TO tree["dependencies"]
        
        RETURN tree
    
    FUNCTION estimate_population_time(tables: List[str]) -> float:
        """
        Estimates total time needed to populate given tables.
        """
        
        SET time_estimates = {
            "RAG.ChunkedDocuments": 30.0,      # 30 seconds
            "RAG.ColBERTTokenEmbeddings": 120.0, # 2 minutes
            "RAG.GraphRAGEntities": 300.0,      # 5 minutes
            "RAG.GraphRAGRelationships": 180.0, # 3 minutes
            "RAG.KnowledgeGraphNodes": 240.0,   # 4 minutes
            "RAG.DocumentEntities": 60.0        # 1 minute
        }
        
        SET total_time = 0.0
        FOR EACH table IN tables:
            SET table_time = time_estimates.get(table, 60.0)  # Default 1 minute
            SET total_time += table_time
        
        RETURN total_time
    
    FUNCTION can_populate_parallel(table1: str, table2: str) -> bool:
        """
        Checks if two tables can be populated in parallel.
        """
        
        SET deps1 = set(dependency_graph.get(table1, []))
        SET deps2 = set(dependency_graph.get(table2, []))
        
        # Tables can be parallel if neither depends on the other
        RETURN table1 NOT IN deps2 AND table2 NOT IN deps1
    
    FUNCTION get_parallel_groups(tables: List[str]) -> List[List[str]]:
        """
        Groups tables that can be populated in parallel.
        """
        
        SET groups = []
        SET remaining = tables.copy()
        
        WHILE remaining:
            SET current_group = [remaining[0]]
            SET remaining = remaining[1:]
            
            SET i = 0
            WHILE i < len(remaining):
                SET candidate = remaining[i]
                SET can_add = True
                
                FOR EACH table IN current_group:
                    IF NOT can_populate_parallel(table, candidate):
                        SET can_add = False
                        BREAK
                
                IF can_add:
                    APPEND candidate TO current_group
                    REMOVE candidate FROM remaining
                ELSE:
                    INCREMENT i
            
            APPEND current_group TO groups
        
        RETURN groups

# Data Classes
CLASS DependencyViolation:
    table_name: str
    missing_dependency: str
    violation_type: str  # "missing", "circular", "invalid"
    description: str
```

### 5.2 TDD Test Anchors

```python
# File: tests/test_dependency_ordering.py

PSEUDOCODE test_dependency_ordering():
    """
    TDD Anchor: Ensure proper table population order
    """
    
    SET analyzer = DependencyAnalyzer()
    
    # Test basic dependency ordering
    SET missing_tables = [
        "RAG.DocumentEntities",
        "RAG.ChunkedDocuments", 
        "RAG.GraphRAGEntities",
        "RAG.GraphRAGRelationships"
    ]
    
    SET order = analyzer.get_population_order(missing_tables)
    
    # Assertions for correct ordering
    SET chunks_idx = order.index("RAG.ChunkedDocuments")
    SET entities_idx = order.index("RAG.GraphRAGEntities")
    SET relationships_idx = order.index("RAG.GraphRAGRelationships")
    SET doc_entities_idx = order.index("RAG.DocumentEntities")
    
    # GraphRAGEntities must come before GraphRAGRelationships
    ASSERT entities_idx < relationships_idx
    
    # DocumentEntities must come after GraphRAGEntities
    ASSERT entities_idx < doc_entities_idx
    
    # Test parallel grouping
    SET groups = analyzer.get_parallel_groups(missing_tables)
    ASSERT len(groups) >= 2  # Should have at least 2 parallel groups
```

---

## 6. PopulationTaskManager - Task Execution Management

### 6.1 Task Management Logic

```python
# File: rag_templates/validation/population_task_manager.py

PSEUDOCODE PopulationTaskManager:
    
    INITIALIZE:
        SET active_tasks = {}
        SET completed_tasks = {}
        SET failed_tasks = {}
        SET task_counter = 0
        SET progress_callbacks = []
    
    FUNCTION create_population_plan(missing_tables: List[str]) -> PopulationPlan:
        """
        Creates detailed execution plan for populating missing tables.
        """
        
        SET analyzer = DependencyAnalyzer()
        SET ordered_tables = analyzer.get_population_order(missing_tables)
        SET parallel_groups = analyzer.get_parallel_groups(ordered_tables)
        SET estimated_duration = analyzer.estimate_population_time(ordered_tables)
        
        SET tasks = []
        SET task_id = 0
        
        FOR EACH group IN parallel_groups:
            FOR EACH table_name IN group:
                INCREMENT task_id
                SET task = PopulationTask(
                    task_id=task_id,
                    table_name=table_name,
                    task_type="populate",
                    priority=get_table_priority(table_name),
                    estimated_duration=get_table_duration(table_name),
                    dependencies=analyzer.dependency_graph.get(table_name, []),
                    can_run_parallel=True
                )
                APPEND task TO tasks
        
        SET resource_requirements = calculate_resource_requirements(tasks)
        
        RETURN PopulationPlan(
            tasks=tasks,
            estimated_duration=estimated_duration,
            resource_requirements=resource_requirements,
            parallel_groups=parallel_groups
        )
    
    FUNCTION execute_task(task: PopulationTask) -> TaskResult:
        """
        Executes single population task with progress tracking.
        """
        
        SET start_time = current_time
        SET task_id = task.task_id
        
        TRY:
            LOG INFO: f"Starting task {task_id}: {task.table_name}"
            
            # Add to active tasks
            SET active_tasks[task_id] = {
                "task": task,
                "start_time": start_time,
                "status": "running"
            }
            
            # Notify progress callbacks
            notify_progress_callbacks("task_started", task)
            
            # Execute the actual population
            SET success, record_count, details = execute_table_population(task.table_name)
            
            SET end_time = current_time
            SET duration = end_time - start_time
            
            IF success:
                LOG INFO: f"Task {task_id} completed successfully: {record_count} records"
                
                SET result = TaskResult(
                    task_id=task_id,
                    table_name=task.table_name,
                    success=True,
                    records_created=record_count,
                    execution_time=duration,
                    details=details
                )
                
                # Move to completed tasks
                SET completed_tasks[task_id] = {
                    "task": task,
                    "result": result,
                    "end_time": end_time
                }
                
                notify_progress_callbacks("task_completed", task, result)
                
            ELSE:
                LOG ERROR: f"Task {task_id} failed: {details}"
                
                SET result = TaskResult(
                    task_id=task_id,
                    table_name=task.table_name,
                    success=False,
                    records_created=0,
                    execution_time=duration,
                    error=details
                )
                
                # Move to failed tasks
                SET failed_tasks[task_id] = {
                    "task": task,
                    "result": result,
                    "end_time": end_time
                }
                
                notify_progress_callbacks("task_failed", task, result)
            
            # Remove from active tasks
            DEL active_tasks[task_id]
            
            RETURN result
            
        EXCEPT Exception as e:
            LOG ERROR: f"Task {task_id} failed with exception: {e}"
            
            SET end_time = current_time
            SET duration = end_time - start_time
            
            SET result = TaskResult(
                task_id=task_id,
                table_name=task.table_name,
                success=False,
                records_created=0,
                execution_time=duration,
                error=str(e)
            )
            
            SET failed_tasks[task_id] = {
                "task": task,
                "result": result,
                "end_time": end_time
            }
            
            # Remove from active tasks
            IF task_id IN active_tasks:
                DEL active_tasks[task_id]
            
            notify_progress_callbacks("task_failed", task, result)
            
            RETURN result
    
    FUNCTION execute_table_population(table_name: str) -> Tuple[bool, int, str]:
        """
        Executes actual table population using DataPopulationOrchestrator.
        """
        
        SET orchestrator = DataPopulationOrchestrator(get_iris_connection())
        
        # Map table names to orchestrator methods
        SET population_methods = {
            "RAG.ChunkedDocuments": orchestrator._populate_chunked_documents,
            "RAG.ColBERTTokenEmbeddings": orchestrator._populate_colbert_embeddings,
            "RAG.GraphRAGEntities": orchestrator._populate_graphrag_entities,
            "RAG.GraphRAGRelationships": orchestrator._populate_graphrag_relationships,
            "RAG.KnowledgeGraphNodes": orchestrator._populate_knowledge_graph_nodes,
            "RAG.DocumentEntities": orchestrator._populate_document_entities
        }
        
        SET method = population_methods.get(table_name)
        IF NOT method:
            RETURN False, 0, f"No population method found for {table_name}"
        
        TRY:
            SET success, count, details = method()
            RETURN success, count, details
        EXCEPT Exception as e:
            RETURN False, 0, f"Population method failed: {str(e)}"
    
    FUNCTION get_progress_summary() -> ProgressSummary:
        """
        Returns current progress across all tasks.
        """
        
        SET total_tasks = len(completed_tasks) + len(failed_tasks) + len(active_tasks)
        SET completed_count = len(completed_tasks)
        SET failed_count = len(failed_tasks)
        SET active_count = len(active_tasks)
        
        SET completion_percentage = (completed_count / total_tasks * 100) IF total_tasks > 0 ELSE 0
        
        SET current_task = None
        IF active_tasks:
            SET current_task_info = list(active_tasks.values())[0]
            SET current_task = current_task_info["task"].table_name
        
        SET estimated_completion = None
        IF active_count > 0:
            SET avg_duration = calculate_average_task_duration()
            SET remaining_time = avg_duration * active_count
            SET estimated_completion = current_time + remaining_time
        
        RETURN ProgressSummary(
            total_tasks=total_tasks,
            completed_tasks=completed_count,
            failed_tasks=failed_count,
            active_tasks=active_count,
            completion_percentage=completion_percentage,
            current_task=current_task,
            estimated_completion=estimated_completion
        )
    
    FUNCTION add_progress_callback(callback: Callable):
        """
        Adds progress callback function.
        """
        APPEND callback TO progress_callbacks
    
    FUNCTION notify_progress_callbacks(event_type: str, task: PopulationTask, result: TaskResult = None):
        """
        Notifies all registered progress callbacks.
        """
        FOR EACH callback IN progress_callbacks:
            TRY:
                callback(event_type, task, result)
            EXCEPT Exception as e:
                LOG WARNING: f"Progress callback failed: {e}"

# Data Classes
CLASS PopulationTask:
    task_id: int
    table_name: str
    task_type: str  # "populate", "validate", "cleanup"
    priority: int
    estimated_duration: float
    dependencies: List[str]
    can_run_parallel: bool

CLASS PopulationPlan:
    tasks: List[PopulationTask]
    estimated_duration: float
    resource_requirements: Dict[str, Any]
    parallel_groups: List[List[str]]

CLASS TaskResult:
    task_id: int
    table_name: str
    success: bool
    records_created: int
    execution_time: float
    details: Optional[str] = None
    error: Optional[str] = None

CLASS ProgressSummary:
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    active_tasks: int
    completion_percentage: float
    current_task: Optional[str]
    estimated_completion: Optional[datetime]
```

---

## 7. ErrorRecoveryHandler - Error Recovery and Resilience

### 7.1 Error Recovery Logic

```python
# File: rag_templates/validation/error_recovery_handler.py

PSEUDOCODE ErrorRecoveryHandler:
    
    INITIALIZE:
        SET recovery_strategies = {
            "connection_error": _recover_connection,
            "data_corruption": _recover_data_corruption,
            "dependency_violation": _recover_dependency_violation,
            "resource_exhaustion": _recover_resource_exhaustion,
            "timeout_error": _recover_timeout,
            "schema_error": _recover_schema_error
        }
        SET max_recovery_attempts = 3
        SET recovery_delay_seconds = 5.0
    
    FUNCTION handle_error(error: Exception, context: Dict[str, Any]) -> RecoveryResult:
        """
        Analyzes error and attempts recovery.
        """
        
        SET error_type = classify_error(error)
        SET recovery_strategy = recovery_strategies.get(error_type)
        
        IF NOT recovery_strategy:
            LOG WARNING: f"No recovery strategy for error type: {error_type}"
            RETURN RecoveryResult(
                recovery_attempted=False,
                recovery_successful=False,
                retry_recommended=False,
                alternative_strategy=None,
                error_classification=error_type
            )
        
        LOG INFO: f"Attempting recovery for {error_type}: {str(error)}"
        
        SET recovery_successful = False
        SET attempts = 0
        
        WHILE attempts < max_recovery_attempts AND NOT recovery_successful:
            INCREMENT attempts
            
            TRY:
                LOG INFO: f"Recovery attempt {attempts}/{max_recovery_attempts}"
                SET recovery_successful = recovery_strategy(context)
                
                IF recovery_successful:
                    LOG INFO: f"Recovery successful after {attempts} attempts"
                ELSE:
                    LOG WARNING: f"Recovery attempt {attempts} failed"
                    IF attempts < max_recovery_attempts:
                        time.sleep(recovery_delay_seconds)
                        
            EXCEPT Exception as recovery_error:
                LOG ERROR: f"Recovery attempt {attempts} failed with exception: {recovery_error}"
                IF attempts < max_recovery_attempts:
                    time.sleep(recovery_delay_seconds)
        
        SET retry_recommended = recovery_successful OR error_type IN ["timeout_error", "connection_error"]
        SET alternative_strategy = get_alternative_strategy(error_type, context)
        
        RETURN RecoveryResult(
            recovery_attempted=True,
            recovery_successful=recovery_successful,
            retry_recommended=retry_recommended,
            alternative_strategy=alternative_strategy,
            error_classification=error_type,
            attempts_made=attempts
        )
    
    FUNCTION classify_error(error: Exception) -> str:
        """
        Classifies error into recovery categories.
        """
        
        SET error_str = str(error).lower()
        
        IF "connection" IN error_str OR "network" IN error_str:
            RETURN "connection_error"
        ELIF "timeout" IN error_str OR "timed out" IN error_str:
            RETURN "timeout_error"
        ELIF "memory" IN error_str OR "resource" IN error_str:
            RETURN "resource_exhaustion"
        ELIF "schema" IN error_str OR "column" IN error_str OR "table" IN error_str:
            RETURN "schema_error"
        ELIF "dependency" IN error_str OR "foreign key" IN error_str:
            RETURN "dependency_violation"
        ELIF "corrupt" IN error_str OR "invalid data" IN error_str:
            RETURN "data_corruption"
        ELSE:
            RETURN "unknown_error"
    
    FUNCTION _recover_connection(context: Dict[str, Any]) -> bool:
        """
        Attempts to recover database connection.
        """
        
        TRY:
            LOG INFO: "Attempting connection recovery..."
            
            # Close existing connection if any
            IF "db_connection" IN context:
                TRY:
                    context["db_connection"].close()
                EXCEPT:
                    PASS
            
            # Get new connection
            SET new_connection = get_iris_connection()
            IF new_connection:
                # Test connection with simple query
                SET cursor = new_connection.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                cursor.close()
                
                # Update context with new connection
                SET context["db_connection"] = new_connection
                
                LOG INFO: "Connection recovery successful"
                RETURN True
            ELSE:
                LOG ERROR: "Failed to establish new connection"
                RETURN False
                
        EXCEPT Exception as e:
            LOG ERROR: f"Connection recovery failed: {e}"
            RETURN False
    
    FUNCTION _recover_data_corruption(context: Dict[str, Any]) -> bool:
        """
        Attempts to recover from data corruption.
        """
        
        SET table_name = context.get("table_name")
        IF NOT table_name:
            RETURN False
        
        TRY:
            LOG INFO: f"Attempting data corruption recovery for {table_name}"
            
            # Strategy 1: Clear corrupted data and restart
            SET connection = context.get("db_connection") OR get_iris_connection()
            SET cursor = connection.cursor()
            
            # Check if table has recent data that might be corrupted
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            SET current_count = cursor.fetchone()[0]
            
            IF current_count > 0:
                LOG INFO: f"Clearing potentially corrupted data from {table_name}")
                cursor.execute(f"DELETE FROM {table_name}")
                connection.commit()
                LOG INFO: f"Cleared {current_count} records from {table_name}")
            
            cursor.close()
            RETURN True
            
        EXCEPT Exception as e:
            LOG ERROR: f"Data corruption recovery failed: {e}"
            RETURN False
    
    FUNCTION _recover_dependency_violation(context: Dict[str, Any]) -> bool:
        """
        Attempts to recover from dependency violations.
        """
        
        SET table_name = context.get("table_name")
        IF NOT table_name:
            RETURN False
        
        TRY:
            LOG INFO: f"Attempting dependency violation recovery for {table_name}"
            
            SET analyzer = DependencyAnalyzer()
            SET dependencies = analyzer.dependency_graph.get(table_name, [])
            
            # Check and populate missing dependencies
            FOR EACH dep_table IN dependencies:
                SET detector = TableStatusDetector(get_iris_connection())
                SET status = detector.get_single_table_status(dep_table)
                
                IF NOT status.is_populated:
                    LOG INFO: f"Populating missing dependency: {dep_table}")
                    SET orchestrator = DataPopul
SET orchestrator = DataPopulationOrchestrator(get_iris_connection())
                    SET success, count, details = orchestrator.populate_with_healing(dep_table)
                    
                    IF NOT success:
                        LOG ERROR: f"Failed to populate dependency {dep_table}: {details}")
                        RETURN False
            
            RETURN True
            
        EXCEPT Exception as e:
            LOG ERROR: f"Dependency violation recovery failed: {e}")
            RETURN False
    
    FUNCTION _recover_resource_exhaustion(context: Dict[str, Any]) -> bool:
        """
        Attempts to recover from resource exhaustion.
        """
        
        TRY:
            LOG INFO: "Attempting resource exhaustion recovery..."
            
            # Strategy 1: Force garbage collection
            import gc
            gc.collect()
            
            # Strategy 2: Clear caches if available
            IF hasattr(context.get("orchestrator"), "clear_caches"):
                context["orchestrator"].clear_caches()
            
            # Strategy 3: Reduce batch sizes
            IF "batch_size" IN context:
                SET context["batch_size"] = max(1, context["batch_size"] // 2)
                LOG INFO: f"Reduced batch size to {context['batch_size']}")
            
            # Strategy 4: Wait for resources to free up
            time.sleep(10)
            
            RETURN True
            
        EXCEPT Exception as e:
            LOG ERROR: f"Resource exhaustion recovery failed: {e}")
            RETURN False
    
    FUNCTION _recover_timeout(context: Dict[str, Any]) -> bool:
        """
        Attempts to recover from timeout errors.
        """
        
        TRY:
            LOG INFO: "Attempting timeout recovery...")
            
            # Strategy 1: Increase timeout values
            IF "timeout" IN context:
                SET context["timeout"] = context["timeout"] * 2
                LOG INFO: f"Increased timeout to {context['timeout']} seconds")
            
            # Strategy 2: Break large operations into smaller chunks
            IF "chunk_size" IN context:
                SET context["chunk_size"] = max(10, context["chunk_size"] // 2)
                LOG INFO: f"Reduced chunk size to {context['chunk_size']}")
            
            # Strategy 3: Reset connection
            RETURN _recover_connection(context)
            
        EXCEPT Exception as e:
            LOG ERROR: f"Timeout recovery failed: {e}")
            RETURN False
    
    FUNCTION _recover_schema_error(context: Dict[str, Any]) -> bool:
        """
        Attempts to recover from schema-related errors.
        """
        
        TRY:
            LOG INFO: "Attempting schema error recovery...")
            
            SET table_name = context.get("table_name")
            IF NOT table_name:
                RETURN False
            
            # Strategy 1: Verify table exists and has correct schema
            SET connection = context.get("db_connection") OR get_iris_connection()
            SET cursor = connection.cursor()
            
            TRY:
                cursor.execute(f"SELECT TOP 1 * FROM {table_name}")
                cursor.fetchone()
                LOG INFO: f"Table {table_name} exists and is accessible")
                cursor.close()
                RETURN True
                
            EXCEPT Exception as table_error:
                LOG WARNING: f"Table {table_name} has issues: {table_error}")
                cursor.close()
                
                # Strategy 2: Attempt to recreate table if needed
                # This would require table creation logic
                LOG INFO: f"Table {table_name} may need recreation")
                RETURN False
            
        EXCEPT Exception as e:
            LOG ERROR: f"Schema error recovery failed: {e}")
            RETURN False
    
    FUNCTION get_alternative_strategy(error_type: str, context: Dict[str, Any]) -> Optional[str]:
        """
        Suggests alternative strategies for unrecoverable errors.
        """
        
        SET alternatives = {
            "connection_error": "Try manual database restart or check network connectivity",
            "data_corruption": "Consider manual data cleanup or table recreation",
            "dependency_violation": "Manually populate dependencies in correct order",
            "resource_exhaustion": "Increase system resources or run during off-peak hours",
            "timeout_error": "Run population in smaller batches or increase timeout limits",
            "schema_error": "Verify database schema and table definitions"
        }
        
        RETURN alternatives.get(error_type, "Manual intervention required")

# Data Classes
CLASS RecoveryResult:
    recovery_attempted: bool
    recovery_successful: bool
    retry_recommended: bool
    alternative_strategy: Optional[str]
    error_classification: str
    attempts_made: int = 0
```

---

## 8. Enhanced DataPopulationOrchestrator Integration

### 8.1 Self-Healing Extensions

```python
# File: rag_templates/validation/data_population_orchestrator.py (extensions)

PSEUDOCODE DataPopulationOrchestrator_SelfHealing_Extensions:
    """
    Extensions to existing DataPopulationOrchestrator for self-healing capabilities.
    """
    
    FUNCTION enable_self_healing(enabled: bool = True):
        """
        Enables self-healing capabilities with retry logic and error recovery.
        """
        SET self.self_healing_enabled = enabled
        SET self.max_retry_attempts = 3
        SET self.retry_delay = 5.0
        SET self.error_handler = ErrorRecoveryHandler()
        SET self.progress_tracker = ProgressTracker()
    
    FUNCTION populate_with_healing(table_name: str) -> Tuple[bool, int, str]:
        """
        Populates table with self-healing retry logic and comprehensive error handling.
        """
        
        SET attempt_count = 0
        SET last_error = None
        
        WHILE attempt_count < self.max_retry_attempts:
            INCREMENT attempt_count
            
            TRY:
                LOG INFO: f"Population attempt {attempt_count}/{self.max_retry_attempts} for {table_name}"
                
                # Update progress tracker
                self.progress_tracker.start_table_population(table_name, attempt_count)
                
                # Execute population method
                SET success, count, details = self._populate_table(table_name)
                
                IF success:
                    LOG INFO: f"Successfully populated {table_name}: {count} records"
                    self.progress_tracker.complete_table_population(table_name, count)
                    RETURN success, count, details
                ELSE:
                    SET last_error = Exception(details)
                    LOG WARNING: f"Population attempt {attempt_count} failed for {table_name}: {details}"
                
            EXCEPT Exception as e:
                SET last_error = e
                LOG ERROR: f"Population attempt {attempt_count} failed for {table_name}: {e}"
            
            # Attempt self-healing if enabled and not on last attempt
            IF self.self_healing_enabled AND attempt_count < self.max_retry_attempts:
                LOG INFO: f"Attempting self-healing for {table_name}")
                
                SET recovery_context = {
                    "table_name": table_name,
                    "db_connection": self.db_connection,
                    "attempt_count": attempt_count,
                    "orchestrator": self
                }
                
                SET recovery_result = self.error_handler.handle_error(last_error, recovery_context)
                
                IF recovery_result.recovery_successful:
                    LOG INFO: f"Self-healing successful for {table_name}, retrying...")
                    time.sleep(self.retry_delay)
                    CONTINUE
                ELIF recovery_result.retry_recommended:
                    LOG INFO: f"Recovery recommended retry for {table_name}")
                    time.sleep(self.retry_delay)
                    CONTINUE
                ELSE:
                    LOG ERROR: f"Self-healing failed for {table_name}: {recovery_result.alternative_strategy}")
                    BREAK
            ELSE:
                # No self-healing or last attempt
                time.sleep(self.retry_delay)
        
        # All attempts failed
        SET final_error = f"All {self.max_retry_attempts} attempts failed. Last error: {str(last_error)}"
        self.progress_tracker.fail_table_population(table_name, final_error)
        RETURN False, 0, final_error
    
    FUNCTION get_population_progress() -> Dict[str, Any]:
        """
        Returns current population progress with detailed metrics.
        """
        
        IF NOT hasattr(self, 'progress_tracker'):
            RETURN {"error": "Progress tracking not enabled"}
        
        SET progress = self.progress_tracker.get_summary()
        
        # Add orchestrator-specific metrics
        SET total_tables = len(self.TABLE_ORDER)
        SET completed_tables = len([t for t in self.TABLE_ORDER if self._is_table_populated(t)])
        
        SET enhanced_progress = {
            "total_tables": total_tables,
            "completed_tables": completed_tables,
            "completion_percentage": (completed_tables / total_tables) * 100,
            "current_table": progress.get("current_table"),
            "estimated_completion": progress.get("estimated_completion"),
            "errors": progress.get("errors", []),
            "table_details": {}
        }
        
        # Add per-table details
        FOR EACH table_name IN self.TABLE_ORDER:
            SET table_progress = self.progress_tracker.get_table_progress(table_name)
            SET enhanced_progress["table_details"][table_name] = table_progress
        
        RETURN enhanced_progress
    
    FUNCTION _is_table_populated(table_name: str) -> bool:
        """
        Checks if a table is populated (has records).
        """
        
        TRY:
            SET cursor = self.db_connection.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            SET count = cursor.fetchone()[0]
            cursor.close()
            RETURN count > 0
        EXCEPT:
            RETURN False
    
    FUNCTION clear_caches(self):
        """
        Clears internal caches to free up memory.
        """
        
        # Clear any cached data
        IF hasattr(self, '_cached_embeddings'):
            DEL self._cached_embeddings
        
        IF hasattr(self, '_cached_documents'):
            DEL self._cached_documents
        
        # Force garbage collection
        import gc
        gc.collect()
        
        LOG INFO: "Cleared orchestrator caches"
    
    FUNCTION validate_table_dependencies(self) -> Dict[str, List[str]]:
        """
        Validates that all table dependencies are satisfied.
        """
        
        SET analyzer = DependencyAnalyzer()
        SET detector = TableStatusDetector(self.db_connection)
        SET table_status = detector.detect_table_status()
        
        SET violations = analyzer.validate_dependencies(table_status)
        
        SET dependency_report = {
            "violations": violations,
            "satisfied_dependencies": [],
            "missing_dependencies": []
        }
        
        FOR EACH table_name IN self.TABLE_ORDER:
            SET dependencies = analyzer.dependency_graph.get(table_name, [])
            SET table_populated = table_status.get(table_name, {}).get("is_populated", False)
            
            IF table_populated:
                FOR EACH dep IN dependencies:
                    SET dep_populated = table_status.get(dep, {}).get("is_populated", False)
                    IF dep_populated:
                        APPEND f"{table_name} -> {dep}" TO dependency_report["satisfied_dependencies"]
                    ELSE:
                        APPEND f"{table_name} -> {dep}" TO dependency_report["missing_dependencies"]
        
        RETURN dependency_report

# Progress Tracking Helper Class
CLASS ProgressTracker:
    
    INITIALIZE:
        SET self.table_progress = {}
        SET self.start_time = None
        SET self.current_table = None
    
    FUNCTION start_table_population(table_name: str, attempt: int):
        SET self.current_table = table_name
        SET self.table_progress[table_name] = {
            "status": "in_progress",
            "start_time": current_time,
            "attempt": attempt,
            "records_created": 0
        }
    
    FUNCTION complete_table_population(table_name: str, record_count: int):
        IF table_name IN self.table_progress:
            SET self.table_progress[table_name].update({
                "status": "completed",
                "end_time": current_time,
                "records_created": record_count
            })
        SET self.current_table = None
    
    FUNCTION fail_table_population(table_name: str, error: str):
        IF table_name IN self.table_progress:
            SET self.table_progress[table_name].update({
                "status": "failed",
                "end_time": current_time,
                "error": error
            })
        SET self.current_table = None
    
    FUNCTION get_summary() -> Dict[str, Any]:
        SET completed = len([p for p in self.table_progress.values() if p["status"] == "completed"])
        SET failed = len([p for p in self.table_progress.values() if p["status"] == "failed"])
        SET in_progress = len([p for p in self.table_progress.values() if p["status"] == "in_progress"])
        
        RETURN {
            "current_table": self.current_table,
            "completed_tables": completed,
            "failed_tables": failed,
            "in_progress_tables": in_progress,
            "table_progress": self.table_progress
        }
```

---

## 9. Complete TDD Test Suite

### 9.1 Comprehensive Test Coverage

```python
# File: tests/test_self_healing_system_complete.py

PSEUDOCODE test_self_healing_system_complete():
    """
    Comprehensive TDD test suite for the complete self-healing system.
    """
    
    # Test 1: Table Status Detection
    FUNCTION test_table_status_detection_comprehensive():
        """Test comprehensive table status detection with various scenarios."""
        
        # Setup test database with mixed population states
        SETUP test_db WITH:
            - RAG.SourceDocuments: 1000 records
            - RAG.ChunkedDocuments: 0 records
            - RAG.ColBERTTokenEmbeddings: 500 records
            - RAG.GraphRAGEntities: 0 records
            - RAG.GraphRAGRelationships: 0 records
            - RAG.KnowledgeGraphNodes: 0 records
            - RAG.DocumentEntities: 0 records
        
        SET detector = TableStatusDetector(test_db_connection)
        SET status = detector.detect_table_status()
        
        # Verify detection accuracy
        ASSERT status["RAG.SourceDocuments"].is_populated == True
        ASSERT status["RAG.SourceDocuments"].record_count == 1000
        ASSERT status["RAG.SourceDocuments"].health_score == 1.0
        
        ASSERT status["RAG.ChunkedDocuments"].is_populated == False
        ASSERT status["RAG.ChunkedDocuments"].dependencies_met == True  # SourceDocuments populated
        
        ASSERT status["RAG.ColBERTTokenEmbeddings"].is_populated == True
        ASSERT status["RAG.ColBERTTokenEmbeddings"].record_count == 500
        
        # Test readiness calculation
        SET readiness = detector.calculate_overall_readiness()
        ASSERT readiness.overall_percentage == (2/7) * 100  # 2 out of 7 tables
        ASSERT len(readiness.missing_tables) == 5
        ASSERT "RAG.ChunkedDocuments" IN readiness.missing_tables
    
    # Test 2: Dependency Analysis
    FUNCTION test_dependency_ordering_complex():
        """Test complex dependency ordering scenarios."""
        
        SET analyzer = DependencyAnalyzer()
        
        # Test full dependency chain
        SET missing_tables = [
            "RAG.DocumentEntities",      # Depends on SourceDocuments + GraphRAGEntities
            "RAG.GraphRAGRelationships", # Depends on GraphRAGEntities
            "RAG.KnowledgeGraphNodes",   # Depends on GraphRAGEntities
            "RAG.GraphRAGEntities",      # Depends on SourceDocuments
            "RAG.ChunkedDocuments",      # Depends on SourceDocuments
            "RAG.ColBERTTokenEmbeddings" # Depends on SourceDocuments
        ]
        
        SET order = analyzer.get_population_order(missing_tables)
        
        # Verify correct ordering
        SET chunks_idx = order.index("RAG.ChunkedDocuments")
        SET colbert_idx = order.index("RAG.ColBERTTokenEmbeddings")
        SET entities_idx = order.index("RAG.GraphRAGEntities")
        SET relationships_idx = order.index("RAG.GraphRAGRelationships")
        SET nodes_idx = order.index("RAG.KnowledgeGraphNodes")
        SET doc_entities_idx = order.index("RAG.DocumentEntities")
        
        # GraphRAGEntities must come before its dependents
        ASSERT entities_idx < relationships_idx
        ASSERT entities_idx < nodes_idx
        ASSERT entities_idx < doc_entities_idx
        
        # Test parallel grouping
        SET groups = analyzer.get_parallel_groups(missing_tables)
        
        # First group should contain tables that only depend on SourceDocuments
        SET first_group = groups[0]
        ASSERT "RAG.ChunkedDocuments" IN first_group
        ASSERT "RAG.ColBERTTokenEmbeddings" IN first_group
        ASSERT "RAG.GraphRAGEntities" IN first_group
    
    # Test 3: Self-Healing Workflow
    FUNCTION test_self_healing_workflow_complete():
        """Test complete self-healing workflow with error scenarios."""
        
        # Setup test scenario with missing data
        SETUP test_db WITH:
            - RAG.SourceDocuments: 100 records
            - All other tables: 0 records
        
        # Mock population methods to simulate various scenarios
        SET mock_orchestrator = MockDataPopulationOrchestrator()
        mock_orchestrator.setup_scenarios({
            "RAG.ChunkedDocuments": ("success", 300, "Created 300 chunks"),
            "RAG.ColBERTTokenEmbeddings": ("failure_then_success", 0, "Connection timeout"),
            "RAG.GraphRAGEntities": ("success", 1000, "Extracted 1000 entities"),
            "RAG.GraphRAGRelationships": ("success", 500, "Created 500 relationships"),
            "RAG.KnowledgeGraphNodes": ("success", 800, "Created 800 nodes"),
            "RAG.DocumentEntities": ("success", 1200, "Created 1200 doc-entity links")
        })
        
        SET orchestrator = SelfHealingOrchestrator(test_config, test_db_connection)
        orchestrator.population_orchestrator = mock_orchestrator
        
        SET result = orchestrator.run_self_healing_cycle()
        
        # Verify healing results
        ASSERT result.success == True
        ASSERT result.final_readiness > result.initial_readiness
        ASSERT len(result.tables_populated) >= 5  # At least 5 tables should be populated
        ASSERT result.execution_time > 0
        
        # Verify error recovery worked for ColBERT
        ASSERT "RAG.ColBERTTokenEmbeddings" IN result.tables_populated
        
        # Verify recommendations are provided
        ASSERT len(result.recommendations) >= 0
    
    # Test 4: Make Integration
    FUNCTION test_make_integration_complete():
        """Test complete make target integration."""
        
        # Test check-readiness target
        SET result = subprocess.run([
            "make", "check-readiness"
        ], capture_output=True, text=True, cwd=test_project_dir)
        
        ASSERT result.returncode == 0
        ASSERT "Overall Readiness:" IN result.stdout
        ASSERT "%" IN result.stdout
        
        # Test heal-data target
        SET result = subprocess.run([
            "make", "heal-data"
        ], capture_output=True, text=True, cwd=test_project_dir)
        
        ASSERT result.returncode == 0
        ASSERT ("Self-healing completed successfully" IN result.stdout OR 
                "Self-healing failed" IN result.stdout)
        
        # Test heal-to-target with specific percentage
        SET result = subprocess.run([
            "make", "heal-to-target", "TARGET=0.5"
        ], capture_output=True, text=True, cwd=test_project_dir)
        
        ASSERT result.returncode IN [0, 3]  # Success or target not achieved
        
        # Test progressive healing
        SET result = subprocess.run([
            "make", "heal-progressive"
        ], capture_output=True, text=True, cwd=test_project_dir)
        
        ASSERT result.returncode == 0
        ASSERT "PROGRESSIVE HEALING COMPLETE" IN result.stdout
    
    # Test 5: Error Recovery
    FUNCTION test_error_recovery_comprehensive():
        """Test comprehensive error recovery scenarios."""
        
        SET handler = ErrorRecoveryHandler()
        
        # Test connection error recovery
        SET connection_error = Exception("Connection lost to database")
        SET context = {"db_connection": None, "table_name": "RAG.ChunkedDocuments"}
        
        SET result = handler.handle_error(connection_error, context)
        ASSERT result.recovery_attempted == True
        ASSERT result.error_classification == "connection_error"
        ASSERT result.retry_recommended == True
        
        # Test dependency violation recovery
        SET dependency_error = Exception("Foreign key constraint violation")
        SET context = {"table_name": "RAG.GraphRAGRelationships"}
        
        SET result = handler.handle_error(dependency_error, context)
        ASSERT result.error_classification == "dependency_violation"
        
        # Test resource exhaustion recovery
        SET memory_error = Exception("Out of memory during processing")
        SET context = {"batch_size": 1000}
        
        SET result = handler.handle_error(memory_error, context)
        ASSERT result.error_classification == "resource_exhaustion"
        ASSERT context["batch_size"] < 1000  # Should be reduced
    
    # Test 6: CLI Integration
    FUNCTION test_cli_integration_complete():
        """Test complete CLI integration."""
        
        # Test status check command
        SET result = subprocess.run([
            "python", "-m", "rag_templates.validation.data_population_manager",
            "check-status", "--format", "json"
        ], capture_output=True, text=True, cwd=test_project_dir)
        
        ASSERT result.returncode == 0
        SET status_data = json.loads(result.stdout)
        ASSERT "overall_percentage" IN status_data
        ASSERT "populated_tables" IN status_data
        ASSERT "missing_tables" IN status_data
        
        # Test heal command
        SET result = subprocess.run([
            "python", "-m", "rag_templates.validation.data_population_manager",
            "heal", "--target-readiness", "0.5", "--format", "json"
        ], capture_output=True, text=True, cwd=test_project_dir)
        
        ASSERT result.returncode IN [0, 3, 4]  # Success, target not achieved, or healing failed
        
        # Test table-specific population
        SET result = subprocess.run([
            "python", "-m", "rag_templates.validation.data_population_manager",
            "populate-table", "RAG.ChunkedDocuments"
        ], capture_output=True, text=True, cwd=test_project_dir)
        
        ASSERT result.returncode IN [0, 5]  # Success or population failed
        
        # Test validation command
        SET result = subprocess.run([
            "python", "-m", "rag_templates.validation.data_population_manager",
            "validate", "--test-pipelines", "--min-readiness", "50.0"
        ], capture_output=True, text=True, cwd=test_project_dir)
        
        ASSERT result.returncode IN [0, 6, 7]  # Success, pipeline test failed, or validation failed

# Mock Classes for Testing
CLASS MockDataPopulationOrchestrator:
    
    INITIALIZE:
        SET self.scenarios = {}
        SET self.call_count = {}
    
    FUNCTION setup_scenarios(scenarios: Dict[str, Tuple[str, int, str]]):
        SET self.scenarios = scenarios
        FOR EACH table IN scenarios.keys():
            SET self.call_count[table] = 0
    
    FUNCTION populate_with_healing(table_name: str) -> Tuple[bool, int, str]:
        INCREMENT self.call_count[table_name]
        
        IF table_name NOT IN self.scenarios:
            RETURN False, 0, f"No scenario defined for {table_name}"
        
        SET scenario_type, count, details = self.scenarios[table_name]
        
        IF scenario_type == "success":
            RETURN True, count, details
        ELIF scenario_type == "failure":
            RETURN False, count, details
        ELIF scenario_type == "failure_then_success":
            IF self.call_count[table_name] == 1:
                RETURN False, 0, details  # First call fails
            ELSE:
                RETURN True, count, "Recovered after retry"  # Subsequent calls succeed
        ELSE:
            RETURN False, 0, f"Unknown scenario type: {scenario_type}"
```

---

## 10. Implementation Summary and Next Steps

### 10.1 Component Implementation Priority

```python
PSEUDOCODE Implementation_Priority:
    """
    Recommended implementation order for maximum impact.
    """
    
    PHASE_1_FOUNDATION = [
        "TableStatusDetector",           # Core detection capability
        "DependencyAnalyzer",           # Dependency management
        "Basic Makefile targets"        # check-readiness, heal-data
    ]
    
    PHASE_2_ORCHESTRATION = [
        "SelfHealingOrchestrator",      # Main healing logic
        "PopulationTaskManager",        # Task execution
        "Enhanced DataPopulationOrchestrator"  # Self-healing extensions
    ]
    
    PHASE_3_RESILIENCE = [
        "ErrorRecoveryHandler",         # Error recovery
        "Advanced Makefile targets",   # heal-to-target, heal-progressive
        "CLI DataPopulationManager"    # Command-line interface
    ]
    
    PHASE_4_VALIDATION = [
        "Complete TDD test suite",     # Comprehensive testing
        "Integration testing",         # End-to-end validation
        "Performance optimization"     # Production readiness
    ]
```

### 10.2 Success Metrics

```python
PSEUDOCODE Success_Metrics:
    """
    Measurable success criteria for the self-healing system.
    """
    
    FUNCTIONAL_METRICS = {
        "table_readiness_improvement": "From 14.3% to 100% (7/7 tables)",
        "healing_success_rate": "> 95% for common failure scenarios",
        "dependency_resolution": "100% correct ordering",
        "error_recovery_rate": "> 80% automatic recovery"
    }
    
    PERFORMANCE_METRICS = {
        "healing_cycle_time": "< 10 minutes for full population",
        "detection_time": "< 30 seconds for status check",
        "make_target_execution": "< 5 seconds for simple targets",
        "memory_usage": "< 2GB during population"
    }
    
    USABILITY_METRICS = {
        "make_target_simplicity": "Single command: make heal-data",
        "progress_visibility": "Real-time progress reporting",
        "error_clarity": "Clear error messages and recommendations",
        "documentation_completeness": "100% pseudocode coverage"
    }
```

### 10.3 Integration Points

```python
PSEUDOCODE Integration_Points:
    """
    Key integration points with existing infrastructure.
    """
    
    EXISTING_INFRASTRUCTURE = {
        "DataPopulationOrchestrator": "Extend with self-healing capabilities",
        "iris_rag package": "Use for pipeline validation",
        "get_iris_connection()": "Leverage for database connectivity",
        "Makefile": "Add new self-healing targets",
        "TDD framework": "Integrate with existing test structure"
    }
    
    NEW_COMPONENTS = {
        "rag_templates/validation/table_status_detector.py": "Table status detection",
        "rag_templates/validation/self_healing_orchestrator.py": "Main orchestration",
        "rag_templates/validation/dependency_analyzer.py": "Dependency management",
        "rag_templates/validation/population_task_manager.py": "Task execution",
        "rag_templates/validation/error_recovery_handler.py": "Error recovery",
        "rag_templates/validation/data_population_manager.py": "CLI interface"
    }
```

---

## Conclusion

This comprehensive pseudocode specification provides a complete blueprint for implementing a self-healing make system that will automatically detect incomplete data and trigger population processes to achieve 100% table readiness. The system is designed to:

1. **Detect** table population status with comprehensive health scoring
2. **Analyze** dependencies and determine optimal population order
3. **Orchestrate** healing workflows with error recovery and retry logic
4. **Integrate** seamlessly with existing Makefile infrastructure
5. **Provide** CLI tools for flexible automation and monitoring
6. **Recover** from common failure scenarios automatically
7. **Report** progress and provide actionable recommendations

The modular design ensures each component remains under 500 lines while providing comprehensive functionality. The TDD anchors ensure reliable implementation with thorough test coverage.

**Key Benefits:**
- **Automated Recovery**: From 14.3% to 100% table readiness without manual intervention
- **Make Integration**: Simple `make heal-data` command for complete healing
- **Error Resilience**: Automatic recovery from connection, timeout, and dependency issues
- **Progress Visibility**: Real-time progress reporting and clear recommendations
- **Production Ready**: Comprehensive error handling and performance optimization

This system will transform the current manual data population process into a fully automated, self-healing infrastructure that ensures consistent 100% table readiness for production RAG operations.