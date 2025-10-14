"""Pytest fixtures for GraphRAG contract tests."""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List
from tests.fixtures.graphrag.fixture_service import FixtureService
from tests.fixtures.graphrag.validator_service import ValidatorService


@pytest.fixture
def fixture_service():
    """Fixture service for managing test fixtures."""
    service = FixtureService()
    
    # Enhanced fixture service with contract test methods
    class EnhancedFixtureService(FixtureService):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._fixtures_by_type = {}
            self._fixtures_by_id = {}
        
        def create_fixture(self, fixture_data: Dict[str, Any]) -> Dict[str, Any]:
            """Create a fixture (contract test implementation)."""
            fixture_id = fixture_data.get("fixture_id")
            fixture_type = fixture_data.get("fixture_type", "document")
            
            # Store in memory
            if fixture_type not in self._fixtures_by_type:
                self._fixtures_by_type[fixture_type] = []
            
            self._fixtures_by_type[fixture_type].append(fixture_data)
            self._fixtures_by_id[fixture_id] = fixture_data
            
            return fixture_data
        
        def list_fixtures(self, fixture_type: str = None, tags: List[str] = None) -> List[Dict[str, Any]]:
            """List fixtures by type or tags."""
            if fixture_type:
                return self._fixtures_by_type.get(fixture_type, [])
            
            if tags:
                all_fixtures = []
                for fixtures in self._fixtures_by_type.values():
                    all_fixtures.extend(fixtures)
                
                # Filter by tags
                return [f for f in all_fixtures if any(tag in f.get("tags", []) for tag in tags)]
            
            return []
        
        def get_fixture(self, fixture_id: str) -> Dict[str, Any]:
            """Get a fixture by ID."""
            return self._fixtures_by_id.get(fixture_id)
    
    return EnhancedFixtureService()


@pytest.fixture
def test_run_service():
    """Test run service for managing test execution metadata."""
    class TestRunService:
        def __init__(self):
            self.runs = {}
            self.results = {}
        
        def start_run(self, run_data: Dict[str, Any]) -> Dict[str, Any]:
            """Start a new test run."""
            import uuid
            run_id = str(uuid.uuid4())
            
            run = {
                "run_id": run_id,
                "test_suite": run_data.get("test_suite"),
                "parallel_execution": run_data.get("parallel_execution", False),
                "environment": run_data.get("environment", {}),
                "start_time": datetime.utcnow().isoformat() + "Z",
                "status": "running"
            }
            
            self.runs[run_id] = run
            self.results[run_id] = []
            
            return run
        
        def update_run(self, run_id: str, update_data: Dict[str, Any]) -> Dict[str, Any]:
            """Update a test run with completion data."""
            if run_id not in self.runs:
                # Create a mock run for testing
                self.runs[run_id] = {
                    "run_id": run_id,
                    "start_time": "2024-10-10T12:00:00Z"
                }
            
            run = self.runs[run_id]
            run.update(update_data)
            
            # Calculate duration
            if "start_time" in run and "end_time" in update_data:
                start = datetime.fromisoformat(run["start_time"].replace("Z", ""))
                end = datetime.fromisoformat(update_data["end_time"].replace("Z", ""))
                duration = (end - start).total_seconds()
                run["duration_seconds"] = duration
            
            run["status"] = "completed"
            
            return run
        
        def add_result(self, run_id: str, result_data: Dict[str, Any]) -> Dict[str, Any]:
            """Add a test result to a run."""
            if run_id not in self.results:
                self.results[run_id] = []
            
            self.results[run_id].append(result_data)
            
            return result_data
        
        def get_results(self, run_id: str) -> List[Dict[str, Any]]:
            """Get all results for a run."""
            return self.results.get(run_id, [])
    
    return TestRunService()


@pytest.fixture
def validator_service():
    """Validator service for test data validation."""
    return ValidatorService()


@pytest.fixture
def performance_monitor():
    """Performance monitor for test execution metrics."""
    class PerformanceMonitor:
        def validate_duration(self, test_run: Dict[str, Any]) -> Dict[str, bool]:
            """Validate test run duration against limits."""
            start = datetime.fromisoformat(test_run["start_time"].replace("Z", ""))
            end = datetime.fromisoformat(test_run["end_time"].replace("Z", ""))

            duration = (end - start).total_seconds()
            duration_minutes = duration / 60.0

            # 30 minute limit
            within_limit = duration_minutes <= 30

            return {
                "within_limit": within_limit,
                "duration_minutes": duration_minutes,
                "duration_seconds": duration
            }

        def calculate_improvement(
            self,
            sequential_run: Dict[str, Any],
            parallel_run: Dict[str, Any]
        ) -> Dict[str, float]:
            """Calculate performance improvement from parallel execution."""
            seq_duration = sequential_run["duration_seconds"]
            par_duration = parallel_run["duration_seconds"]

            speedup_factor = seq_duration / par_duration
            time_saved = seq_duration - par_duration

            return {
                "speedup_factor": speedup_factor,
                "time_saved_seconds": time_saved,
                "improvement_percentage": ((seq_duration - par_duration) / seq_duration) * 100
            }

    return PerformanceMonitor()


@pytest.fixture
def graphrag_sample_fixture():
    """
    GraphRAG sample fixture with 3 source documents and 6 entities.

    This is a minimal fixture for testing entity insertion and FK constraints.
    """
    from dataclasses import dataclass
    from typing import List

    @dataclass
    class SourceDocument:
        doc_id: str
        title: str
        content: str
        metadata: dict

    @dataclass
    class Entity:
        entity_id: str
        name: str
        entity_type: str
        source_document_id: str
        description: str = ""
        metadata: dict = None

    @dataclass
    class GraphRAGFixture:
        source_documents: List[SourceDocument]
        entities: List[Entity]

    # Create 3 source documents
    source_documents = [
        SourceDocument(
            doc_id="med-doc-001",
            title="Diabetes Treatment Guidelines",
            content="Diabetes mellitus is treated with insulin therapy. Metformin is first-line for Type 2 diabetes.",
            metadata={"source": "contract_test", "category": "medical"}
        ),
        SourceDocument(
            doc_id="med-doc-002",
            title="Cardiovascular Disease Management",
            content="Hypertension is managed with ACE inhibitors. Beta blockers reduce cardiac workload.",
            metadata={"source": "contract_test", "category": "medical"}
        ),
        SourceDocument(
            doc_id="med-doc-003",
            title="Respiratory Conditions",
            content="Asthma is controlled with inhaled corticosteroids. COPD requires bronchodilators.",
            metadata={"source": "contract_test", "category": "medical"}
        ),
    ]

    # Create 6 entities referencing these documents by doc_id
    entities = [
        Entity(
            entity_id="ent-001",
            name="Diabetes mellitus",
            entity_type="Disease",
            source_document_id="med-doc-001",
            description="Chronic metabolic disease"
        ),
        Entity(
            entity_id="ent-002",
            name="Metformin",
            entity_type="Medication",
            source_document_id="med-doc-001",
            description="First-line diabetes medication"
        ),
        Entity(
            entity_id="ent-003",
            name="Hypertension",
            entity_type="Disease",
            source_document_id="med-doc-002",
            description="High blood pressure"
        ),
        Entity(
            entity_id="ent-004",
            name="ACE inhibitors",
            entity_type="Medication",
            source_document_id="med-doc-002",
            description="Blood pressure medication"
        ),
        Entity(
            entity_id="ent-005",
            name="Asthma",
            entity_type="Disease",
            source_document_id="med-doc-003",
            description="Chronic respiratory condition"
        ),
        Entity(
            entity_id="ent-006",
            name="Inhaled corticosteroids",
            entity_type="Medication",
            source_document_id="med-doc-003",
            description="Asthma controller medication"
        ),
    ]

    return GraphRAGFixture(source_documents=source_documents, entities=entities)
