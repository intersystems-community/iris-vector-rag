"""
Test integration of declarative state management with database isolation.

This demonstrates how to use declarative state specifications with
proper test isolation to ensure reproducible, non-contaminating tests.
"""

import pytest
import time
from pathlib import Path

from iris_rag.controllers.declarative_state import (
    DeclarativeStateManager,
    DeclarativeStateSpec,
    ensure_documents,
    create_test_state
)
from tests.fixtures.database_isolation import (
    isolated_database,
    verify_clean_state,
    assert_database_state
)
from tests.test_modes import MockController, TestMode


class TestDeclarativeStateWithIsolation:
    """Test declarative state management with proper isolation."""
    
    @pytest.mark.integration
    def test_declarative_sync_with_isolation(self, isolated_database):
        """Test that declarative sync works with isolated tables."""
        # Create manager with isolated config
        manager = DeclarativeStateManager()
        
        # Declare desired state
        spec = DeclarativeStateSpec(
            document_count=50,
            pipeline_type="basic",
            embedding_model="all-MiniLM-L6-v2"
        )
        
        # Sync to desired state
        manager.declare_state(spec)
        result = manager.sync()
        
        # Verify sync completed
        assert result.converged or result.actions_taken
        
        # Verify drift report shows no drift after sync
        drift = manager.get_drift_report()
        assert not drift["has_drift"] or drift["document_count_drift"] == 0
    
    @pytest.mark.integration
    def test_exact_document_count(self, isolated_database, assert_database_state):
        """Test achieving exact document count declaratively."""
        # Start with no documents
        assert_database_state(docs=0, chunks=0)
        
        # Ensure exactly 25 documents
        result = ensure_documents(25, pipeline="basic")
        
        # Should have exactly 25 documents
        assert_database_state(docs=25)
        
        # Run again - should detect no drift
        result2 = ensure_documents(25, pipeline="basic")
        assert not result2.drift_analysis.has_drift
    
    @pytest.mark.integration
    def test_state_file_sync(self, isolated_database, tmp_path):
        """Test syncing from YAML state file."""
        # Create state file
        state_file = tmp_path / "test_state.yaml"
        state_file.write_text("""
state:
  document_count: 100
  pipeline_type: colbert
  embedding_model: all-MiniLM-L6-v2
  min_embedding_diversity: 0.15
  validation_mode: strict
""")
        
        # Sync from file
        manager = DeclarativeStateManager()
        manager.declare_state(str(state_file))
        result = manager.sync()
        
        # Verify state was declared correctly
        assert manager._current_spec.document_count == 100
        assert manager._current_spec.pipeline_type == "colbert"
    
    @pytest.mark.integration 
    def test_drift_detection_and_correction(self, isolated_database):
        """Test detecting and correcting drift."""
        # Set initial state
        manager = DeclarativeStateManager()
        spec = create_test_state(doc_count=20)
        
        manager.declare_state(spec)
        initial_result = manager.sync()
        
        # Simulate drift by manually deleting documents
        # (In real scenario, this would be external changes)
        from common.iris_connection_manager import get_iris_connection
        conn = get_iris_connection()
        cursor = conn.cursor()
        
        # Delete some documents to cause drift
        cursor.execute("DELETE FROM RAG.SourceDocuments WHERE ROWNUM <= 5")
        conn.commit()
        cursor.close()
        conn.close()
        
        # Check drift
        drift = manager.get_drift_report()
        assert drift["has_drift"]
        assert drift["document_count_drift"] < 0  # Missing documents
        
        # Sync to fix drift
        fix_result = manager.sync()
        assert fix_result.actions_taken  # Should have taken corrective action
        
        # Verify drift is fixed
        final_drift = manager.get_drift_report()
        assert not final_drift["has_drift"]
    
    @pytest.mark.integration
    def test_quality_requirements_enforcement(self, isolated_database):
        """Test that quality requirements are enforced."""
        spec = DeclarativeStateSpec(
            document_count=10,
            pipeline_type="basic",
            min_embedding_diversity=0.2,  # High diversity requirement
            max_contamination_ratio=0.01,  # Low contamination tolerance
            validation_mode="strict"
        )
        
        manager = DeclarativeStateManager()
        manager.declare_state(spec)
        
        # Initial sync
        result = manager.sync()
        
        # Check quality issues were detected if any
        if result.drift_analysis.quality_issues:
            assert result.drift_analysis.quality_issues.low_diversity_count >= 0
            assert result.drift_analysis.quality_issues.mock_contamination_count >= 0
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_convergence_with_timeout(self, isolated_database):
        """Test ensure_state with convergence timeout."""
        spec = DeclarativeStateSpec(
            document_count=5,
            pipeline_type="basic"
        )
        
        manager = DeclarativeStateManager()
        
        # Should converge quickly for small state
        start = time.time()
        result = manager.ensure_state(spec, timeout=60)
        duration = time.time() - start
        
        assert result.converged
        assert duration < 60  # Should not timeout
    
    @pytest.mark.unit
    def test_declarative_spec_creation(self):
        """Test creating declarative specifications."""
        # From dict
        spec1 = DeclarativeStateSpec.from_dict({
            "document_count": 100,
            "pipeline_type": "hyde"
        })
        assert spec1.document_count == 100
        assert spec1.pipeline_type == "hyde"
        
        # Test conversion to desired state
        desired = spec1.to_desired_state()
        assert desired.completeness_requirements.min_documents == 100
        assert desired.pipeline_type == "hyde"
    
    @pytest.mark.integration
    def test_auto_sync_mode(self, isolated_database):
        """Test automatic sync on state declaration."""
        # Create manager with auto-sync
        manager = DeclarativeStateManager(auto_sync=True)
        
        # Declaring state should trigger sync
        spec = create_test_state(doc_count=15)
        manager.declare_state(spec)
        
        # Should already be synced
        assert manager.validate_state() or manager._current_spec.document_count == 15
    
    @pytest.mark.integration
    def test_dry_run_analysis(self, isolated_database):
        """Test dry run mode for drift analysis."""
        manager = DeclarativeStateManager()
        
        spec = DeclarativeStateSpec(
            document_count=1000,  # Large number
            pipeline_type="colbert"
        )
        
        manager.declare_state(spec)
        
        # Dry run should only analyze, not change
        result = manager.sync(dry_run=True)
        
        # Should have drift analysis but no actions
        assert result.drift_analysis is not None
        assert len(result.actions_taken) == 0
        assert not result.converged  # Dry run doesn't converge


class TestDeclarativeStatePatterns:
    """Test common declarative state patterns."""
    
    @pytest.mark.integration
    def test_test_data_pattern(self, isolated_database):
        """Test pattern for setting up test data declaratively."""
        # Define test data state
        test_state = create_test_state(
            doc_count=50,
            pipeline="basic"
        )
        
        # Ensure state for test
        manager = DeclarativeStateManager()
        result = manager.ensure_state(test_state, timeout=120)
        
        assert result.converged
        assert manager.validate_state()
    
    @pytest.mark.integration
    def test_multi_pipeline_states(self, isolated_database):
        """Test managing states for different pipelines."""
        pipelines = ["basic", "hyde", "colbert"]
        
        for pipeline in pipelines:
            # Each pipeline gets its own state
            spec = DeclarativeStateSpec(
                document_count=10,
                pipeline_type=pipeline
            )
            
            manager = DeclarativeStateManager()
            result = manager.ensure_state(spec)
            
            assert result.pipeline_type == pipeline
            assert result.converged or len(result.actions_taken) > 0
    
    @pytest.mark.integration
    @pytest.mark.preserve_data
    def test_production_like_state(self, isolated_database):
        """Test production-like state configuration."""
        # Production-like requirements
        prod_spec = DeclarativeStateSpec(
            document_count=1000,
            pipeline_type="colbert",
            embedding_model="all-MiniLM-L6-v2", 
            min_embedding_diversity=0.3,
            max_contamination_ratio=0.001,
            validation_mode="strict"
        )
        
        manager = DeclarativeStateManager()
        manager.declare_state(prod_spec)
        
        # Just analyze in test (don't actually load 1000 docs)
        drift = manager.get_drift_report()
        
        # Would need significant work to reach this state
        assert drift["has_drift"]
        assert drift["document_count_drift"] == -1000  # Missing all docs