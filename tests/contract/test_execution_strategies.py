"""
Contract tests for execution strategy determination.

These tests define the expected API contract for:
- Sequential execution strategy for community mode (FR-004)
- Parallel execution strategy for enterprise mode (FR-005)
- Execution strategy as property of BackendConfiguration

Contract: specs/035-make-2-modes/contracts/backend_config_contract.yaml
Status: Tests MUST FAIL until Phase 3.3 implementation
"""

import pytest

# These imports will fail until implementation - expected behavior for TDD
from iris_rag.testing.backend_manager import (
    BackendConfiguration,
    BackendMode,
    ConfigSource,
    ExecutionStrategy,
)


@pytest.mark.contract
@pytest.mark.requires_backend_mode
class TestExecutionStrategyDetermination:
    """Contract tests for execution_strategy property."""

    def test_community_mode_sequential_strategy(self):
        """
        Community mode uses SEQUENTIAL execution strategy.

        Given: BackendConfiguration with mode=COMMUNITY
        When: execution_strategy property accessed
        Then: Returns ExecutionStrategy.SEQUENTIAL

        Requirement: FR-004
        """
        config = BackendConfiguration(
            mode=BackendMode.COMMUNITY,
            source=ConfigSource.DEFAULT,
        )

        assert config.execution_strategy == ExecutionStrategy.SEQUENTIAL

    def test_enterprise_mode_parallel_strategy(self):
        """
        Enterprise mode uses PARALLEL execution strategy.

        Given: BackendConfiguration with mode=ENTERPRISE
        When: execution_strategy property accessed
        Then: Returns ExecutionStrategy.PARALLEL

        Requirement: FR-005
        """
        config = BackendConfiguration(
            mode=BackendMode.ENTERPRISE,
            source=ConfigSource.DEFAULT,
        )

        assert config.execution_strategy == ExecutionStrategy.PARALLEL


@pytest.mark.contract
@pytest.mark.requires_backend_mode
class TestExecutionStrategyEnum:
    """Contract tests for ExecutionStrategy enum."""

    def test_execution_strategy_enum_values(self):
        """
        ExecutionStrategy enum has expected values.

        Given: ExecutionStrategy enum
        When: Enum values accessed
        Then: SEQUENTIAL and PARALLEL values exist

        Requirement: FR-004, FR-005
        """
        assert hasattr(ExecutionStrategy, "SEQUENTIAL")
        assert hasattr(ExecutionStrategy, "PARALLEL")

        # Check enum values
        assert ExecutionStrategy.SEQUENTIAL.value == "sequential"
        assert ExecutionStrategy.PARALLEL.value == "parallel"

    def test_execution_strategy_immutable(self):
        """
        ExecutionStrategy is part of immutable BackendConfiguration.

        Given: BackendConfiguration instance
        When: Attempt to modify execution_strategy
        Then: Raises AttributeError (frozen dataclass)

        Requirement: NFR-002 (Immutability)
        """
        config = BackendConfiguration(
            mode=BackendMode.COMMUNITY,
            source=ConfigSource.DEFAULT,
        )

        with pytest.raises(AttributeError):
            config.execution_strategy = ExecutionStrategy.PARALLEL


@pytest.mark.contract
@pytest.mark.requires_backend_mode
class TestExecutionStrategyConfiguration:
    """Contract tests for execution strategy configuration mapping."""

    def test_strategy_matches_connection_limit(self):
        """
        Execution strategy aligns with connection limits.

        Given: BackendConfiguration for each mode
        When: Checking execution_strategy and max_connections
        Then: SEQUENTIAL mode has 1 connection, PARALLEL has 999

        Requirement: FR-003, FR-004, FR-005
        """
        community_config = BackendConfiguration(
            mode=BackendMode.COMMUNITY,
            source=ConfigSource.DEFAULT,
        )
        enterprise_config = BackendConfiguration(
            mode=BackendMode.ENTERPRISE,
            source=ConfigSource.DEFAULT,
        )

        # Community: sequential + single connection
        assert community_config.execution_strategy == ExecutionStrategy.SEQUENTIAL
        assert community_config.max_connections == 1

        # Enterprise: parallel + many connections
        assert enterprise_config.execution_strategy == ExecutionStrategy.PARALLEL
        assert enterprise_config.max_connections == 999

    def test_strategy_from_environment_variable(self, monkeypatch):
        """
        Execution strategy determined from environment variable mode.

        Given: IRIS_BACKEND_MODE=enterprise
        When: load_configuration() called
        Then: execution_strategy is PARALLEL

        Requirement: FR-002, FR-005
        """
        from iris_rag.testing.backend_manager import load_configuration

        monkeypatch.setenv("IRIS_BACKEND_MODE", "enterprise")

        config = load_configuration()

        assert config.execution_strategy == ExecutionStrategy.PARALLEL
