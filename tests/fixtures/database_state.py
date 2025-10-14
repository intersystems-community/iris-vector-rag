"""
Test database state management.

Manages isolated test data for reliable test execution.
Implements T013 from Feature 028.
"""

import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Any
from datetime import datetime


@dataclass
class TestDatabaseState:
    """
    Represents the state of test data in the database.

    Attributes:
        test_run_id: Unique identifier for this test run
        test_class: Test class name
        test_method: Test method name
        created_at: When this test state was created
        document_ids: List of document IDs created by test
        chunk_ids: List of chunk IDs created by test
        entity_ids: List of entity IDs created by test
        relationship_ids: List of relationship IDs created by test
        metadata: Additional test metadata
    """
    test_run_id: str
    test_class: str
    test_method: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    document_ids: List[str] = field(default_factory=list)
    chunk_ids: List[str] = field(default_factory=list)
    entity_ids: List[str] = field(default_factory=list)
    relationship_ids: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @classmethod
    def create_for_test(cls, test_class: str, test_method: Optional[str] = None) -> 'TestDatabaseState':
        """
        Create a new test database state.

        Args:
            test_class: Name of the test class
            test_method: Name of the test method (optional)

        Returns:
            New TestDatabaseState with unique test_run_id
        """
        return cls(
            test_run_id=str(uuid.uuid4()),
            test_class=test_class,
            test_method=test_method
        )

    def add_document(self, document_id: str) -> None:
        """
        Track a document created during this test.

        Args:
            document_id: ID of created document
        """
        if document_id not in self.document_ids:
            self.document_ids.append(document_id)

    def add_chunk(self, chunk_id: str) -> None:
        """
        Track a chunk created during this test.

        Args:
            chunk_id: ID of created chunk
        """
        if chunk_id not in self.chunk_ids:
            self.chunk_ids.append(chunk_id)

    def add_entity(self, entity_id: str) -> None:
        """
        Track an entity created during this test.

        Args:
            entity_id: ID of created entity
        """
        if entity_id not in self.entity_ids:
            self.entity_ids.append(entity_id)

    def add_relationship(self, relationship_id: str) -> None:
        """
        Track a relationship created during this test.

        Args:
            relationship_id: ID of created relationship
        """
        if relationship_id not in self.relationship_ids:
            self.relationship_ids.append(relationship_id)

    @property
    def total_entities_created(self) -> int:
        """Total number of database entities created."""
        return (
            len(self.document_ids)
            + len(self.chunk_ids)
            + len(self.entity_ids)
            + len(self.relationship_ids)
        )

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"TestDatabaseState({self.test_class}.{self.test_method}, "
            f"{self.total_entities_created} entities, "
            f"run_id={self.test_run_id[:8]}...)"
        )


class TestStateRegistry:
    """
    Global registry of active test states.

    Maintains a registry of all active test database states for cleanup tracking.
    """

    _instance: Optional['TestStateRegistry'] = None
    _states: dict = {}

    def __new__(cls) -> 'TestStateRegistry':
        """Singleton pattern for global registry."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._states = {}
        return cls._instance

    def register_state(self, state: TestDatabaseState) -> None:
        """
        Register a new test state.

        Args:
            state: TestDatabaseState to register
        """
        self._states[state.test_run_id] = state

    def get_state(self, test_run_id: str) -> Optional[TestDatabaseState]:
        """
        Retrieve a registered test state.

        Args:
            test_run_id: ID of test run to retrieve

        Returns:
            TestDatabaseState if found, None otherwise
        """
        return self._states.get(test_run_id)

    def remove_state(self, test_run_id: str) -> None:
        """
        Remove a test state from registry.

        Args:
            test_run_id: ID of test run to remove
        """
        if test_run_id in self._states:
            del self._states[test_run_id]

    def get_all_states(self) -> List[TestDatabaseState]:
        """
        Get all registered test states.

        Returns:
            List of all active TestDatabaseStates
        """
        return list(self._states.values())

    def clear(self) -> None:
        """Clear all registered states (use with caution)."""
        self._states.clear()

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"TestStateRegistry({len(self._states)} active states)"
