"""Fixture loading and management service for GraphRAG tests."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional


class FixtureService:
    """Service for loading and managing test fixtures."""
    
    def __init__(self, fixtures_dir: Optional[Path] = None):
        """Initialize the fixture service.
        
        Args:
            fixtures_dir: Path to fixtures directory. Defaults to this directory.
        """
        if fixtures_dir is None:
            fixtures_dir = Path(__file__).parent
        self.fixtures_dir = fixtures_dir
        self._cache: Dict[str, Any] = {}
    
    def load_fixtures(self, fixture_type: str, use_cache: bool = True) -> List[Dict[str, Any]]:
        """Load fixtures by type.
        
        Args:
            fixture_type: Type of fixtures to load ('medical', 'technical', etc.)
            use_cache: Whether to use cached fixtures if available
            
        Returns:
            List of fixture documents
        """
        cache_key = fixture_type
        
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        fixture_file = self.fixtures_dir / f"{fixture_type}_docs.json"
        
        if not fixture_file.exists():
            raise FileNotFoundError(f"Fixture file not found: {fixture_file}")
        
        with open(fixture_file, 'r') as f:
            data = json.load(f)
        
        documents = data.get('documents', [])
        
        if use_cache:
            self._cache[cache_key] = documents
        
        return documents
    
    def get_fixture_by_id(self, doc_id: str, fixture_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get a specific fixture by document ID.
        
        Args:
            doc_id: Document ID to find
            fixture_type: Optional fixture type to search in. If None, searches all.
            
        Returns:
            Fixture document if found, None otherwise
        """
        if fixture_type:
            fixtures = self.load_fixtures(fixture_type)
            for doc in fixtures:
                if doc.get('doc_id') == doc_id:
                    return doc
            return None
        
        # Search all fixture types
        for fixture_file in self.fixtures_dir.glob('*_docs.json'):
            fixture_type = fixture_file.stem.replace('_docs', '')
            fixtures = self.load_fixtures(fixture_type)
            for doc in fixtures:
                if doc.get('doc_id') == doc_id:
                    return doc
        
        return None
    
    def list_fixture_types(self) -> List[str]:
        """List all available fixture types.
        
        Returns:
            List of fixture type names
        """
        fixture_files = self.fixtures_dir.glob('*_docs.json')
        return [f.stem.replace('_docs', '') for f in fixture_files]
    
    def get_fixture_count(self, fixture_type: str) -> int:
        """Get count of fixtures for a given type.
        
        Args:
            fixture_type: Type of fixtures to count
            
        Returns:
            Number of fixtures
        """
        fixtures = self.load_fixtures(fixture_type)
        return len(fixtures)
    
    def filter_fixtures(
        self, 
        category: Optional[str] = None,
        complexity: Optional[str] = None,
        min_entity_count: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Filter fixtures by various criteria.
        
        Args:
            category: Filter by category (e.g., 'medical', 'technical')
            complexity: Filter by complexity level ('simple', 'medium', 'high')
            min_entity_count: Minimum number of expected entities
            
        Returns:
            List of matching fixtures
        """
        all_fixtures = []
        
        for fixture_type in self.list_fixture_types():
            all_fixtures.extend(self.load_fixtures(fixture_type))
        
        filtered = all_fixtures
        
        if category:
            filtered = [f for f in filtered if f.get('category') == category]
        
        if complexity:
            filtered = [f for f in filtered if f.get('complexity') == complexity]
        
        if min_entity_count is not None:
            filtered = [
                f for f in filtered 
                if len(f.get('expected_entities', [])) >= min_entity_count
            ]
        
        return filtered
    
    def create_fixture(self, fixture_data: Dict[str, Any], fixture_type: str) -> Dict[str, Any]:
        """Create a new fixture (for testing purposes).
        
        Args:
            fixture_data: Fixture document data
            fixture_type: Type of fixture to add to
            
        Returns:
            The created fixture data
        """
        # This is a test implementation - doesn't persist to disk
        if fixture_type not in self._cache:
            self._cache[fixture_type] = []
        
        self._cache[fixture_type].append(fixture_data)
        return fixture_data
    
    def clear_cache(self):
        """Clear the fixture cache."""
        self._cache.clear()


def load_fixtures(fixture_type: str) -> List[Dict[str, Any]]:
    """Convenience function to load fixtures.
    
    Args:
        fixture_type: Type of fixtures to load
        
    Returns:
        List of fixture documents
    """
    service = FixtureService()
    return service.load_fixtures(fixture_type)
