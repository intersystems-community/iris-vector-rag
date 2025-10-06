"""
Sample Data Utilities

Provides functions to load and manage sample queries and test data for the RAG demo application.
Handles sample query loading, categorization, and selection for testing different pipeline capabilities.
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SampleQuery:
    """Represents a sample query with metadata."""

    id: str
    query: str
    category: str
    difficulty: str
    expected_topics: List[str]
    description: Optional[str] = None


@dataclass
class QueryCategory:
    """Represents a category of sample queries."""

    name: str
    description: str
    queries: List[SampleQuery]


class SampleDataManager:
    """Manages sample queries and test data for the RAG demo application."""

    def __init__(self, data_path: Optional[Path] = None):
        """Initialize the sample data manager.

        Args:
            data_path: Optional path to the sample data directory
        """
        if data_path is None:
            data_path = Path(__file__).parent.parent / "data"

        self.data_path = data_path
        self._sample_queries: Optional[Dict[str, QueryCategory]] = None
        self._test_scenarios: Optional[Dict[str, Any]] = None

    def load_sample_queries(self) -> Dict[str, QueryCategory]:
        """Load sample queries from JSON file.

        Returns:
            Dictionary mapping category names to QueryCategory objects
        """
        if self._sample_queries is not None:
            return self._sample_queries

        try:
            queries_file = self.data_path / "sample_queries.json"
            with open(queries_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            categories = {}
            for cat_key, cat_data in data.get("sample_queries", {}).items():
                queries = []
                for q_data in cat_data.get("queries", []):
                    query = SampleQuery(
                        id=q_data["id"],
                        query=q_data["query"],
                        category=q_data["category"],
                        difficulty=q_data["difficulty"],
                        expected_topics=q_data["expected_topics"],
                        description=q_data.get("description"),
                    )
                    queries.append(query)

                category = QueryCategory(
                    name=cat_data["name"],
                    description=cat_data["description"],
                    queries=queries,
                )
                categories[cat_key] = category

            self._sample_queries = categories
            return categories

        except FileNotFoundError:
            # Return empty categories if file not found
            return self._get_fallback_queries()
        except Exception as e:
            print(f"Error loading sample queries: {e}")
            return self._get_fallback_queries()

    def get_queries_by_category(self, category: str) -> List[SampleQuery]:
        """Get all queries for a specific category.

        Args:
            category: Category name

        Returns:
            List of SampleQuery objects
        """
        categories = self.load_sample_queries()
        if category in categories:
            return categories[category].queries
        return []

    def get_queries_by_difficulty(self, difficulty: str) -> List[SampleQuery]:
        """Get all queries for a specific difficulty level.

        Args:
            difficulty: Difficulty level (easy, medium, hard)

        Returns:
            List of SampleQuery objects
        """
        all_queries = self.get_all_queries()
        return [q for q in all_queries if q.difficulty == difficulty]

    def get_all_queries(self) -> List[SampleQuery]:
        """Get all sample queries across all categories.

        Returns:
            List of all SampleQuery objects
        """
        categories = self.load_sample_queries()
        all_queries = []
        for category in categories.values():
            all_queries.extend(category.queries)
        return all_queries

    def get_random_query(
        self, category: Optional[str] = None, difficulty: Optional[str] = None
    ) -> Optional[SampleQuery]:
        """Get a random query with optional filtering.

        Args:
            category: Optional category to filter by
            difficulty: Optional difficulty to filter by

        Returns:
            Random SampleQuery object or None if no queries match
        """
        if category:
            queries = self.get_queries_by_category(category)
        elif difficulty:
            queries = self.get_queries_by_difficulty(difficulty)
        else:
            queries = self.get_all_queries()

        if difficulty and category:
            # Filter by both category and difficulty
            queries = [
                q
                for q in self.get_queries_by_category(category)
                if q.difficulty == difficulty
            ]

        return random.choice(queries) if queries else None

    def get_category_names(self) -> List[str]:
        """Get list of available category names.

        Returns:
            List of category names
        """
        categories = self.load_sample_queries()
        return list(categories.keys())

    def get_category_info(self) -> List[Tuple[str, str, int]]:
        """Get category information with counts.

        Returns:
            List of tuples (category_key, name, query_count)
        """
        categories = self.load_sample_queries()
        return [(key, cat.name, len(cat.queries)) for key, cat in categories.items()]

    def get_difficulty_levels(self) -> List[str]:
        """Get list of available difficulty levels.

        Returns:
            List of difficulty levels
        """
        all_queries = self.get_all_queries()
        difficulties = set(q.difficulty for q in all_queries)
        return sorted(list(difficulties))

    def search_queries(self, search_term: str) -> List[SampleQuery]:
        """Search queries by content.

        Args:
            search_term: Term to search for in query text

        Returns:
            List of matching SampleQuery objects
        """
        all_queries = self.get_all_queries()
        search_term_lower = search_term.lower()

        matches = []
        for query in all_queries:
            if search_term_lower in query.query.lower() or any(
                search_term_lower in topic.lower() for topic in query.expected_topics
            ):
                matches.append(query)

        return matches

    def get_comparison_scenarios(self) -> Dict[str, Any]:
        """Get test scenarios for pipeline comparison.

        Returns:
            Dictionary of test scenarios
        """
        if self._test_scenarios is not None:
            return self._test_scenarios

        try:
            queries_file = self.data_path / "sample_queries.json"
            with open(queries_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._test_scenarios = data.get("test_scenarios", {})
            return self._test_scenarios

        except Exception as e:
            print(f"Error loading test scenarios: {e}")
            return {}

    def get_pipeline_comparison_queries(self) -> List[Dict[str, Any]]:
        """Get queries specifically designed for pipeline comparison.

        Returns:
            List of comparison scenario dictionaries
        """
        scenarios = self.get_comparison_scenarios()
        pipeline_comp = scenarios.get("pipeline_comparison", {})
        return pipeline_comp.get("scenarios", [])

    def _get_fallback_queries(self) -> Dict[str, QueryCategory]:
        """Get fallback queries when sample data file is not available.

        Returns:
            Dictionary with basic fallback queries
        """
        fallback_queries = [
            SampleQuery(
                id="fallback_001",
                query="What is machine learning?",
                category="definition",
                difficulty="easy",
                expected_topics=["machine learning", "AI", "algorithms"],
            ),
            SampleQuery(
                id="fallback_002",
                query="How does a neural network work?",
                category="explanation",
                difficulty="medium",
                expected_topics=["neural networks", "deep learning", "neurons"],
            ),
            SampleQuery(
                id="fallback_003",
                query="What are the main types of machine learning?",
                category="classification",
                difficulty="easy",
                expected_topics=["supervised", "unsupervised", "reinforcement"],
            ),
        ]

        fallback_category = QueryCategory(
            name="Basic Questions",
            description="Basic machine learning questions",
            queries=fallback_queries,
        )

        return {"basic": fallback_category}

    def export_query_stats(self) -> Dict[str, Any]:
        """Export statistics about the sample queries.

        Returns:
            Dictionary with query statistics
        """
        categories = self.load_sample_queries()
        all_queries = self.get_all_queries()

        stats = {
            "total_queries": len(all_queries),
            "total_categories": len(categories),
            "difficulty_distribution": {},
            "category_distribution": {},
            "average_topics_per_query": 0,
        }

        # Calculate difficulty distribution
        for difficulty in self.get_difficulty_levels():
            count = len(self.get_queries_by_difficulty(difficulty))
            stats["difficulty_distribution"][difficulty] = count

        # Calculate category distribution
        for cat_key, category in categories.items():
            stats["category_distribution"][cat_key] = len(category.queries)

        # Calculate average topics per query
        if all_queries:
            total_topics = sum(len(q.expected_topics) for q in all_queries)
            stats["average_topics_per_query"] = total_topics / len(all_queries)

        return stats


# Global instance for easy access
sample_data_manager = SampleDataManager()


# Convenience functions
def get_sample_queries() -> Dict[str, QueryCategory]:
    """Get all sample queries."""
    return sample_data_manager.load_sample_queries()


def get_random_sample_query(
    category: Optional[str] = None, difficulty: Optional[str] = None
) -> Optional[SampleQuery]:
    """Get a random sample query."""
    return sample_data_manager.get_random_query(category, difficulty)


def get_queries_for_category(category: str) -> List[SampleQuery]:
    """Get queries for a specific category."""
    return sample_data_manager.get_queries_by_category(category)


def search_sample_queries(search_term: str) -> List[SampleQuery]:
    """Search sample queries."""
    return sample_data_manager.search_queries(search_term)


def get_comparison_test_scenarios() -> List[Dict[str, Any]]:
    """Get test scenarios for pipeline comparison."""
    return sample_data_manager.get_pipeline_comparison_queries()
