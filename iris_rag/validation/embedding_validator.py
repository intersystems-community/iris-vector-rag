"""
Embedding Quality Validator for Reconciliation Framework.

This module provides validation logic to detect mock embeddings, calculate diversity scores,
and identify quality issues in embedding data for the reconciliation system.
"""

import logging
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from ..config.manager import ConfigurationManager
from ..core.connection import ConnectionManager

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingQualityIssues:
    """Represents quality issues detected in embedding data."""
    mock_embeddings_detected: bool = False
    avg_diversity_score: float = 1.0
    mock_document_count: int = 0
    low_diversity_document_count: int = 0
    total_analyzed: int = 0
    diversity_threshold: float = 0.7
    mock_detection_threshold: float = 0.95
    issues_summary: List[str] = None
    # Backward compatibility attributes
    missing_embeddings_count: int = 0
    corrupted_embeddings_count: int = 0
    
    def __post_init__(self):
        if self.issues_summary is None:
            self.issues_summary = []


class EmbeddingValidator:
    """
    Validates embedding quality for the reconciliation framework.
    
    This validator detects:
    - Mock embeddings (too many identical values)
    - Low diversity embeddings (insufficient variance)
    - Corrupted or malformed embeddings
    """
    
    def __init__(self, config_manager: ConfigurationManager):
        """
        Initialize the embedding validator.
        
        Args:
            config_manager: Configuration manager for accessing settings
        """
        self.config_manager = config_manager
        self.connection_manager = ConnectionManager(config_manager)
        self.logger = logging.getLogger(__name__)
        
        # Get validation thresholds from configuration
        validation_config = self.config_manager.get_desired_embedding_state().get("validation", {})
        self.diversity_threshold = validation_config.get("diversity_threshold", 0.7)
        self.mock_detection_threshold = validation_config.get("mock_detection_threshold", 0.95)
        self.mock_detection_enabled = validation_config.get("mock_detection_enabled", True)
        
        self.logger.debug(f"EmbeddingValidator initialized with diversity_threshold={self.diversity_threshold}, "
                         f"mock_detection_threshold={self.mock_detection_threshold}")
    
    def analyze_quality(self, sample_embeddings: List[Tuple[str, str]]) -> EmbeddingQualityIssues:
        """
        Analyze embedding quality from a sample of embeddings.
        
        Args:
            sample_embeddings: List of (doc_id, embedding_string) tuples
            
        Returns:
            EmbeddingQualityIssues object with analysis results
        """
        self.logger.debug(f"Analyzing quality of {len(sample_embeddings)} embedding samples")
        
        if not sample_embeddings:
            return EmbeddingQualityIssues(
                total_analyzed=0,
                issues_summary=["No embeddings provided for analysis"]
            )
        
        # Parse embeddings and analyze
        parsed_embeddings = []
        mock_count = 0
        low_diversity_count = 0
        diversity_scores = []
        issues = []
        
        for doc_id, embedding_str in sample_embeddings:
            try:
                # Parse embedding string to numerical array
                embedding_array = self._parse_embedding_string(embedding_str)
                if embedding_array is None:
                    issues.append(f"Failed to parse embedding for document {doc_id}")
                    continue
                
                parsed_embeddings.append((doc_id, embedding_array))
                
                # Check for mock embedding
                if self.mock_detection_enabled and self._is_mock_embedding(embedding_array):
                    mock_count += 1
                    issues.append(f"Mock embedding detected in document {doc_id}")
                
                # Calculate diversity score for this embedding
                diversity_score = self._calculate_embedding_diversity(embedding_array)
                diversity_scores.append(diversity_score)
                
                # Check if diversity is too low
                if diversity_score < self.diversity_threshold:
                    low_diversity_count += 1
                    issues.append(f"Low diversity embedding in document {doc_id} (score: {diversity_score:.3f})")
                
            except Exception as e:
                issues.append(f"Error analyzing embedding for document {doc_id}: {e}")
                self.logger.warning(f"Error analyzing embedding for {doc_id}: {e}")
        
        # Calculate overall diversity score
        avg_diversity_score = np.mean(diversity_scores) if diversity_scores else 0.0
        
        # Determine if mock embeddings were detected
        mock_embeddings_detected = mock_count > 0
        
        # Additional cross-embedding analysis for mock detection
        if len(parsed_embeddings) > 1 and self.mock_detection_enabled:
            cross_similarity = self._analyze_cross_embedding_similarity(parsed_embeddings)
            if cross_similarity > self.mock_detection_threshold:
                mock_embeddings_detected = True
                issues.append(f"High cross-embedding similarity detected ({cross_similarity:.3f}), indicating potential mock data")
        
        result = EmbeddingQualityIssues(
            mock_embeddings_detected=mock_embeddings_detected,
            avg_diversity_score=avg_diversity_score,
            mock_document_count=mock_count,
            low_diversity_document_count=low_diversity_count,
            total_analyzed=len(sample_embeddings),
            diversity_threshold=self.diversity_threshold,
            mock_detection_threshold=self.mock_detection_threshold,
            issues_summary=issues
        )
        
        self.logger.info(f"Embedding quality analysis complete: {len(sample_embeddings)} analyzed, "
                        f"mock_detected={mock_embeddings_detected}, avg_diversity={avg_diversity_score:.3f}")
        
        return result
    
    def sample_embeddings_from_database(self, table_name: str = "RAG.DocumentTokenEmbeddings", 
                                      sample_size: int = 100) -> List[Tuple[str, str]]:
        """
        Sample embeddings from the database for quality analysis.
        
        Args:
            table_name: Name of the table containing embeddings
            sample_size: Number of embeddings to sample
            
        Returns:
            List of (doc_id, embedding_string) tuples
        """
        self.logger.debug(f"Sampling {sample_size} embeddings from {table_name}")
        
        try:
            iris_connector = self.connection_manager.get_connection("iris")
            cursor = iris_connector.cursor()
            
            # Sample embeddings using IRIS SQL TOP syntax
            cursor.execute(f"""
                SELECT TOP ? doc_id, token_embedding
                FROM {table_name}
                WHERE token_embedding IS NOT NULL
                ORDER BY doc_id, token_index
            """, [sample_size])
            
            results = cursor.fetchall()
            cursor.close()
            
            # Convert to list of tuples with string embeddings
            sample_embeddings = []
            for doc_id, embedding in results:
                if embedding is not None:
                    embedding_str = str(embedding)
                    sample_embeddings.append((doc_id, embedding_str))
            
            self.logger.info(f"Sampled {len(sample_embeddings)} embeddings from {table_name}")
            return sample_embeddings
            
        except Exception as e:
            self.logger.error(f"Failed to sample embeddings from {table_name}: {e}")
            return []
    
    def _parse_embedding_string(self, embedding_str: str) -> Optional[np.ndarray]:
        """
        Parse embedding string to numerical array.
        
        Args:
            embedding_str: String representation of embedding
            
        Returns:
            Numpy array of embedding values, or None if parsing fails
        """
        try:
            # Handle IRIS VECTOR format (comma-separated values)
            if ',' in embedding_str and not embedding_str.startswith('['):
                # IRIS VECTOR format: "0.1,0.2,0.3,..."
                values = [float(x.strip()) for x in embedding_str.split(',')]
                return np.array(values)
            
            # Handle JSON array format: "[0.1, 0.2, 0.3, ...]"
            elif embedding_str.startswith('[') and embedding_str.endswith(']'):
                # Remove brackets and split by comma
                inner = embedding_str[1:-1]
                values = [float(x.strip()) for x in inner.split(',')]
                return np.array(values)
            
            # Handle space-separated format
            elif ' ' in embedding_str:
                values = [float(x.strip()) for x in embedding_str.split()]
                return np.array(values)
            
            else:
                self.logger.warning(f"Unrecognized embedding format: {embedding_str[:50]}...")
                return None
                
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Failed to parse embedding string: {e}")
            return None
    
    def _is_mock_embedding(self, embedding: np.ndarray) -> bool:
        """
        Detect if an embedding is likely a mock (too many identical values).
        
        Args:
            embedding: Numpy array of embedding values
            
        Returns:
            True if embedding appears to be mock data
        """
        if len(embedding) == 0:
            return True
        
        # Check for all zeros
        if np.allclose(embedding, 0.0):
            return True
        
        # Check for all identical values
        if np.allclose(embedding, embedding[0]):
            return True
        
        # Check for too many repeated values
        unique_values = len(np.unique(embedding))
        repetition_ratio = unique_values / len(embedding)
        
        # If less than 10% of values are unique, likely mock
        if repetition_ratio < 0.1:
            return True
        
        # Check for obvious patterns (e.g., incrementing sequence)
        if len(embedding) > 2:
            diffs = np.diff(embedding)
            if np.allclose(diffs, diffs[0]) and not np.allclose(diffs, 0.0):
                return True  # Arithmetic sequence
        
        return False
    
    def _calculate_embedding_diversity(self, embedding: np.ndarray) -> float:
        """
        Calculate diversity score for an embedding.
        
        Args:
            embedding: Numpy array of embedding values
            
        Returns:
            Diversity score between 0.0 and 1.0 (higher is more diverse)
        """
        if len(embedding) == 0:
            return 0.0
        
        # Calculate variance-based diversity
        variance = np.var(embedding)
        
        # Normalize variance to [0, 1] range using sigmoid-like function
        # This maps variance to a diversity score
        diversity_score = 1.0 / (1.0 + np.exp(-10 * (variance - 0.1)))
        
        # Additional diversity measures
        
        # Unique value ratio
        unique_ratio = len(np.unique(embedding)) / len(embedding)
        
        # Range diversity (normalized range of values)
        value_range = np.max(embedding) - np.min(embedding)
        range_diversity = min(1.0, value_range / 2.0)  # Assume good range is around 2.0
        
        # Combine measures with weights
        combined_diversity = (
            0.5 * diversity_score +  # Variance-based
            0.3 * unique_ratio +     # Uniqueness
            0.2 * range_diversity    # Range
        )
        
        return min(1.0, max(0.0, combined_diversity))
    
    def _analyze_cross_embedding_similarity(self, embeddings: List[Tuple[str, np.ndarray]]) -> float:
        """
        Analyze similarity across multiple embeddings to detect mock data.
        
        Args:
            embeddings: List of (doc_id, embedding_array) tuples
            
        Returns:
            Average cosine similarity across all embedding pairs
        """
        if len(embeddings) < 2:
            return 0.0
        
        similarities = []
        
        # Calculate pairwise cosine similarities
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                _, emb1 = embeddings[i]
                _, emb2 = embeddings[j]
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(emb1, emb2)
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity between -1 and 1
        """
        try:
            # Handle zero vectors
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 1.0 if norm_a == norm_b else 0.0
            
            return np.dot(a, b) / (norm_a * norm_b)
            
        except Exception:
            return 0.0