"""
Integration tests for the Evaluation Framework.
"""

import pytest
from iris_vector_rag.evaluation.datasets import DatasetLoader

@pytest.mark.integration
@pytest.mark.requires_internet
def test_dataset_loader_streaming():
    """Verify we can stream queries from a real dataset."""
    loader = DatasetLoader()
    # Use a small sample of HotpotQA
    queries = list(loader.load("hotpotqa", sample_size=2))
    
    assert len(queries) == 2
    assert queries[0].question is not None
    assert len(queries[0].supporting_docs) > 0
