# tests/test_all_rag_with_real_data.py

import pytest
import os
import sys
import logging
from typing import List, Dict, Any, Callable  # Added Callable
from unittest.mock import MagicMock

logger = logging.getLogger(__name__)

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import RAG pipelines
from basic_rag.pipeline import BasicRAGPipeline
from hyde.pipeline import HyDEPipeline
from colbert import ColbertRAGPipeline # Corrected import
from crag.pipeline import CRAGPipeline
from noderag.pipeline import NodeRAGPipeline
from graphrag.pipeline import GraphRAGPipeline

# Import fixtures for real dependencies and evaluation data
# These fixtures are expected to be defined in tests/conftest.py
from tests.conftest import (
    iris_with_pmc_data, # Provides real IRIS connection with data
    embedding_model_fixture, # Provides real embedding function
    llm_client_fixture, # Provides real LLM function (might need config for OpenAI)
    evaluation_dataset # Provides the evaluation queries and ground truth
)

# Import evaluation metrics
# These require ragas and potentially ragchecker
try:
    from ragas import evaluate
    from ragas.metrics import context_recall, faithfulness, answer_relevancy # Add other relevant metrics
    # from ragchecker import check_faithfulness # If using ragchecker
    RAGAS_AVAILABLE = True
except ImportError:
    logger.warning("Ragas library not installed. Skipping Ragas evaluation tests.")
    RAGAS_AVAILABLE = False

# Define the list of pipelines to test
# Each entry is a tuple: (pipeline_name, pipeline_class)
RAG_PIPELINES_TO_TEST = [
    ("BasicRAG", BasicRAGPipeline),
    ("HyDE", HyDEPipeline),
    ("ColBERT", ColbertRAGPipeline),
    ("CRAG", CRAGPipeline),
    ("NodeRAG", NodeRAGPipeline),
    ("GraphRAG", GraphRAGPipeline),
]

# Define metric thresholds as per IMPLEMENTATION_PLAN.md
METRIC_THRESHOLDS = {
    "context_recall": 0.8,
    "faithfulness": 0.7,
    # Add thresholds for other metrics if used
}

# Parametrize the test function over all pipelines and all queries
@pytest.mark.e2e # Mark as end-to-end test
@pytest.mark.real_data # Mark as requiring real data
@pytest.mark.parametrize("pipeline_name, pipeline_class", RAG_PIPELINES_TO_TEST)
@pytest.mark.parametrize("eval_query_data", "evaluation_dataset") # Pass fixture name as string
@pytest.mark.skipif(not RAGAS_AVAILABLE, reason="Ragas library not installed")
def test_rag_pipeline_e2e_metrics(
    pipeline_name: str,
    pipeline_class: type,
    eval_query_data: Dict[str, Any],
    iris_with_pmc_data: Any, # Real IRIS connection with data
    embedding_model_fixture: Callable[[List[str]], List[List[float]]], # Real embedding func
    llm_client_fixture: Callable[[str], str] # Real LLM func
):
    """
    Runs an end-to-end test for a RAG pipeline using real data and evaluates metrics.
    """
    logger.info(f"\n--- Running E2E test for {pipeline_name} with query: '{eval_query_data['query'][:50]}...' ---")

    # Instantiate the pipeline
    # Note: Some pipelines might require additional dependencies (e.g., ColBERT encoders, web search for CRAG, graph_lib for Node/GraphRAG)
    # These should ideally be provided by fixtures or handled within the pipeline instantiation.
    # For simplicity in this test, we'll pass None for optional/complex dependencies if not provided by fixtures.
    
    pipeline_kwargs = {
        "iris_connector": iris_with_pmc_data,
        "embedding_func": embedding_model_fixture,
        "llm_func": llm_client_fixture,
    }

    # Add pipeline-specific dependencies if needed and available from fixtures/mocks
    if pipeline_name == "ColBERT":
        # ColBERT needs query and doc encoders. Doc encoder is used offline, query encoder at query time.
        # Assuming embedding_model_fixture can act as a stand-in for query encoder for this test.
        # A proper ColBERT query encoder fixture might be needed.
        # The pipeline also expects colbert_doc_encoder_func in __init__, even if used offline.
        # Let's use embedding_model_fixture as a mock for both for this test's instantiation.
        pipeline_kwargs["colbert_query_encoder_func"] = embedding_model_fixture # Using general embedder as placeholder
        pipeline_kwargs["colbert_doc_encoder_func"] = MagicMock() # Mock doc encoder as it's offline
    elif pipeline_name == "CRAG":
        # CRAG needs web_search_func. Use a mock for the test.
        pipeline_kwargs["web_search_func"] = MagicMock(return_value=[f"Mock web result for {eval_query_data['query']}"])
    elif pipeline_name == "NodeRAG":
        # NodeRAG needs graph_lib. Use None or a mock.
        pipeline_kwargs["graph_lib"] = None # Placeholder
    elif pipeline_name == "GraphRAG":
        # GraphRAG doesn't need extra dependencies in __init__ based on its pipeline.py
        pass # No extra kwargs needed

    try:
        pipeline = pipeline_class(**pipeline_kwargs)
    except TypeError as e:
        logger.error(f"Failed to instantiate {pipeline_name} pipeline: {e}. Check __init__ signature and provided kwargs.")
        pytest.fail(f"Pipeline instantiation failed: {e}")
        return # Stop test if instantiation fails

    # Run the pipeline
    query = eval_query_data["query"]
    ground_truth_contexts = eval_query_data["ground_truth_contexts"]
    ground_truth_answer = eval_query_data["ground_truth_answer"]

    try:
        result = pipeline.run(query)
    except Exception as e:
        logger.error(f"Error running {pipeline_name} pipeline for query '{query[:50]}...': {e}")
        pytest.fail(f"Pipeline run failed: {e}")
        return # Stop test if run fails

    # Extract results
    retrieved_contexts = [doc.content for doc in result.get("retrieved_documents", [])]
    generated_answer = result.get("answer", "")

    # For CRAG, retrieved_documents is a list of strings (chunks)
    if pipeline_name == "CRAG":
         retrieved_contexts = result.get("retrieved_context_chunks", [])
         # Need to convert list of strings to list of Document objects for Ragas context_recall
         # Ragas context_recall expects List[str] for 'contexts' parameter, so retrieved_contexts is fine.
         # However, other metrics might expect Document objects.
         # Let's ensure retrieved_contexts is always List[str] for Ragas.

    # Prepare data for Ragas evaluation
    # Ragas expects a Dataset object or a dictionary with lists
    # Keys: 'question', 'contexts', 'ground_truth'
    # 'contexts' should be List[List[str]] if evaluating multiple queries in batch,
    # but for a single query, it's List[str].
    # 'ground_truth' should be List[str] for a single query.

    ragas_dataset = {
        'question': [query],
        'contexts': [retrieved_contexts],
        'ground_truth': [ground_truth_answer],
        'ground_truth_contexts': [ground_truth_contexts]
    }

    # Define metrics to use
    metrics = [
        context_recall,
        faithfulness,
        # answer_relevancy, # Requires LLM, might add cost/latency
    ]

    # Evaluate metrics
    try:
        # Ragas evaluate expects a Dataset object, not a raw dict for single evaluation
        # Convert the dict to a Dataset
        from datasets import Dataset
        ragas_dataset_obj = Dataset.from_dict(ragas_dataset)

        results = evaluate(ragas_dataset_obj, metrics=metrics)
        logger.info(f"Evaluation Results for {pipeline_name} (Query: '{query[:50]}...'): {results}")

        # Assert metrics meet thresholds
        for metric_name, threshold in METRIC_THRESHOLDS.items():
            if metric_name in results:
                metric_value = results[metric_name]
                assert metric_value >= threshold, (
                    f"{pipeline_name} failed {metric_name} threshold: {metric_value:.4f} < {threshold}"
                    f" for query '{query}'"
                )
                logger.info(f"✅ {pipeline_name} passed {metric_name} threshold ({metric_value:.4f} ≥ {threshold})")
            else:
                logger.warning(f"Metric '{metric_name}' not found in Ragas results for {pipeline_name}.")

        # TODO: Add RAGChecker faithfulness check if needed/preferred
        # ragchecker_score = check_faithfulness(query, generated_answer, retrieved_contexts)
        # assert ragchecker_score >= METRIC_THRESHOLDS["faithfulness"]

    except ImportError as e:
        logger.error(f"Evaluation library not available: {e}. Skipping metric evaluation.")
        pytest.skip(f"Evaluation library not available: {e}")
    except Exception as e:
        logger.error(f"Error during metric evaluation for {pipeline_name} (Query: '{query[:50]}...'): {e}")
        pytest.fail(f"Metric evaluation failed: {e}")

    logger.info(f"--- E2E test for {pipeline_name} with query '{query[:50]}...' completed successfully ---")

# Note: This test requires a running IRIS instance with data loaded (via iris_with_pmc_data fixture)
# and potentially access to real LLMs (if llm_client_fixture is configured for real LLMs).
# Ensure environment variables (IRIS_CONNECTION_URL, OPENAI_API_KEY, etc.) are set as needed.
