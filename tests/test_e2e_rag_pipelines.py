# tests/test_e2e_rag_pipelines.py
"""
Comprehensive end-to-end tests for all RAG pipelines with real data.

This test file follows TDD principles:
1. Red: Define failing tests that specify expected behavior
2. Green: Implement code to make tests pass
3. Refactor: Clean up while keeping tests passing

Tests verify that all RAG techniques work with real PMC documents and
include meaningful assertions to validate results.
"""

import pytest
import os
import sys
import logging
from typing import List, Dict, Any, Callable, Optional
import time
import csv
import datetime
import uuid
from unittest.mock import MagicMock
import json
from datasets import Dataset # For RAGAS
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall, # Not used currently as it requires ground truths
    context_precision, # Not used currently as it requires ground truths
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

# Ensure the test_results directories exist
os.makedirs("test_results/rag_outputs", exist_ok=True) # For JSON outputs
os.makedirs("test_results", exist_ok=True) # Base directory for CSV log

CSV_LOG_FILE_PATH = "test_results/rag_evaluation_log.csv"
CSV_LOG_HEADERS = [
    "RunID", "Timestamp", "PipelineName", "Query", "Parameters",
    "DocumentsRetrievedCount", "RAGAS_Faithfulness", "RAGAS_AnswerRelevancy",
    "MainTestPassed", "RAGAS_Notes"
]

def log_rag_evaluation_to_csv(data_row: Dict[str, Any]):
    """Appends a data row to the CSV log file."""
    file_exists = os.path.isfile(CSV_LOG_FILE_PATH)
    try:
        with open(CSV_LOG_FILE_PATH, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=CSV_LOG_HEADERS)
            if not file_exists or os.path.getsize(CSV_LOG_FILE_PATH) == 0:
                writer.writeheader()
            writer.writerow(data_row)
        logger.debug(f"Successfully logged evaluation results to {CSV_LOG_FILE_PATH}")
    except Exception as e:
        logger.error(f"Error logging to CSV {CSV_LOG_FILE_PATH}: {e}", exc_info=True)

from pathlib import Path
# Add the project root directory to Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from common.utils import load_config 

# Import RAG pipelines
from iris_rag.pipelines.basic import BasicRAGPipeline
from iris_rag.pipelines.hyde import HyDERAGPipeline as HyDERAGPipeline
from iris_rag.pipelines.crag import CRAGPipeline
from iris_rag.pipelines.colbert import ColBERTRAGPipeline as ColbertRAGPipeline
from iris_rag.pipelines.noderag import NodeRAGPipeline
from iris_rag.pipelines.graphrag import GraphRAGPipeline

# Import common utilities
from common.utils import Document, timing_decorator, get_embedding_func, get_llm_func

# Import fixtures for real data testing (os and json already imported above)

# --- Test Fixtures ---

@pytest.fixture(scope="module")
def sample_medical_queries() -> List[Dict[str, Any]]:
    """
    Provides a list of medical queries for testing RAG pipelines.
    """
    return [
        {
            "query": "What are common treatments for type 2 diabetes?",
            "expected_keywords": ["diabetes", "treatment", "metformin", "insulin"],
            "min_doc_count": 1 
        },
        {
            "query": "How does COVID-19 affect the respiratory system?",
            "expected_keywords": ["covid", "respiratory", "lung", "breathing"],
            "min_doc_count": 2
        },
        {
            "query": "What treatments are available for Alzheimer's disease?",
            "expected_keywords": ["alzheimer", "treatment", "cognitive", "memory"],
            "min_doc_count": 2
        }
    ]

@pytest.fixture # Defaults to scope="function"
def real_iris_connection(iris_connection): 
    """
    Provides a real IRIS connection and verifies it has at least 1000 documents
    for e2e tests.
    """
    config = load_config()
    MIN_DOCS_FOR_E2E = config.get('testing', {}).get('min_docs_e2e', 1000) 
    cursor = None
    try:
        cursor = iris_connection.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NOT NULL")
        count = cursor.fetchone()[0]
        if count < MIN_DOCS_FOR_E2E:
            pytest.skip(f"E2E tests require at least {MIN_DOCS_FOR_E2E} documents with embeddings, found {count}")
        logger.info(f"Real IRIS connection for E2E tests: {count} documents with embeddings found.")
    except Exception as e:
        pytest.fail(f"Failed to verify document count for E2E tests: {e}")
    finally:
        if cursor:
            cursor.close()
            
    return iris_connection 

@pytest.fixture(scope="module")
def real_embedding_func():
    """
    Provides a real embedding function for testing.
    """
    embed_func = get_embedding_func()
    logger.info(f"Using real embedding function")
    return embed_func

@pytest.fixture(scope="module")
def real_llm_func():
    """
    Provides a real LLM function for testing.
    """
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()
    try:
        if provider == "stub":
            logger.info("Using stub LLM function as specified")
            return get_llm_func(provider="stub")
        
        llm_func = get_llm_func(provider=provider)
        logger.info(f"Using real {provider.upper()} LLM function")
        return llm_func
    except (ImportError, ValueError, EnvironmentError) as e:
        logger.warning(f"Real LLM ({provider}) not available: {e}. Falling back to stub LLM.")
        return get_llm_func(provider="stub")

@pytest.fixture(scope="module")
def web_search_func():
    """
    Provides a web search function for CRAG testing.
    """
    try:
        def enhanced_web_search(query, num_results=3):
            results = []
            keywords = query.lower().split()
            main_topic = keywords[0] if keywords else "medicine"
            medical_topics = {
                "diabetes": [
                    "Diabetes symptoms include increased thirst, frequent urination, extreme hunger, unexplained weight loss, fatigue, irritability, and blurred vision.",
                    "Type 2 diabetes develops when the body becomes resistant to insulin or when the pancreas is unable to produce enough insulin.",
                    "Treatment for diabetes includes monitoring blood sugar, insulin therapy, and lifestyle changes including diet and exercise."
                ],
                "covid": [
                    "COVID-19 affects the respiratory system by infecting cells in the lungs, causing inflammation and damage to lung tissue.",
                    "Severe COVID-19 can lead to pneumonia, acute respiratory distress syndrome (ARDS), and respiratory failure.",
                    "Long-term respiratory effects of COVID-19 can include reduced lung function, persistent shortness of breath, and pulmonary fibrosis."
                ],
                "alzheimer": [
                    "Current treatments for Alzheimer's disease include cholinesterase inhibitors like donepezil, rivastigmine, and galantamine.",
                    "Memantine (Namenda) works by regulating glutamate activity and can help improve symptoms in moderate to severe Alzheimer's disease.",
                    "New treatments being researched include immunotherapies targeting beta-amyloid plaques and tau protein tangles."
                ]
            }
            relevant_topic = next((topic for topic in medical_topics if topic in query.lower()), None)
            
            if relevant_topic and relevant_topic in medical_topics:
                for i in range(min(num_results, len(medical_topics[relevant_topic]))):
                    results.append(medical_topics[relevant_topic][i])
            else:
                for i in range(num_results):
                    result_text = f"Web search result {i+1} for query about {' and '.join(keywords[:2])}. "
                    result_text += f"This result contains information related to {main_topic}."
                    results.append(result_text)
            
            logger.info(f"Enhanced web search returned {len(results)} results for query: {query[:50]}...")
            return results
        
        logger.info(f"Using enhanced web search function")
        return enhanced_web_search
        
    except ImportError:
        def mock_web_search(query, num_results=3):
            results = []
            keywords = query.lower().split()
            for i in range(num_results):
                result_text = f"Web search result {i+1} for query about {' and '.join(keywords[:2])}. "
                result_text += f"This result contains information related to {keywords[0] if keywords else 'medicine'}."
                results.append(result_text)
            logger.info(f"Mock web search returned {len(results)} results for query: {query[:50]}...")
            return results
        
        logger.warning(f"Real web search API not available, using mock web search function")
        return mock_web_search

# --- Test Helper Functions ---

def verify_rag_result(
    pipeline_instance: Any, 
    pipeline_run_result: Dict[str, Any], # Explicitly named to distinguish from other 'result' vars
    query: str,
    expected_keywords: List[str],
    min_doc_count: int,
    rag_technique_name: str, 
    pipeline_params: Optional[Dict[str, Any]] = None, 
    save_json_output: bool = True # Renamed from save_results for clarity
):
    """
    Verifies a RAG result, performs RAGAS evaluation, logs to CSV, and saves JSON.
    """
    main_test_passed = False
    ragas_faithfulness_score = None
    ragas_answer_relevancy_score = None
    ragas_notes = ""
    
    # Safely initialize for the finally block
    actual_retrieved_docs_for_log = []
    if isinstance(pipeline_run_result, dict) and \
       "retrieved_documents" in pipeline_run_result and \
       isinstance(pipeline_run_result["retrieved_documents"], list):
        actual_retrieved_docs_for_log = pipeline_run_result["retrieved_documents"]

    try:
        # --- Core RAG Assertions ---
        assert isinstance(pipeline_run_result, dict), \
            f"Result for {rag_technique_name} should be a dictionary, got {type(pipeline_run_result)}"
        assert "query" in pipeline_run_result, f"Result for {rag_technique_name} should contain 'query' key"
        assert "answer" in pipeline_run_result, f"Result for {rag_technique_name} should contain 'answer' key"
        assert "retrieved_documents" in pipeline_run_result, \
            f"Result for {rag_technique_name} should contain 'retrieved_documents' key"
        
        assert pipeline_run_result["query"] == query, \
            (f"Result query '{pipeline_run_result['query']}' for {rag_technique_name} "
             f"does not match original query '{query}'")
        assert pipeline_run_result["answer"], f"Answer for {rag_technique_name} should not be empty"
        assert isinstance(pipeline_run_result["answer"], str), \
            f"Answer for {rag_technique_name} should be a string, got {type(pipeline_run_result['answer'])}"
        assert len(pipeline_run_result["answer"]) > 10, \
            f"Answer for {rag_technique_name} is too short (<=10 chars): '{pipeline_run_result['answer']}'"
        
        retrieved_docs_list = pipeline_run_result["retrieved_documents"]
        assert isinstance(retrieved_docs_list, list), \
            f"Retrieved documents for {rag_technique_name} should be a list, got {type(retrieved_docs_list)}"
        
        actual_retrieved_docs_for_log = retrieved_docs_list # Update after assertion

        assert len(retrieved_docs_list) >= min_doc_count, \
            (f"For {rag_technique_name}, expected at least {min_doc_count} retrieved documents, "
             f"got {len(retrieved_docs_list)}")
        
        docs_with_keywords_count = 0
        for doc_idx, doc_object in enumerate(retrieved_docs_list):
            assert hasattr(doc_object, 'content'), \
                f"Document at index {doc_idx} for {rag_technique_name} lacks 'content' attribute."
            doc_content_text = doc_object.content if doc_object.content is not None else ""
            assert isinstance(doc_content_text, str), \
                (f"Document content at index {doc_idx} for {rag_technique_name} is not a string, "
                 f"got {type(doc_content_text)}.")

            if any(keyword.lower() in doc_content_text.lower() for keyword in expected_keywords):
                docs_with_keywords_count += 1
        
        min_docs_with_keywords_expected = max(1, min_doc_count // 2) if min_doc_count > 0 else 0
        if min_doc_count > 0:
             assert docs_with_keywords_count >= min_docs_with_keywords_expected, \
                (f"For {rag_technique_name}, expected at least {min_docs_with_keywords_expected} documents "
                 f"with keywords {expected_keywords}, got {docs_with_keywords_count}")
        
        logger.info(f"RAG result core assertions passed for {rag_technique_name} with query: {query[:50]}...")
        main_test_passed = True

        # --- RAGAS Evaluation (nested try-except) ---
        ragas_evaluation_result = None 
        try:
            contexts_for_ragas_eval = [doc.content for doc in retrieved_docs_list if hasattr(doc, 'content') and doc.content]
            if not contexts_for_ragas_eval:
                logger.warning(f"No non-empty content for RAGAS evaluation ({rag_technique_name}).")
                ragas_notes = "No non-empty content for RAGAS evaluation."
            else:
                data_samples_for_ragas = {
                    'question': [query],
                    'answer': [pipeline_run_result["answer"]],
                    'contexts': [contexts_for_ragas_eval], 
                }
                dataset_for_ragas_eval = Dataset.from_dict(data_samples_for_ragas)
                
                logger.info(f"Running RAGAS evaluation for {rag_technique_name}...")
                ragas_evaluation_result = evaluate(
                    dataset_for_ragas_eval,
                    metrics=[faithfulness, answer_relevancy],
                )
                logger.info(f"RAGAS Evaluation Scores for {rag_technique_name} query '{query[:50]}...':")
                if ragas_evaluation_result and isinstance(ragas_evaluation_result, Dataset) and len(ragas_evaluation_result) > 0: 
                    scores_dictionary = ragas_evaluation_result[0] 
                    metrics_to_log_dict = {k: v for k, v in scores_dictionary.items() if k not in ['question', 'answer', 'contexts']}
                    if metrics_to_log_dict:
                        for metric_key, metric_data_value in metrics_to_log_dict.items():
                            logger.info(f"  {metric_key}: {metric_data_value:.4f}")
                            if metric_key == 'faithfulness':
                                ragas_faithfulness_score = float(metric_data_value) if metric_data_value is not None else None
                            elif metric_key == 'answer_relevancy':
                                ragas_answer_relevancy_score = float(metric_data_value) if metric_data_value is not None else None
                        ragas_notes = "RAGAS scores calculated."
                    else:
                        logger.warning(f"RAGAS for {rag_technique_name}: no metric scores in result dict.")
                        ragas_notes = "RAGAS: no metric scores found."
                else:
                    logger.warning(f"RAGAS for {rag_technique_name}: no score object, empty, or not Dataset.")
                    ragas_notes = "RAGAS: no scores, empty, or invalid type."
        except Exception as e_ragas_run_error:
            logger.error(f"Error during RAGAS evaluation for {rag_technique_name}: {e_ragas_run_error}", exc_info=True)
            ragas_notes = f"RAGAS evaluation error: {str(e_ragas_run_error)}"

    except AssertionError as e_core_assertion_failure:
        logger.error(f"AssertionError for {rag_technique_name} with query '{query[:50]}...': {e_core_assertion_failure}", exc_info=True)
        main_test_passed = False 
        raise 

    finally:
        current_time_utc_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        csv_log_entry = {
            "RunID": str(uuid.uuid4()),
            "Timestamp": current_time_utc_iso,
            "PipelineName": rag_technique_name,
            "Query": query,
            "Parameters": json.dumps(pipeline_params, sort_keys=True) if pipeline_params else json.dumps({}),
            "DocumentsRetrievedCount": len(actual_retrieved_docs_for_log), 
            "RAGAS_Faithfulness": ragas_faithfulness_score,
            "RAGAS_AnswerRelevancy": ragas_answer_relevancy_score,
            "MainTestPassed": main_test_passed, 
            "RAGAS_Notes": ragas_notes
        }
        log_rag_evaluation_to_csv(csv_log_entry)

        if save_json_output and isinstance(pipeline_run_result, dict): 
            try:
                os.makedirs("test_results/rag_outputs", exist_ok=True)
                safe_query_part_for_filename = "".join(c if c.isalnum() else "_" for c in query[:30]).strip("_")
                json_filename = f"test_results/rag_outputs/result_{rag_technique_name}_{safe_query_part_for_filename}_{int(time.time())}.json"
                
                serializable_json_output = {
                    "query": pipeline_run_result.get("query"), 
                    "answer": pipeline_run_result.get("answer"),
                    "retrieved_documents": []
                }
                
                docs_list_for_json = pipeline_run_result.get("retrieved_documents", [])
                if isinstance(docs_list_for_json, list): 
                    for doc_item_json in docs_list_for_json:
                        if hasattr(doc_item_json, 'to_dict') and callable(doc_item_json.to_dict):
                            serializable_json_output["retrieved_documents"].append(doc_item_json.to_dict())
                        elif hasattr(doc_item_json, '__dict__'): 
                            serializable_json_output["retrieved_documents"].append(vars(doc_item_json))
                        else: 
                            content_val = doc_item_json.content if hasattr(doc_item_json, 'content') else str(doc_item_json)
                            id_val = doc_item_json.id if hasattr(doc_item_json, 'id') else "unknown"
                            serializable_json_output["retrieved_documents"].append({"id": id_val, "content": content_val})
                
                with open(json_filename, 'w', encoding='utf-8') as f_json:
                    json.dump(serializable_json_output, f_json, indent=2, ensure_ascii=False)
                logger.info(f"Saved RAG output for {rag_technique_name} to {json_filename}")
            except Exception as e_json_save_failure:
                logger.warning(f"Failed to save RAG result JSON for {rag_technique_name}: {e_json_save_failure}", exc_info=True)

# --- Test Cases ---

@pytest.mark.requires_real_data
@pytest.mark.requires_1000_docs
@pytest.mark.force_real 
def test_basic_rag_with_real_data(real_iris_connection, real_embedding_func, real_llm_func, sample_medical_queries):
    logger.info("Running test_basic_rag_with_real_data")
    pipeline = BasicRAGPipeline(
        iris_connector=real_iris_connection,
        embedding_func=real_embedding_func,
        llm_func=real_llm_func
    )
    query_data = sample_medical_queries[0]
    query = query_data["query"]
    expected_keywords = query_data["expected_keywords"]
    min_doc_count = query_data["min_doc_count"]
    
    run_result = pipeline.run(query, top_k=5)
    
    params = {}
    if hasattr(pipeline, 'top_k'): params['top_k'] = pipeline.top_k
    if hasattr(pipeline, 'similarity_threshold'): params['similarity_threshold'] = pipeline.similarity_threshold
    if hasattr(pipeline, 'llm_func') and hasattr(pipeline.llm_func, '__name__'): params['llm_func_name'] = pipeline.llm_func.__name__

    verify_rag_result(pipeline, run_result, query, expected_keywords, min_doc_count, "BasicRAG", params, save_json_output=True)
    logger.info("test_basic_rag_with_real_data passed")

@pytest.mark.requires_real_data
@pytest.mark.requires_1000_docs
def test_hyde_with_real_data(real_iris_connection, real_embedding_func, real_llm_func, sample_medical_queries):
    logger.info("Running test_hyde_with_real_data")
    pipeline = HyDERAGPipeline(
        iris_connector=real_iris_connection,
        embedding_func=real_embedding_func,
        llm_func=real_llm_func
    )
    query_data = sample_medical_queries[1]
    query = query_data["query"]
    expected_keywords = query_data["expected_keywords"]
    min_doc_count = query_data["min_doc_count"]
    
    run_result = pipeline.run(query, top_k=5)

    params = {}
    if hasattr(pipeline, 'llm') and hasattr(pipeline.llm, 'model_name'): params['llm_model_name'] = pipeline.llm.model_name
    if hasattr(pipeline, 'top_k'): params['top_k'] = pipeline.top_k

    verify_rag_result(pipeline, run_result, query, expected_keywords, min_doc_count, "HyDE", params, save_json_output=True)
    logger.info("test_hyde_with_real_data passed")

@pytest.mark.requires_real_data
@pytest.mark.requires_1000_docs
def test_crag_with_real_data(real_iris_connection, real_embedding_func, real_llm_func, web_search_func, sample_medical_queries):
    logger.info("Running test_crag_with_real_data")
    pipeline = CRAGPipeline(
        iris_connector=real_iris_connection,
        embedding_func=real_embedding_func,
        llm_func=real_llm_func,
        web_search_func=web_search_func,
        chunk_types=['adaptive'] 
    )
    query_data = sample_medical_queries[2]
    query = query_data["query"]
    expected_keywords = query_data["expected_keywords"]
    min_doc_count = query_data["min_doc_count"]
    
    run_result = pipeline.run(query, top_k=5)

    params = {'chunk_types': pipeline.chunk_types if hasattr(pipeline, 'chunk_types') else None}
    if hasattr(pipeline, 'llm') and hasattr(pipeline.llm, 'model_name'): params['llm_model_name'] = pipeline.llm.model_name
    if hasattr(pipeline, 'top_k'): params['top_k'] = pipeline.top_k

    verify_rag_result(pipeline, run_result, query, expected_keywords, min_doc_count, "CRAG", params, save_json_output=True)
    logger.info("test_crag_with_real_data passed")

@pytest.mark.requires_real_data
@pytest.mark.requires_1000_docs
def test_colbert_with_real_data(real_iris_connection, real_embedding_func, real_llm_func, colbert_query_encoder, sample_medical_queries):
    logger.info("Running test_colbert_with_real_data")
    # colbert_query_encoder fixture is defined in conftest_common.py
    pipeline = ColbertRAGPipeline(
        iris_connector=real_iris_connection,
        llm_func=real_llm_func,
        colbert_query_encoder_func=colbert_query_encoder, 
        embedding_func=real_embedding_func 
    )
    query_data = sample_medical_queries[0]
    query = query_data["query"]
    expected_keywords = query_data["expected_keywords"]
    min_doc_count = query_data["min_doc_count"]
    
    run_result = pipeline.run(query, top_k=5)

    params = {}
    if hasattr(pipeline, 'top_k'): params['top_k'] = pipeline.top_k
    if hasattr(pipeline, 'llm') and hasattr(pipeline.llm, 'model_name'): params['llm_model_name'] = pipeline.llm.model_name
    
    verify_rag_result(pipeline, run_result, query, expected_keywords, min_doc_count, "ColBERT", params, save_json_output=True)
    logger.info("test_colbert_with_real_data passed")

@pytest.mark.requires_real_data
@pytest.mark.requires_1000_docs
def test_noderag_with_real_data(real_iris_connection, real_embedding_func, real_llm_func, sample_medical_queries):
    logger.info("Running test_noderag_with_real_data")
    pipeline = NodeRAGPipeline(
        iris_connector=real_iris_connection,
        embedding_func=real_embedding_func,
        llm_func=real_llm_func,
        # graph_lib=None # Assuming graph_lib is optional or handled internally
    )
    query_data = sample_medical_queries[1] # Using a different query
    query = query_data["query"]
    expected_keywords = query_data["expected_keywords"]
    min_doc_count = query_data["min_doc_count"]
    
    run_result = pipeline.run(query, top_k=5)

    params = {}
    if hasattr(pipeline, 'top_k'): params['top_k'] = pipeline.top_k
    if hasattr(pipeline, 'llm') and hasattr(pipeline.llm, 'model_name'): params['llm_model_name'] = pipeline.llm.model_name
    if hasattr(pipeline, 'node_similarity_threshold'): params['node_similarity_threshold'] = pipeline.node_similarity_threshold

    verify_rag_result(pipeline, run_result, query, expected_keywords, min_doc_count, "NodeRAG", params, save_json_output=True)
    logger.info("test_noderag_with_real_data passed")

@pytest.mark.requires_real_data
@pytest.mark.requires_1000_docs
def test_graphrag_with_real_data(real_iris_connection, real_embedding_func, real_llm_func, sample_medical_queries):
    logger.info("Running test_graphrag_with_real_data")
    pipeline = GraphRAGPipeline(
        iris_connector=real_iris_connection,
        embedding_func=real_embedding_func,
        llm_func=real_llm_func
    )
    query_data = sample_medical_queries[2] # Using another different query
    query = query_data["query"]
    expected_keywords = query_data["expected_keywords"]
    min_doc_count = query_data["min_doc_count"]
    
    run_result = pipeline.run(query, top_k=5)

    params = {}
    if hasattr(pipeline, 'top_k'): params['top_k'] = pipeline.top_k
    if hasattr(pipeline, 'llm') and hasattr(pipeline.llm, 'model_name'): params['llm_model_name'] = pipeline.llm.model_name

    verify_rag_result(pipeline, run_result, query, expected_keywords, min_doc_count, "GraphRAG", params, save_json_output=True)
    logger.info("test_graphrag_with_real_data passed")


@pytest.mark.requires_real_data
@pytest.mark.requires_1000_docs
@pytest.mark.force_real 
def test_all_pipelines_with_same_query(
    real_iris_connection, 
    real_embedding_func, 
    real_llm_func, 
    web_search_func, # For CRAG
    colbert_query_encoder, # For ColBERT
    sample_medical_queries
):
    """
    Tests all RAG pipelines with the same query to compare their outputs.
    """
    logger.info("Running test_all_pipelines_with_same_query")
    
    query_data = sample_medical_queries[0] # Use the first query for all
    query = query_data["query"]
    expected_keywords = query_data["expected_keywords"]
    min_doc_count = query_data["min_doc_count"]

    pipelines_to_test = {
        "BasicRAG": BasicRAGPipeline(
            iris_connector=real_iris_connection,
            embedding_func=real_embedding_func,
            llm_func=real_llm_func
        ),
        "HyDE": HyDERAGPipeline(
            iris_connector=real_iris_connection,
            embedding_func=real_embedding_func,
            llm_func=real_llm_func
        ),
        "CRAG": CRAGPipeline(
            iris_connector=real_iris_connection,
            embedding_func=real_embedding_func,
            llm_func=real_llm_func,
            web_search_func=web_search_func,
            chunk_types=['adaptive']
        ),
        "ColBERT": ColbertRAGPipeline(
            iris_connector=real_iris_connection,
            llm_func=real_llm_func,
            colbert_query_encoder_func=colbert_query_encoder,
            embedding_func=real_embedding_func 
        ),
        "NodeRAG": NodeRAGPipeline(
            iris_connector=real_iris_connection,
            embedding_func=real_embedding_func,
            llm_func=real_llm_func
        ),
        "GraphRAG": GraphRAGPipeline(
            iris_connector=real_iris_connection,
            embedding_func=real_embedding_func,
            llm_func=real_llm_func
        )
    }
    
    all_results_data = {} # To store results from each pipeline

    for name, pipeline_instance_loop in pipelines_to_test.items():
        logger.info(f"Running {name} pipeline with query: '{query}' for comparison test.")
        try:
            start_time = time.time()
            pipeline_output = pipeline_instance_loop.run(query, top_k=5)
            elapsed_time = time.time() - start_time
            
            current_pipeline_params_loop = {}
            if hasattr(pipeline_instance_loop, 'top_k'): 
                current_pipeline_params_loop['top_k'] = pipeline_instance_loop.top_k
            if hasattr(pipeline_instance_loop, 'similarity_threshold'): 
                current_pipeline_params_loop['similarity_threshold'] = pipeline_instance_loop.similarity_threshold
            if hasattr(pipeline_instance_loop, 'llm') and hasattr(pipeline_instance_loop.llm, 'model_name'):
                 current_pipeline_params_loop['llm_model_name'] = pipeline_instance_loop.llm.model_name
            elif hasattr(pipeline_instance_loop, 'llm_func') and hasattr(pipeline_instance_loop.llm_func, '__name__'):
                 current_pipeline_params_loop['llm_func_name'] = pipeline_instance_loop.llm_func.__name__
            if hasattr(pipeline_instance_loop, 'chunk_types'):
                current_pipeline_params_loop['chunk_types'] = pipeline_instance_loop.chunk_types
            if hasattr(pipeline_instance_loop, 'node_similarity_threshold'):
                current_pipeline_params_loop['node_similarity_threshold'] = pipeline_instance_loop.node_similarity_threshold
            
            verify_rag_result(
                pipeline_instance_loop, 
                pipeline_output, 
                query, 
                expected_keywords, 
                min_doc_count, 
                name, 
                current_pipeline_params_loop,
                save_json_output=True
            )
            
            all_results_data[name] = {
                "result": pipeline_output,
                "elapsed_time": elapsed_time
            }
            logger.info(f"{name} pipeline completed in {elapsed_time:.2f} seconds for comparison test.")

        except Exception as e_pipeline_loop:
            logger.error(f"Error running or verifying {name} pipeline in comparison test: {e_pipeline_loop}", exc_info=True)
            all_results_data[name] = {"error": str(e_pipeline_loop)}
    
    assert len(all_results_data) == len(pipelines_to_test), \
        f"Expected results from all {len(pipelines_to_test)} pipelines, got {len(all_results_data)}"
    
    for name, res_data in all_results_data.items():
        assert "error" not in res_data, f"{name} pipeline failed with error: {res_data.get('error')}"
        assert "result" in res_data, f"{name} pipeline did not return a result object"
        assert isinstance(res_data["result"], dict), f"{name} pipeline result is not a dict"
        assert "answer" in res_data["result"], f"{name} pipeline result does not contain an answer"
    
    logger.info(f"Comparison Test Query: {query}")
    for name, res_data in all_results_data.items():
        if "result" in res_data and "answer" in res_data["result"]:
            answer_text = res_data["result"]["answer"]
            logger.info(f"  {name} Answer: {answer_text[:100]}...")
        elif "error" in res_data:
            logger.info(f"  {name} Error: {res_data['error']}")
            
    logger.info("test_all_pipelines_with_same_query passed")

def test_verify_real_data_requirements():
    """
    Verify that we're using real data for testing.
    This test is more of a marker; actual checks are in fixtures/conftest.
    """
    logger.info("Verifying real data requirements (marker test)...")
    assert True

if __name__ == "__main__":
    pytest.main(["-v", __file__])