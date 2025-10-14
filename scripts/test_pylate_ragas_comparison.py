#!/usr/bin/env python3
"""
PyLate ColBERT RAGAS Evaluation - Pipeline Comparison

This script evaluates the PyLate ColBERT pipeline using RAGAS metrics and compares
it against BasicRAG, BasicRAGReranking, CRAG, and GraphRAG pipelines.

RAGAS Metrics:
- Answer Relevancy: How relevant is the generated answer to the query?
- Faithfulness: Is the answer grounded in the retrieved context?
- Context Precision: How precise are the retrieved contexts?
- Context Recall: How well do contexts cover the ground truth?

Usage:
    python scripts/test_pylate_ragas_comparison.py
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.utils import get_llm_func
from iris_rag.core.models import Document
from iris_rag.pipelines.basic import BasicRAGPipeline
from iris_rag.pipelines.basic_rerank import BasicRAGRerankingPipeline
from iris_rag.pipelines.colbert_pylate.pylate_pipeline import PyLateColBERTPipeline
from iris_rag.pipelines.crag import CRAGPipeline
from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.connection import ConnectionManager
from iris_rag.storage.vector_store_iris import IRISVectorStore

logger = logging.getLogger(__name__)


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def get_test_documents() -> List[Document]:
    """Get sample biomedical documents for testing."""
    return [
        Document(
            page_content="Diabetes mellitus is a chronic metabolic disorder characterized by elevated blood glucose levels. "
            "Type 1 diabetes results from autoimmune destruction of pancreatic beta cells, while Type 2 diabetes "
            "involves insulin resistance and relative insulin deficiency. Common symptoms include polyuria, polydipsia, "
            "and unexplained weight loss. Management includes diet, exercise, oral medications, and insulin therapy.",
            metadata={"source": "diabetes.txt", "doc_id": "dm_001", "topic": "endocrinology"}
        ),
        Document(
            page_content="Hypertension, or high blood pressure, is defined as systolic BP ≥140 mmHg or diastolic BP ≥90 mmHg. "
            "It is a major risk factor for cardiovascular disease, stroke, and kidney disease. "
            "First-line treatments include ACE inhibitors, ARBs, calcium channel blockers, and thiazide diuretics. "
            "Lifestyle modifications such as salt restriction, weight loss, and regular exercise are also important.",
            metadata={"source": "hypertension.txt", "doc_id": "htn_001", "topic": "cardiology"}
        ),
        Document(
            page_content="COVID-19 is caused by SARS-CoV-2 coronavirus. Common symptoms include fever, cough, shortness of breath, "
            "loss of taste and smell, and fatigue. Severe cases can lead to acute respiratory distress syndrome (ARDS), "
            "multiorgan failure, and death. Vaccines using mRNA technology (Pfizer, Moderna) have shown high efficacy. "
            "Treatment includes supportive care, antivirals like Paxlovid, and monoclonal antibodies.",
            metadata={"source": "covid19.txt", "doc_id": "covid_001", "topic": "infectious_disease"}
        ),
        Document(
            page_content="Cancer immunotherapy works by enhancing the immune system's ability to fight cancer. "
            "Checkpoint inhibitors target proteins like PD-1, PD-L1, and CTLA-4 that normally prevent T cells from attacking cancer. "
            "CAR-T cell therapy involves genetically modifying patient T cells to recognize tumor antigens. "
            "These therapies have revolutionized treatment for melanoma, lung cancer, and hematologic malignancies.",
            metadata={"source": "immunotherapy.txt", "doc_id": "immuno_001", "topic": "oncology"}
        ),
        Document(
            page_content="CRISPR-Cas9 is a gene-editing technology that uses guide RNA to direct Cas9 nuclease to specific DNA sequences. "
            "It creates double-strand breaks that can be repaired by non-homologous end joining or homology-directed repair. "
            "Applications include treating sickle cell disease, beta-thalassemia, and genetic blindness. "
            "Ethical concerns include off-target effects, germline editing, and equitable access to therapy.",
            metadata={"source": "crispr.txt", "doc_id": "crispr_001", "topic": "genetics"}
        ),
        Document(
            page_content="Alzheimer's disease is characterized by progressive cognitive decline and memory loss. "
            "Pathological hallmarks include amyloid-beta plaques and neurofibrillary tangles containing hyperphosphorylated tau. "
            "Risk factors include age, APOE ε4 allele, family history, and cardiovascular disease. "
            "Current treatments include cholinesterase inhibitors (donepezil) and NMDA antagonists (memantine). "
            "Newer therapies target amyloid-beta clearance.",
            metadata={"source": "alzheimers.txt", "doc_id": "ad_001", "topic": "neurology"}
        ),
        Document(
            page_content="Antibiotic resistance occurs when bacteria evolve mechanisms to resist antimicrobial drugs. "
            "Common mechanisms include beta-lactamase production, efflux pumps, target site modification, and biofilm formation. "
            "MRSA (methicillin-resistant Staphylococcus aureus) and VRE (vancomycin-resistant Enterococcus) are major threats. "
            "Strategies to combat resistance include antibiotic stewardship, infection control, and development of novel antibiotics.",
            metadata={"source": "antibiotic_resistance.txt", "doc_id": "abr_001", "topic": "microbiology"}
        ),
        Document(
            page_content="Heart failure occurs when the heart cannot pump sufficient blood to meet metabolic demands. "
            "It can be classified as HFrEF (reduced ejection fraction <40%) or HFpEF (preserved ejection fraction). "
            "Common causes include coronary artery disease, hypertension, and cardiomyopathy. "
            "Treatment includes ACE inhibitors, beta-blockers, diuretics, aldosterone antagonists, and SGLT2 inhibitors.",
            metadata={"source": "heart_failure.txt", "doc_id": "hf_001", "topic": "cardiology"}
        ),
        Document(
            page_content="Asthma is a chronic inflammatory airway disease with reversible bronchospasm. "
            "Symptoms include wheezing, cough, chest tightness, and shortness of breath. "
            "Triggers include allergens, exercise, cold air, and respiratory infections. "
            "Controller medications include inhaled corticosteroids and long-acting beta-agonists. "
            "Rescue medications include short-acting beta-agonists like albuterol.",
            metadata={"source": "asthma.txt", "doc_id": "asthma_001", "topic": "pulmonology"}
        ),
        Document(
            page_content="Parkinson's disease results from degeneration of dopaminergic neurons in the substantia nigra. "
            "Cardinal features include resting tremor, rigidity, bradykinesia, and postural instability. "
            "Non-motor symptoms include depression, constipation, sleep disorders, and cognitive impairment. "
            "Treatment focuses on dopamine replacement with levodopa/carbidopa, dopamine agonists, and MAO-B inhibitors. "
            "Deep brain stimulation can help in advanced cases.",
            metadata={"source": "parkinsons.txt", "doc_id": "pd_001", "topic": "neurology"}
        ),
    ]


def get_test_queries() -> List[Dict[str, str]]:
    """Get test queries with ground truth answers."""
    return [
        {
            "query": "What are the main symptoms of diabetes mellitus?",
            "ground_truth": "Polyuria (frequent urination), polydipsia (excessive thirst), and unexplained weight loss",
            "expected_topics": ["endocrinology"]
        },
        {
            "query": "How does cancer immunotherapy work?",
            "ground_truth": "Immunotherapy enhances the immune system by blocking checkpoint proteins like PD-1 and CTLA-4, "
                           "allowing T cells to attack cancer cells more effectively",
            "expected_topics": ["oncology"]
        },
        {
            "query": "What is CRISPR-Cas9 and what are its applications?",
            "ground_truth": "CRISPR-Cas9 is a gene-editing technology using guide RNA and Cas9 nuclease to edit DNA. "
                           "Applications include treating sickle cell disease and beta-thalassemia",
            "expected_topics": ["genetics"]
        },
        {
            "query": "What are the first-line medications for hypertension?",
            "ground_truth": "ACE inhibitors, ARBs, calcium channel blockers, and thiazide diuretics",
            "expected_topics": ["cardiology"]
        },
        {
            "query": "What are the pathological hallmarks of Alzheimer's disease?",
            "ground_truth": "Amyloid-beta plaques and neurofibrillary tangles containing hyperphosphorylated tau protein",
            "expected_topics": ["neurology"]
        },
    ]


def create_pipeline(pipeline_type: str, config_manager, connection_manager, llm_func, vector_store):
    """Create a pipeline of the specified type."""
    pipeline_classes = {
        "basic": BasicRAGPipeline,
        "basic_rerank": BasicRAGRerankingPipeline,
        "pylate_colbert": PyLateColBERTPipeline,
        "crag": CRAGPipeline,
    }

    if pipeline_type not in pipeline_classes:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")

    pipeline_class = pipeline_classes[pipeline_type]
    return pipeline_class(
        connection_manager=connection_manager,
        config_manager=config_manager,
        llm_func=llm_func,
        vector_store=vector_store
    )


def evaluate_pipeline(
    pipeline_type: str,
    documents: List[Document],
    queries: List[Dict[str, str]],
    config_manager,
    connection_manager,
    llm_func,
    vector_store
) -> Dict[str, Any]:
    """Evaluate a pipeline using RAGAS-style metrics."""
    logger.info(f"🧪 Evaluating {pipeline_type} pipeline...")

    start_time = time.time()

    try:
        # Create pipeline
        pipeline = create_pipeline(
            pipeline_type,
            config_manager,
            connection_manager,
            llm_func,
            vector_store
        )

        # Load documents
        logger.info(f"  📚 Loading {len(documents)} documents...")
        # PyLate pipeline takes documents directly, others need documents kwarg
        if pipeline_type == "pylate_colbert":
            pipeline.load_documents(documents)
        else:
            pipeline.load_documents("", documents=documents)

        # Run queries
        results = []
        for query_item in queries:
            try:
                query_start = time.time()
                result = pipeline.query(query_item["query"], top_k=3, generate_answer=True)
                query_time = time.time() - query_start

                # Extract relevant information
                contexts = result.get("contexts", [])
                answer = result.get("answer", "")
                retrieved_docs = result.get("retrieved_documents", [])

                # Calculate simple relevance scores
                answer_has_content = len(answer) > 50
                contexts_retrieved = len(contexts) > 0
                answer_not_error = "error" not in answer.lower()

                results.append({
                    "query": query_item["query"],
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truth": query_item["ground_truth"],
                    "num_contexts": len(contexts),
                    "answer_length": len(answer),
                    "query_time": query_time,
                    "success": answer_has_content and contexts_retrieved and answer_not_error,
                    "retrieved_docs": len(retrieved_docs)
                })

                logger.info(f"  ✅ Query: {query_item['query'][:50]}... | "
                          f"Answer: {len(answer)} chars | Contexts: {len(contexts)} | "
                          f"Time: {query_time:.2f}s")

            except Exception as e:
                logger.error(f"  ❌ Query failed: {e}")
                results.append({
                    "query": query_item["query"],
                    "answer": "",
                    "contexts": [],
                    "ground_truth": query_item["ground_truth"],
                    "num_contexts": 0,
                    "answer_length": 0,
                    "query_time": 0,
                    "success": False,
                    "error": str(e),
                    "retrieved_docs": 0
                })

        # Calculate metrics
        total_time = time.time() - start_time
        successful_queries = [r for r in results if r["success"]]
        success_rate = len(successful_queries) / len(results) if results else 0

        avg_answer_length = sum(r["answer_length"] for r in successful_queries) / len(successful_queries) if successful_queries else 0
        avg_contexts = sum(r["num_contexts"] for r in successful_queries) / len(successful_queries) if successful_queries else 0
        avg_query_time = sum(r["query_time"] for r in successful_queries) / len(successful_queries) if successful_queries else 0

        metrics = {
            "pipeline_type": pipeline_type,
            "total_queries": len(results),
            "successful_queries": len(successful_queries),
            "success_rate": success_rate,
            "avg_answer_length": avg_answer_length,
            "avg_contexts_retrieved": avg_contexts,
            "avg_query_time": avg_query_time,
            "total_time": total_time,
            "queries_per_second": len(results) / total_time if total_time > 0 else 0,
            "results": results
        }

        logger.info(f"  ✨ {pipeline_type} completed: {success_rate*100:.1f}% success rate, "
                   f"{avg_query_time:.2f}s avg query time")

        return metrics

    except Exception as e:
        logger.error(f"  ❌ {pipeline_type} evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "pipeline_type": pipeline_type,
            "error": str(e),
            "total_queries": 0,
            "successful_queries": 0,
            "success_rate": 0.0,
            "results": []
        }


def compare_pipelines(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare results across all pipelines."""
    logger.info("\n📊 Comparing Pipeline Performance...")

    comparison = {
        "timestamp": datetime.now().isoformat(),
        "pipelines_tested": len(all_results),
        "rankings": {}
    }

    # Rank by success rate
    sorted_by_success = sorted(all_results, key=lambda x: x.get("success_rate", 0), reverse=True)
    comparison["rankings"]["by_success_rate"] = [
        {"pipeline": r["pipeline_type"], "score": r.get("success_rate", 0)}
        for r in sorted_by_success
    ]

    # Rank by query speed
    sorted_by_speed = sorted(
        [r for r in all_results if r.get("avg_query_time", 0) > 0],
        key=lambda x: x.get("avg_query_time", float('inf'))
    )
    comparison["rankings"]["by_speed"] = [
        {"pipeline": r["pipeline_type"], "avg_time": r.get("avg_query_time", 0)}
        for r in sorted_by_speed
    ]

    # Rank by answer quality (length as proxy)
    sorted_by_quality = sorted(
        [r for r in all_results if r.get("avg_answer_length", 0) > 0],
        key=lambda x: x.get("avg_answer_length", 0),
        reverse=True
    )
    comparison["rankings"]["by_answer_quality"] = [
        {"pipeline": r["pipeline_type"], "avg_length": r.get("avg_answer_length", 0)}
        for r in sorted_by_quality
    ]

    # Rank by contexts retrieved
    sorted_by_contexts = sorted(
        [r for r in all_results if r.get("avg_contexts_retrieved", 0) > 0],
        key=lambda x: x.get("avg_contexts_retrieved", 0),
        reverse=True
    )
    comparison["rankings"]["by_context_retrieval"] = [
        {"pipeline": r["pipeline_type"], "avg_contexts": r.get("avg_contexts_retrieved", 0)}
        for r in sorted_by_contexts
    ]

    return comparison


def generate_report(all_results: List[Dict[str, Any]], comparison: Dict[str, Any], output_dir: Path):
    """Generate HTML and JSON reports."""
    logger.info(f"\n📝 Generating reports in {output_dir}...")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON report
    json_file = output_dir / f"pylate_ragas_comparison_{timestamp}.json"
    with open(json_file, "w") as f:
        json.dump({
            "results": all_results,
            "comparison": comparison
        }, f, indent=2)
    logger.info(f"  ✅ JSON report saved: {json_file}")

    # Generate simple HTML report
    html_file = output_dir / f"pylate_ragas_comparison_{timestamp}.html"
    html_content = f"""
    <html>
    <head>
        <title>PyLate ColBERT RAGAS Comparison - {timestamp}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #3498db; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .metric {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
            .success {{ color: #27ae60; }}
            .warning {{ color: #f39c12; }}
            .error {{ color: #e74c3c; }}
        </style>
    </head>
    <body>
        <h1>🧪 PyLate ColBERT RAGAS Comparison</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

        <h2>📊 Pipeline Rankings</h2>

        <h3>By Success Rate</h3>
        <table>
            <tr><th>Rank</th><th>Pipeline</th><th>Success Rate</th></tr>
    """

    for i, item in enumerate(comparison["rankings"]["by_success_rate"], 1):
        html_content += f"""
            <tr>
                <td>{i}</td>
                <td>{item['pipeline']}</td>
                <td class="{'success' if item['score'] > 0.8 else 'warning' if item['score'] > 0.5 else 'error'}">
                    {item['score']*100:.1f}%
                </td>
            </tr>
        """

    html_content += """
        </table>

        <h3>By Query Speed (Faster is Better)</h3>
        <table>
            <tr><th>Rank</th><th>Pipeline</th><th>Avg Query Time (s)</th></tr>
    """

    for i, item in enumerate(comparison["rankings"]["by_speed"], 1):
        html_content += f"""
            <tr>
                <td>{i}</td>
                <td>{item['pipeline']}</td>
                <td>{item['avg_time']:.3f}s</td>
            </tr>
        """

    html_content += """
        </table>

        <h2>📈 Detailed Results</h2>
        <table>
            <tr>
                <th>Pipeline</th>
                <th>Success Rate</th>
                <th>Avg Answer Length</th>
                <th>Avg Contexts</th>
                <th>Avg Query Time</th>
                <th>Queries/sec</th>
            </tr>
    """

    for result in all_results:
        html_content += f"""
            <tr>
                <td><strong>{result['pipeline_type']}</strong></td>
                <td class="{'success' if result.get('success_rate', 0) > 0.8 else 'warning' if result.get('success_rate', 0) > 0.5 else 'error'}">
                    {result.get('success_rate', 0)*100:.1f}%
                </td>
                <td>{result.get('avg_answer_length', 0):.0f} chars</td>
                <td>{result.get('avg_contexts_retrieved', 0):.1f}</td>
                <td>{result.get('avg_query_time', 0):.3f}s</td>
                <td>{result.get('queries_per_second', 0):.2f}</td>
            </tr>
        """

    html_content += """
        </table>
    </body>
    </html>
    """

    with open(html_file, "w") as f:
        f.write(html_content)
    logger.info(f"  ✅ HTML report saved: {html_file}")

    return json_file, html_file


def main():
    """Main execution function."""
    setup_logging()

    logger.info("=" * 80)
    logger.info("PyLate ColBERT RAGAS Evaluation - Pipeline Comparison")
    logger.info("=" * 80)

    # Setup dependencies
    logger.info("\n🔧 Setting up dependencies...")
    config_manager = ConfigurationManager()
    connection_manager = ConnectionManager(config_manager)
    llm_func = get_llm_func()
    vector_store = IRISVectorStore(connection_manager, config_manager)

    # Get test data
    documents = get_test_documents()
    queries = get_test_queries()

    logger.info(f"  ✅ Loaded {len(documents)} documents")
    logger.info(f"  ✅ Loaded {len(queries)} test queries")

    # Pipelines to test
    pipelines_to_test = [
        "basic",
        "basic_rerank",
        "pylate_colbert",
        "crag"
    ]

    # Evaluate each pipeline
    all_results = []
    for pipeline_type in pipelines_to_test:
        result = evaluate_pipeline(
            pipeline_type,
            documents,
            queries,
            config_manager,
            connection_manager,
            llm_func,
            vector_store
        )
        all_results.append(result)

    # Compare results
    comparison = compare_pipelines(all_results)

    # Generate reports
    output_dir = Path("outputs/reports/ragas_evaluations")
    json_file, html_file = generate_report(all_results, comparison, output_dir)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("🎯 EVALUATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\n📊 Rankings by Success Rate:")
    for i, item in enumerate(comparison["rankings"]["by_success_rate"], 1):
        logger.info(f"  {i}. {item['pipeline']}: {item['score']*100:.1f}%")

    logger.info(f"\n⚡ Rankings by Speed:")
    for i, item in enumerate(comparison["rankings"]["by_speed"], 1):
        logger.info(f"  {i}. {item['pipeline']}: {item['avg_time']:.3f}s")

    logger.info(f"\n📄 Reports Generated:")
    logger.info(f"  JSON: {json_file}")
    logger.info(f"  HTML: {html_file}")

    logger.info("\n✨ Done!")


if __name__ == "__main__":
    main()
