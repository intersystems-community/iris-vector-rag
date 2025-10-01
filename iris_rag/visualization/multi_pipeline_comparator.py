"""
Multi-Pipeline Comparator

Provides comprehensive comparison capabilities between different RAG pipeline types,
including IRIS-Global-GraphRAG, HybridGraphRAG, and standard GraphRAG pipelines.

This enables side-by-side evaluation of different GraphRAG approaches to understand
their strengths, performance characteristics, and use case suitability.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from iris_rag import create_pipeline

from ..config.manager import ConfigurationManager
from ..core.exceptions import RAGException

logger = logging.getLogger(__name__)


class MultiPipelineComparator:
    """
    Multi-pipeline comparison system for evaluating different RAG approaches.

    Supports comparison between:
    - IRIS Global GraphRAG (globals-based, academic focus)
    - HybridGraphRAG (enterprise, multi-modal fusion)
    - Standard GraphRAG (traditional graph-based RAG)
    - BasicRAG (vector-only baseline)
    - LLM-only (no retrieval baseline)
    """

    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        self.config_manager = config_manager or ConfigurationManager()
        self.pipelines = {}
        self.pipeline_info = {}

        # Initialize available pipelines
        self._initialize_pipelines()

    def _initialize_pipelines(self):
        """Initialize all available pipelines for comparison."""
        pipeline_configs = [
            {
                "name": "IRIS Global GraphRAG",
                "type": "IRISGlobalGraphRAG",
                "description": "Academic papers with IRIS globals and 3D visualization",
                "features": ["globals_storage", "3d_visualization", "academic_focus"],
            },
            {
                "name": "Hybrid GraphRAG",
                "type": "HybridGraphRAG",
                "description": "Enterprise multi-modal fusion with 50x performance",
                "features": ["rrf_fusion", "hnsw_optimization", "enterprise_scale"],
            },
            {
                "name": "Standard GraphRAG",
                "type": "GraphRAG",
                "description": "Traditional graph-based RAG with entity relationships",
                "features": ["entity_extraction", "graph_traversal", "general_purpose"],
            },
            {
                "name": "Basic RAG",
                "type": "BasicRAG",
                "description": "Vector similarity search baseline",
                "features": ["vector_search", "simple", "fast"],
            },
        ]

        for config in pipeline_configs:
            try:
                logger.info(f"Initializing {config['name']} pipeline...")

                pipeline = create_pipeline(
                    pipeline_type=config["type"],
                    config_manager=self.config_manager,
                    validate_requirements=False,  # Skip validation for comparison
                )

                self.pipelines[config["type"]] = pipeline
                self.pipeline_info[config["type"]] = {
                    "name": config["name"],
                    "description": config["description"],
                    "features": config["features"],
                    "status": "available",
                    "info": (
                        pipeline.get_pipeline_info()
                        if hasattr(pipeline, "get_pipeline_info")
                        else {}
                    ),
                }

                logger.info(f"✅ {config['name']} pipeline initialized")

            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize {config['name']}: {e}")
                self.pipeline_info[config["type"]] = {
                    "name": config["name"],
                    "description": config["description"],
                    "features": config["features"],
                    "status": "unavailable",
                    "error": str(e),
                }

    def get_available_pipelines(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available pipelines."""
        return self.pipeline_info

    def compare_pipelines(
        self,
        query: str,
        pipeline_types: Optional[List[str]] = None,
        include_llm_baseline: bool = True,
        top_k: int = 5,
        parallel_execution: bool = True,
    ) -> Dict[str, Any]:
        """
        Compare multiple pipelines on the same query.

        Args:
            query: The query to test
            pipeline_types: List of pipeline types to compare (None = all available)
            include_llm_baseline: Whether to include LLM-only baseline
            top_k: Number of results for retrieval
            parallel_execution: Whether to run pipelines in parallel

        Returns:
            Dict containing results from all pipelines with performance metrics
        """
        start_time = time.time()

        # Determine which pipelines to run
        if pipeline_types is None:
            pipeline_types = list(self.pipelines.keys())

        # Filter to available pipelines
        available_types = [pt for pt in pipeline_types if pt in self.pipelines]

        if not available_types:
            raise RAGException("No available pipelines for comparison")

        logger.info(
            f"Comparing {len(available_types)} pipelines on query: '{query[:50]}...'"
        )

        results = {
            "query": query,
            "comparison_timestamp": time.time(),
            "pipelines": {},
            "performance_summary": {},
            "metadata": {
                "top_k": top_k,
                "parallel_execution": parallel_execution,
                "total_pipelines": len(available_types),
            },
        }

        # Add LLM baseline if requested
        if include_llm_baseline:
            results["pipelines"]["LLM_Baseline"] = self._get_llm_baseline(query)

        if parallel_execution:
            # Run pipelines in parallel
            results["pipelines"].update(
                self._run_pipelines_parallel(available_types, query, top_k)
            )
        else:
            # Run pipelines sequentially
            results["pipelines"].update(
                self._run_pipelines_sequential(available_types, query, top_k)
            )

        # Generate performance summary
        results["performance_summary"] = self._generate_performance_summary(
            results["pipelines"]
        )
        results["total_execution_time"] = time.time() - start_time

        return results

    def _run_pipelines_parallel(
        self, pipeline_types: List[str], query: str, top_k: int
    ) -> Dict[str, Any]:
        """Run pipelines in parallel for faster comparison."""
        results = {}

        with ThreadPoolExecutor(max_workers=len(pipeline_types)) as executor:
            # Submit all pipeline executions
            future_to_type = {
                executor.submit(
                    self._execute_pipeline, pipeline_type, query, top_k
                ): pipeline_type
                for pipeline_type in pipeline_types
            }

            # Collect results as they complete
            for future in as_completed(future_to_type):
                pipeline_type = future_to_type[future]
                try:
                    result = future.result(timeout=60)  # 60 second timeout
                    results[pipeline_type] = result
                except Exception as e:
                    logger.error(f"Pipeline {pipeline_type} failed: {e}")
                    results[pipeline_type] = {
                        "error": str(e),
                        "status": "failed",
                        "execution_time": None,
                    }

        return results

    def _run_pipelines_sequential(
        self, pipeline_types: List[str], query: str, top_k: int
    ) -> Dict[str, Any]:
        """Run pipelines sequentially for detailed monitoring."""
        results = {}

        for pipeline_type in pipeline_types:
            try:
                logger.info(f"Executing {pipeline_type}...")
                result = self._execute_pipeline(pipeline_type, query, top_k)
                results[pipeline_type] = result
                logger.info(
                    f"✅ {pipeline_type} completed in {result.get('execution_time', 0):.2f}s"
                )
            except Exception as e:
                logger.error(f"❌ {pipeline_type} failed: {e}")
                results[pipeline_type] = {
                    "error": str(e),
                    "status": "failed",
                    "execution_time": None,
                }

        return results

    def _execute_pipeline(
        self, pipeline_type: str, query: str, top_k: int
    ) -> Dict[str, Any]:
        """Execute a single pipeline and return formatted results."""
        pipeline = self.pipelines[pipeline_type]
        start_time = time.time()

        try:
            # Execute query based on pipeline type
            if pipeline_type == "IRISGlobalGraphRAG":
                # Use GraphRAG mode with visualization
                result = pipeline.query(
                    query, mode="graphrag", top_k=top_k, enable_visualization=True
                )
            else:
                # Standard query for other pipelines
                result = pipeline.query(query, top_k=top_k)

            execution_time = time.time() - start_time

            # Format result
            formatted_result = {
                "answer": result.get("answer", str(result)),
                "execution_time": execution_time,
                "status": "success",
                "pipeline_info": self.pipeline_info[pipeline_type],
                "metadata": {
                    "top_k": top_k,
                    "processing_time": result.get("processing_time", execution_time),
                },
            }

            # Add pipeline-specific data
            if pipeline_type == "IRISGlobalGraphRAG":
                if "graph_data" in result:
                    formatted_result["graph_data"] = result["graph_data"]
                    formatted_result["metadata"]["graph_nodes"] = len(
                        result["graph_data"].get("nodes", [])
                    )
                    formatted_result["metadata"]["graph_links"] = len(
                        result["graph_data"].get("links", [])
                    )

                if "retrieved_papers" in result:
                    formatted_result["retrieved_papers"] = result["retrieved_papers"]
                    formatted_result["metadata"]["retrieved_papers_count"] = len(
                        result["retrieved_papers"]
                    )

            elif pipeline_type == "HybridGraphRAG":
                # Add hybrid-specific metadata
                if isinstance(result, dict):
                    formatted_result["metadata"]["fusion_method"] = result.get(
                        "fusion_method", "unknown"
                    )
                    formatted_result["metadata"]["modalities_used"] = result.get(
                        "modalities_used", []
                    )

            return formatted_result

        except Exception as e:
            execution_time = time.time() - start_time
            raise RAGException(
                f"Pipeline execution failed after {execution_time:.2f}s: {e}"
            )

    def _get_llm_baseline(self, query: str) -> Dict[str, Any]:
        """Get LLM-only baseline response."""
        start_time = time.time()

        try:
            # Try to get LLM function from any available pipeline
            llm_func = None
            for pipeline in self.pipelines.values():
                if hasattr(pipeline, "llm_func") and pipeline.llm_func:
                    llm_func = pipeline.llm_func
                    break
                elif hasattr(pipeline, "global_graphrag_module"):
                    # Use IRIS Global GraphRAG's LLM function
                    llm_response = pipeline.global_graphrag_module.send_to_llm(
                        [
                            {
                                "role": "user",
                                "content": f"Answer this question concisely: {query}",
                            }
                        ]
                    )
                    answer = llm_response.choices[0].message.content

                    return {
                        "answer": answer,
                        "execution_time": time.time() - start_time,
                        "status": "success",
                        "pipeline_info": {
                            "name": "LLM Baseline",
                            "description": "Direct LLM response without retrieval",
                            "features": ["no_retrieval", "baseline"],
                        },
                        "metadata": {"retrieval_used": False, "model": "OpenAI GPT"},
                    }

            if llm_func:
                answer = llm_func(f"Answer this question concisely: {query}")
                return {
                    "answer": answer,
                    "execution_time": time.time() - start_time,
                    "status": "success",
                    "pipeline_info": {
                        "name": "LLM Baseline",
                        "description": "Direct LLM response without retrieval",
                        "features": ["no_retrieval", "baseline"],
                    },
                    "metadata": {"retrieval_used": False},
                }
            else:
                return {
                    "answer": "LLM baseline not available",
                    "execution_time": time.time() - start_time,
                    "status": "unavailable",
                    "error": "No LLM function available",
                }

        except Exception as e:
            return {
                "answer": f"LLM baseline error: {e}",
                "execution_time": time.time() - start_time,
                "status": "error",
                "error": str(e),
            }

    def _generate_performance_summary(
        self, pipeline_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate performance comparison summary."""
        summary = {
            "fastest_pipeline": None,
            "slowest_pipeline": None,
            "average_execution_time": 0,
            "success_rate": 0,
            "execution_times": {},
            "answer_lengths": {},
            "feature_comparison": {},
        }

        successful_results = {
            name: result
            for name, result in pipeline_results.items()
            if result.get("status") == "success"
            and result.get("execution_time") is not None
        }

        if not successful_results:
            return summary

        # Calculate execution times
        execution_times = {
            name: result["execution_time"]
            for name, result in successful_results.items()
        }

        summary["execution_times"] = execution_times
        summary["fastest_pipeline"] = min(execution_times, key=execution_times.get)
        summary["slowest_pipeline"] = max(execution_times, key=execution_times.get)
        summary["average_execution_time"] = sum(execution_times.values()) / len(
            execution_times
        )

        # Calculate answer lengths
        summary["answer_lengths"] = {
            name: len(str(result.get("answer", "")))
            for name, result in successful_results.items()
        }

        # Success rate
        total_pipelines = len(pipeline_results)
        successful_pipelines = len(successful_results)
        summary["success_rate"] = (
            successful_pipelines / total_pipelines if total_pipelines > 0 else 0
        )

        # Feature comparison
        summary["feature_comparison"] = {
            name: result.get("pipeline_info", {}).get("features", [])
            for name, result in pipeline_results.items()
        }

        return summary

    def generate_comparison_report(self, comparison_result: Dict[str, Any]) -> str:
        """Generate a human-readable comparison report."""
        report = []
        report.append("# Pipeline Comparison Report")
        report.append(f"**Query**: {comparison_result['query']}")
        report.append(
            f"**Total Execution Time**: {comparison_result['total_execution_time']:.2f}s"
        )
        report.append("")

        # Performance Summary
        perf = comparison_result["performance_summary"]
        report.append("## Performance Summary")
        report.append(
            f"- **Fastest Pipeline**: {perf['fastest_pipeline']} ({perf['execution_times'].get(perf['fastest_pipeline'], 0):.2f}s)"
        )
        report.append(
            f"- **Slowest Pipeline**: {perf['slowest_pipeline']} ({perf['execution_times'].get(perf['slowest_pipeline'], 0):.2f}s)"
        )
        report.append(
            f"- **Average Execution Time**: {perf['average_execution_time']:.2f}s"
        )
        report.append(f"- **Success Rate**: {perf['success_rate']:.1%}")
        report.append("")

        # Individual Results
        report.append("## Pipeline Results")
        for pipeline_name, result in comparison_result["pipelines"].items():
            report.append(f"### {pipeline_name}")

            if result.get("status") == "success":
                report.append(f"**Execution Time**: {result['execution_time']:.2f}s")
                report.append(
                    f"**Answer Length**: {len(result.get('answer', ''))} characters"
                )
                report.append(f"**Answer**: {result.get('answer', '')[:200]}...")

                if "metadata" in result:
                    metadata = result["metadata"]
                    if "graph_nodes" in metadata:
                        report.append(
                            f"**Graph Data**: {metadata['graph_nodes']} nodes, {metadata['graph_links']} links"
                        )
                    if "retrieved_papers_count" in metadata:
                        report.append(
                            f"**Retrieved Papers**: {metadata['retrieved_papers_count']}"
                        )
            else:
                report.append(f"**Status**: {result.get('status', 'unknown')}")
                if "error" in result:
                    report.append(f"**Error**: {result['error']}")

            report.append("")

        return "\n".join(report)
