"""
Parallel Processing for GraphRAG Performance Optimization.

This module provides concurrent execution capabilities for GraphRAG operations,
enabling 8-16 parallel operations for sub-200ms response times. Implements
async document retrieval, parallel entity extraction, and concurrent graph traversal.
"""

import asyncio
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from ..core.models import Document

logger = logging.getLogger(__name__)


@dataclass
class ParallelExecutionResult:
    """Result container for parallel execution operations."""

    success: bool
    result: Any
    execution_time: float
    error: Optional[Exception] = None
    worker_id: Optional[str] = None


class GraphRAGParallelProcessor:
    """
    High-performance parallel processor for GraphRAG operations.

    Features:
    - Configurable thread pools for different operation types
    - Parallel entity extraction during document loading
    - Concurrent graph traversal across multiple paths
    - Async document retrieval with batching
    - Performance monitoring and load balancing
    """

    def __init__(
        self,
        max_workers: int = 16,
        io_workers: int = 8,
        entity_workers: int = 4,
        graph_workers: int = 8,
        batch_size: int = 10,
    ):
        """Initialize parallel processor with optimized thread pools."""
        self.max_workers = max_workers
        self.io_workers = io_workers
        self.entity_workers = entity_workers
        self.graph_workers = graph_workers
        self.batch_size = batch_size

        # Initialize thread pools for different operation types
        self.io_executor = ThreadPoolExecutor(
            max_workers=self.io_workers, thread_name_prefix="graphrag_io"
        )
        self.entity_executor = ThreadPoolExecutor(
            max_workers=self.entity_workers, thread_name_prefix="graphrag_entity"
        )
        self.graph_executor = ThreadPoolExecutor(
            max_workers=self.graph_workers, thread_name_prefix="graphrag_graph"
        )

        # Performance monitoring
        self._execution_stats = {
            "total_operations": 0,
            "parallel_operations": 0,
            "total_execution_time": 0.0,
            "avg_execution_time": 0.0,
            "max_concurrency_used": 0,
        }
        self._stats_lock = threading.Lock()

        logger.info(
            f"GraphRAG parallel processor initialized: {max_workers} max workers"
        )

    def parallel_entity_extraction(
        self,
        documents: List[Document],
        extraction_func: Callable[[Document], Dict[str, Any]],
    ) -> List[ParallelExecutionResult]:
        """
        Extract entities from multiple documents in parallel.

        Args:
            documents: List of documents to process
            extraction_func: Function that extracts entities from a single document

        Returns:
            List of extraction results with timing and error information
        """
        start_time = time.perf_counter()
        results = []

        logger.info(
            f"Starting parallel entity extraction for {len(documents)} documents"
        )

        # Submit all extraction tasks
        future_to_doc = {}
        for i, doc in enumerate(documents):
            future = self.entity_executor.submit(
                self._timed_execution,
                extraction_func,
                doc,
                f"entity_worker_{i % self.entity_workers}",
            )
            future_to_doc[future] = doc

        # Collect results as they complete
        completed_count = 0
        for future in as_completed(future_to_doc):
            doc = future_to_doc[future]
            try:
                result = future.result()
                results.append(result)
                completed_count += 1

                if completed_count % 10 == 0:
                    logger.debug(
                        f"Entity extraction progress: {completed_count}/{len(documents)}"
                    )

            except Exception as e:
                logger.error(f"Entity extraction failed for document {doc.id}: {e}")
                results.append(
                    ParallelExecutionResult(
                        success=False, result=None, execution_time=0.0, error=e
                    )
                )

        total_time = time.perf_counter() - start_time
        self._update_stats(len(documents), total_time)

        successful_results = [r for r in results if r.success]
        logger.info(
            f"Parallel entity extraction completed: {len(successful_results)}/{len(documents)} successful in {total_time:.2f}s"
        )

        return results

    def concurrent_graph_traversal(
        self,
        seed_entity_groups: List[List[Tuple[str, str, float]]],
        traversal_func: Callable[[List[Tuple[str, str, float]]], Set[str]],
        max_depth: int = 2,
    ) -> List[ParallelExecutionResult]:
        """
        Perform graph traversal for multiple seed entity groups concurrently.

        Args:
            seed_entity_groups: Groups of seed entities to traverse from
            traversal_func: Function that performs graph traversal
            max_depth: Maximum traversal depth

        Returns:
            List of traversal results with relevant entities
        """
        start_time = time.perf_counter()
        results = []

        logger.info(
            f"Starting concurrent graph traversal for {len(seed_entity_groups)} groups"
        )

        # Submit traversal tasks for each group
        future_to_group = {}
        for i, seed_group in enumerate(seed_entity_groups):
            future = self.graph_executor.submit(
                self._timed_execution,
                traversal_func,
                seed_group,
                f"graph_worker_{i % self.graph_workers}",
            )
            future_to_group[future] = seed_group

        # Collect results
        for future in as_completed(future_to_group):
            seed_group = future_to_group[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Graph traversal failed for seed group: {e}")
                results.append(
                    ParallelExecutionResult(
                        success=False, result=set(), execution_time=0.0, error=e
                    )
                )

        total_time = time.perf_counter() - start_time
        self._update_stats(len(seed_entity_groups), total_time)

        successful_results = [r for r in results if r.success]
        logger.info(
            f"Concurrent graph traversal completed: {len(successful_results)}/{len(seed_entity_groups)} successful in {total_time:.2f}s"
        )

        return results

    def parallel_document_retrieval(
        self,
        entity_id_batches: List[Set[str]],
        retrieval_func: Callable[[Set[str], int], List[Document]],
        top_k: int = 10,
    ) -> List[ParallelExecutionResult]:
        """
        Retrieve documents for multiple entity ID batches in parallel.

        Args:
            entity_id_batches: Batches of entity IDs to retrieve documents for
            retrieval_func: Function that retrieves documents for entity IDs
            top_k: Maximum documents to retrieve per batch

        Returns:
            List of document retrieval results
        """
        start_time = time.perf_counter()
        results = []

        logger.info(
            f"Starting parallel document retrieval for {len(entity_id_batches)} batches"
        )

        # Submit retrieval tasks
        future_to_batch = {}
        for i, entity_batch in enumerate(entity_id_batches):
            future = self.io_executor.submit(
                self._timed_execution,
                lambda batch: retrieval_func(batch, top_k),
                entity_batch,
                f"io_worker_{i % self.io_workers}",
            )
            future_to_batch[future] = entity_batch

        # Collect results
        for future in as_completed(future_to_batch):
            entity_batch = future_to_batch[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Document retrieval failed for batch: {e}")
                results.append(
                    ParallelExecutionResult(
                        success=False, result=[], execution_time=0.0, error=e
                    )
                )

        total_time = time.perf_counter() - start_time
        self._update_stats(len(entity_id_batches), total_time)

        successful_results = [r for r in results if r.success]
        logger.info(
            f"Parallel document retrieval completed: {len(successful_results)}/{len(entity_id_batches)} successful in {total_time:.2f}s"
        )

        return results

    def batch_parallel_execution(
        self,
        items: List[Any],
        processing_func: Callable[[Any], Any],
        batch_size: Optional[int] = None,
        executor_type: str = "io",
    ) -> List[ParallelExecutionResult]:
        """
        Execute a function on batches of items in parallel.

        Args:
            items: Items to process
            processing_func: Function to apply to each item
            batch_size: Size of each batch (uses default if None)
            executor_type: Type of executor to use ("io", "entity", or "graph")

        Returns:
            List of execution results
        """
        if batch_size is None:
            batch_size = self.batch_size

        # Select appropriate executor
        executor_map = {
            "io": self.io_executor,
            "entity": self.entity_executor,
            "graph": self.graph_executor,
        }
        executor = executor_map.get(executor_type, self.io_executor)

        # Create batches
        batches = []
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            batches.append(batch)

        start_time = time.perf_counter()
        logger.info(
            f"Starting batch parallel execution: {len(batches)} batches of {batch_size} items"
        )

        # Submit batch processing tasks
        future_to_batch = {}
        for i, batch in enumerate(batches):
            future = executor.submit(
                self._batch_processor,
                batch,
                processing_func,
                f"{executor_type}_batch_{i}",
            )
            future_to_batch[future] = batch

        # Collect results
        results = []
        for future in as_completed(future_to_batch):
            try:
                batch_results = future.result()
                results.extend(batch_results)
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                # Create error results for the batch
                batch = future_to_batch[future]
                for _ in batch:
                    results.append(
                        ParallelExecutionResult(
                            success=False, result=None, execution_time=0.0, error=e
                        )
                    )

        total_time = time.perf_counter() - start_time
        self._update_stats(len(batches), total_time)

        successful_results = [r for r in results if r.success]
        logger.info(
            f"Batch parallel execution completed: {len(successful_results)}/{len(items)} successful in {total_time:.2f}s"
        )

        return results

    def async_pipeline_execution(
        self,
        pipeline_stages: List[Tuple[str, Callable, List[Any]]],
        dependency_map: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, List[ParallelExecutionResult]]:
        """
        Execute a multi-stage pipeline with dependency management.

        Args:
            pipeline_stages: List of (stage_name, function, inputs) tuples
            dependency_map: Map of stage dependencies (stage -> [dependent_stages])

        Returns:
            Dictionary mapping stage names to their results
        """
        results = {}
        completed_stages = set()

        logger.info(
            f"Starting async pipeline execution with {len(pipeline_stages)} stages"
        )

        # Execute stages based on dependencies
        for stage_name, stage_func, stage_inputs in pipeline_stages:
            # Check if dependencies are satisfied
            if dependency_map and stage_name in dependency_map:
                dependencies = dependency_map[stage_name]
                if not all(dep in completed_stages for dep in dependencies):
                    logger.warning(f"Dependencies not satisfied for stage {stage_name}")
                    continue

            # Execute stage
            stage_results = self.batch_parallel_execution(
                stage_inputs, stage_func, executor_type="io"
            )

            results[stage_name] = stage_results
            completed_stages.add(stage_name)

            logger.info(f"Stage {stage_name} completed: {len(stage_results)} results")

        return results

    def _timed_execution(
        self, func: Callable, *args, worker_id: str = "unknown"
    ) -> ParallelExecutionResult:
        """Execute function with timing and error handling."""
        start_time = time.perf_counter()

        try:
            result = func(*args)
            execution_time = time.perf_counter() - start_time

            return ParallelExecutionResult(
                success=True,
                result=result,
                execution_time=execution_time,
                worker_id=worker_id,
            )

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            logger.error(f"Execution failed in {worker_id}: {e}")

            return ParallelExecutionResult(
                success=False,
                result=None,
                execution_time=execution_time,
                error=e,
                worker_id=worker_id,
            )

    def _batch_processor(
        self, batch: List[Any], processing_func: Callable[[Any], Any], worker_id: str
    ) -> List[ParallelExecutionResult]:
        """Process a batch of items sequentially within a worker."""
        results = []

        for item in batch:
            result = self._timed_execution(processing_func, item, worker_id)
            results.append(result)

        return results

    def _update_stats(self, operation_count: int, total_time: float) -> None:
        """Update execution statistics."""
        with self._stats_lock:
            self._execution_stats["total_operations"] += operation_count
            self._execution_stats["parallel_operations"] += 1
            self._execution_stats["total_execution_time"] += total_time

            # Update average
            total_ops = self._execution_stats["total_operations"]
            total_time_all = self._execution_stats["total_execution_time"]
            self._execution_stats["avg_execution_time"] = (
                total_time_all / total_ops if total_ops > 0 else 0
            )

            # Update max concurrency
            current_concurrency = operation_count
            if current_concurrency > self._execution_stats["max_concurrency_used"]:
                self._execution_stats["max_concurrency_used"] = current_concurrency

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the parallel processor."""
        with self._stats_lock:
            stats = self._execution_stats.copy()

        # Add current thread pool status
        stats["thread_pools"] = {
            "io_executor": {
                "max_workers": self.io_workers,
                "active_threads": getattr(
                    self.io_executor, "_threads", set()
                ).__len__(),
            },
            "entity_executor": {
                "max_workers": self.entity_workers,
                "active_threads": getattr(
                    self.entity_executor, "_threads", set()
                ).__len__(),
            },
            "graph_executor": {
                "max_workers": self.graph_workers,
                "active_threads": getattr(
                    self.graph_executor, "_threads", set()
                ).__len__(),
            },
        }

        return stats

    def shutdown(self) -> None:
        """Shutdown all thread pools gracefully."""
        logger.info("Shutting down parallel processor...")

        self.io_executor.shutdown(wait=True)
        self.entity_executor.shutdown(wait=True)
        self.graph_executor.shutdown(wait=True)

        logger.info("Parallel processor shutdown completed")
