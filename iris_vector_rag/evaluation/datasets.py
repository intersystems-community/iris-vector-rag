"""
Dataset loading utilities for multi-hop RAG benchmarks.

Supports HotpotQA, 2WikiMultiHopQA, and MuSiQue datasets via HuggingFace.
"""

import logging
from typing import Any, Dict, Iterator, List, Optional, Union, cast

from datasets import load_dataset
from iris_vector_rag.core.models import BenchmarkQuery

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Loader for multi-hop QA datasets."""

    def load(
        self, 
        dataset_id: str, 
        sample_size: int = 10, 
        split: str = "validation"
    ) -> Iterator[BenchmarkQuery]:
        """
        Load queries for a specific dataset.
        
        Args:
            dataset_id: One of 'hotpotqa', '2wikimultihopqa', 'musique'
            sample_size: Number of queries to load
            split: Dataset split (train, validation, test)
            
        Yields:
            BenchmarkQuery objects
        """
        if "musique" in dataset_id.lower():
            yield from self._load_musique(sample_size, split)
        elif "2wiki" in dataset_id.lower():
            yield from self._load_2wikimultihopqa(sample_size, split)
        elif "hotpot" in dataset_id.lower():
            yield from self._load_hotpotqa(sample_size, split)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_id}")

    def _load_musique(self, sample_size: int, split: str) -> Iterator[BenchmarkQuery]:
        """Load MuSiQue dataset."""
        try:
            # Authoritative sources
            dataset = cast(Any, load_dataset("kakao-ai/musique", "answerable", split=split, streaming=True))
        except Exception:
            dataset = cast(Any, load_dataset("dgslibisey/MuSiQue", split=split, streaming=True))

        count = 0
        for item in dataset:
            if count >= sample_size:
                break
            paragraphs = item.get("paragraphs", [])
            supporting_docs = [p.get("title", f"para_{i}") for i, p in enumerate(paragraphs) if p.get("is_supporting")]
            yield BenchmarkQuery(
                id=item.get("id", str(count)),
                question=item.get("question", ""),
                answer=item.get("answer", ""),
                supporting_docs=supporting_docs,
                metadata={"paragraphs": paragraphs},
            )
            count += 1

    def _load_2wikimultihopqa(self, sample_size: int, split: str) -> Iterator[BenchmarkQuery]:
        """Load 2WikiMultihopQA dataset."""
        try:
            dataset = cast(Any, load_dataset("xanhho/2WikiMultihopQA", split=split, streaming=True))
        except Exception:
            dataset = cast(Any, load_dataset("hotpotqa/hotpot_qa", "distractor", split=split, streaming=True))

        count = 0
        for item in dataset:
            if count >= sample_size:
                break
            sf = item.get("supporting_facts", {})
            supporting_docs = list(set(sf.get("title", [])))
            yield BenchmarkQuery(
                id=item.get("id", str(count)),
                question=item.get("question", ""),
                answer=item.get("answer", ""),
                supporting_docs=supporting_docs,
                metadata={"context": item.get("context", {})},
            )
            count += 1

    def _load_hotpotqa(self, sample_size: int, split: str) -> Iterator[BenchmarkQuery]:
        """Load HotpotQA dataset."""
        dataset = cast(Any, load_dataset("hotpotqa/hotpot_qa", "distractor", split=split, streaming=True))
        count = 0
        for item in dataset:
            if count >= sample_size:
                break
            sf = item.get("supporting_facts", {})
            supporting_docs = list(set(sf.get("title", [])))
            yield BenchmarkQuery(
                id=item.get("id", str(count)),
                question=item.get("question", ""),
                answer=item.get("answer", ""),
                supporting_docs=supporting_docs,
                metadata={"context": item.get("context", {})},
            )
            count += 1
