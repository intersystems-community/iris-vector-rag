import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..core.models import Document

logger = logging.getLogger(__name__)


def convert_fusion_results_to_documents(
    fusion_results: List[Dict[str, Any]], get_content: Callable[[str], Optional[str]]
) -> List[Document]:
    """Convert fusion search results to Document objects"""
    documents: List[Document] = []
    for result in fusion_results:
        try:
            entity_id = (
                result.get("entity_id")
                if isinstance(result, dict)
                else (
                    result[0] if isinstance(result, (list, tuple)) and result else None
                )
            )
            if not entity_id:
                continue
            doc_content = get_content(str(entity_id))
            if doc_content:
                documents.append(
                    Document(
                        page_content=doc_content,
                        metadata={
                            "entity_id": str(entity_id),
                            "fusion_score": (
                                result.get("fusion_score")
                                if isinstance(result, dict)
                                else None
                            ),
                            "rank": (
                                result.get("rank") if isinstance(result, dict) else None
                            ),
                            "search_modes": (
                                result.get("search_modes")
                                if isinstance(result, dict)
                                else None
                            ),
                            "source": "hybrid_fusion",
                        },
                    )
                )
        except Exception as e:
            logger.warning(f"Could not convert fusion result {result}: {e}")
    return documents


def convert_rrf_results_to_documents(
    rrf_results: List[Tuple[str, float, float, float]],
    get_content: Callable[[str], Optional[str]],
) -> List[Document]:
    """Convert RRF search results to Document objects"""
    documents: List[Document] = []
    for item in rrf_results:
        try:
            if isinstance(item, (list, tuple)) and len(item) >= 4:
                entity_id, rrf_score, vector_score, text_score = (
                    item[0],
                    item[1],
                    item[2],
                    item[3],
                )
            else:
                logger.debug(f"Unexpected RRF result shape: {item}")
                continue
            doc_content = get_content(str(entity_id))
            if doc_content:
                documents.append(
                    Document(
                        page_content=doc_content,
                        metadata={
                            "entity_id": str(entity_id),
                            "rrf_score": float(rrf_score),
                            "vector_score": float(vector_score),
                            "text_score": float(text_score),
                            "source": "rrf_fusion",
                        },
                    )
                )
        except Exception as e:
            logger.warning(f"Could not convert RRF result {item}: {e}")
    return documents


def convert_text_results_to_documents(
    text_results: List[Tuple[str, float]], get_content: Callable[[str], Optional[str]]
) -> List[Document]:
    """Convert text search results to Document objects"""
    documents: List[Document] = []
    for item in text_results:
        try:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                entity_id, relevance_score = item[0], item[1]
            else:
                logger.debug(f"Unexpected text result shape: {item}")
                continue
            doc_content = get_content(str(entity_id))
            if doc_content:
                documents.append(
                    Document(
                        page_content=doc_content,
                        metadata={
                            "entity_id": str(entity_id),
                            "relevance_score": float(relevance_score),
                            "source": "enhanced_text",
                        },
                    )
                )
        except Exception as e:
            logger.warning(f"Could not convert text result {item}: {e}")
    return documents


def convert_vector_results_to_documents(
    vector_results: List[Tuple[str, float]], get_content: Callable[[str], Optional[str]]
) -> List[Document]:
    """Convert vector search results to Document objects"""
    documents: List[Document] = []
    for item in vector_results:
        try:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                entity_id, similarity_score = item[0], item[1]
            else:
                logger.debug(f"Unexpected vector result shape: {item}")
                continue
            doc_content = get_content(str(entity_id))
            if doc_content:
                documents.append(
                    Document(
                        page_content=doc_content,
                        metadata={
                            "entity_id": str(entity_id),
                            "similarity_score": float(similarity_score),
                            "source": "hnsw_vector",
                        },
                    )
                )
        except Exception as e:
            logger.warning(f"Could not convert vector result {item}: {e}")
    return documents
