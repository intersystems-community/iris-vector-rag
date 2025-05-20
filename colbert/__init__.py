"""ColBERT package initialization."""

from .pipeline import ColbertRAGPipeline # Corrected class name
from common.utils import Document # Document is in common.utils, not pipeline

__all__ = ["ColbertRAGPipeline", "Document"] # Corrected class name in __all__
