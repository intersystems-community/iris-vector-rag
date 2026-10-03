"""DSPy modules for IRIS RAG entity extraction.

.. warning::
    This module is **EXPERIMENTAL** and not recommended for production use.

    DSPy integration is under active development and tracked by spec 073.
    APIs may change significantly in future releases without notice.
    This module is available via the optional ``dspy`` extra:

        pip install iris-vector-rag[dspy]

    For production entity extraction, use the core services in
    :mod:`iris_vector_rag.services.entity_extraction`.
"""

import warnings

warnings.warn(
    "The dspy_modules package is experimental and not recommended for production. "
    "It is tracked by specs/073-dspy-optimization-layer.",
    FutureWarning,
    stacklevel=2,
)

from .entity_extraction_module import (
    EntityExtractionSignature,
    TrakCareEntityExtractionModule,
)

__all__ = ["TrakCareEntityExtractionModule", "EntityExtractionSignature"]
