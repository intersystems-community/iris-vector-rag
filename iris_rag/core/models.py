import uuid
from dataclasses import dataclass, field
from typing import Dict, Any

def default_id_factory():
    """Generates a default UUID for document ID."""
    return str(uuid.uuid4())

@dataclass
class Document:
    """
    Represents a single document or a piece of text content.
    Now mutable to allow score to be updated.

    Attributes:
        page_content: The main textual content of the document.
        metadata: A dictionary of additional information about the document.
        id: A unique identifier for the document.
        score: The relevance score of the document.
    """
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=default_id_factory)
    score: float = field(default=0.0, compare=False)

    def __post_init__(self):
        """Post-initialization checks."""
        if not isinstance(self.page_content, str):
            raise TypeError("page_content must be a string.")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary.")
        if not isinstance(self.id, str):
            raise TypeError("id must be a string.")
        if not self.id:
            raise ValueError("id cannot be empty.")

    def __eq__(self, other):
        if not isinstance(other, Document):
            return NotImplemented
        return self.id == other.id and self.page_content == other.page_content and self.metadata == other.metadata

    def __hash__(self):
        # Score is excluded from hash as it's mutable and retrieval-specific
        meta_tuple = tuple(sorted(self.metadata.items()))
        return hash((self.page_content, meta_tuple, self.id))

# Example of how other models might be added later:
# @dataclass(frozen=True)
# class Chunk(Document):
#     """Represents a chunk of a larger document."""
#     parent_document_id: str
#     chunk_index: int
#     # Could have its own metadata or inherit/extend parent's

# @dataclass(frozen=True)
# class RetrievedDocument:
#     """Represents a document retrieved by the RAG pipeline, possibly with a score."""
#     document: Document
#     score: float = field(default=0.0)
#     # Any other retrieval-specific info