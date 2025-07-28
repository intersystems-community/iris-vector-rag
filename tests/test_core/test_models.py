import pytest

# Attempt to import Document, will fail initially
from common.utils import Document

def test_import_document_model():
    """Tests that Document model can be imported."""
    assert Document is not None, "Failed to import Document from iris_rag.core.models"

def test_document_creation_with_content_and_metadata():
    """Tests creating a Document with page_content and metadata."""
    if Document is None:
        pytest.fail("Document model not imported")
    
    page_content = "This is a test document."
    metadata = {"source": "test_source.txt", "page": 1}
    doc = Document(content=page_content, metadata=metadata)
    
    assert doc.content == page_content
    assert doc.metadata == metadata
    assert doc.id is not None # Expect an ID to be generated or present
    assert isinstance(doc.id, str)

def test_document_creation_default_metadata_and_id():
    """Tests Document creation with default metadata (empty dict) and ID generation."""
    if Document is None:
        pytest.fail("Document model not imported")
        
    page_content = "Another test document."
    doc1 = Document(content=page_content)
    
    assert doc1.content == page_content
    assert doc1.metadata == {} # Expect default metadata to be an empty dict
    assert doc1.id is not None
    assert isinstance(doc1.id, str)

    doc2 = Document(content="Yet another test document.")
    assert doc2.id is not None
    assert doc1.id != doc2.id, "Document IDs should be unique by default."

def test_document_is_dataclass_like():
    """
    Tests if the Document behaves like a dataclass 
    (i.e., has __repr__, __eq__, etc., implicitly or explicitly).
    """
    if Document is None:
        pytest.fail("Document model not imported")

    doc1 = Document(content="Content", metadata={"a": 1}, id="fixed_id_1")
    doc2 = Document(content="Content", metadata={"a": 1}, id="fixed_id_1")
    doc3 = Document(content="Different Content", metadata={"a": 1}, id="fixed_id_2")
    doc4 = Document(content="Content", metadata={"b": 2}, id="fixed_id_3")

    assert repr(doc1) is not None # Basic check for a __repr__
    assert doc1 == doc2, "Documents with same content, metadata, and id should be equal."
    assert doc1 != doc3, "Documents with different content should not be equal."
    assert doc1 != doc4, "Documents with different metadata should not be equal."
    
    # Test hashing if it's intended to be hashable (e.g., for set storage)
    # This will fail if not a dataclass with frozen=True or no custom __hash__
    try:
        d = {doc1: "test"}
        assert doc1 in d
    except TypeError:
        pytest.fail("Document instances should be hashable if used as dict keys or in sets.")

def test_document_with_provided_id():
    """Tests creating a Document with a provided ID."""
    if Document is None:
        pytest.fail("Document model not imported")
    
    page_content = "Content with provided ID."
    metadata = {"source": "manual_id.txt"}
    provided_id = "my_custom_id_123"
    doc = Document(content=page_content, metadata=metadata, id=provided_id)
    
    assert doc.id == provided_id
    assert doc.content == page_content
    assert doc.metadata == metadata

# Placeholder for future related models like Chunk if needed
# class Chunk:
#     pass 
# def test_chunk_creation():
#     pass