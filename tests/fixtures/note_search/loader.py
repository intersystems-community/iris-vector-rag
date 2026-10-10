"""Load the note-search test vectors. No iris_vector_rag import: fhir-017 reads the same file."""

import json
from pathlib import Path

VECTORS_PATH = Path(__file__).with_name("vectors.json")


def load_vectors() -> dict:
    return json.loads(VECTORS_PATH.read_text(encoding="utf-8"))
