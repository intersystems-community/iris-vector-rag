#!/usr/bin/env python3
"""
Generate a 50-document knowledge graph fixture for integration testing.

This script extracts 50 PMC documents with their entities and relationships
from the database and saves them as a JSON fixture that can be used for
integration testing without requiring LLM API calls.

Usage:
    python scripts/generate_test_kg_fixture.py --output tests/fixtures/graphrag/medical_kg_50docs.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.iris_dbapi_connector import get_iris_dbapi_connection


def extract_knowledge_graph(num_docs=50):
    """Extract knowledge graph data from database."""
    conn = get_iris_dbapi_connection()
    cursor = conn.cursor()

    print(f"Extracting {num_docs} documents with entities and relationships from database...")

    # Get documents with entities (any type - tickets, PMC, etc.)
    cursor.execute(f"""
        SELECT DISTINCT TOP {num_docs}
            sd.doc_id,
            sd.text_content,
            sd.metadata
        FROM RAG.SourceDocuments sd
        JOIN RAG.Entities e ON e.source_doc_id = sd.doc_id
        ORDER BY sd.doc_id
    """)

    documents = []
    doc_ids = []

    for doc_id, text_content, metadata in cursor.fetchall():
        # Handle IRIS CLOB data
        if hasattr(text_content, 'read'):
            text_content = text_content.read().decode('utf-8')
        else:
            text_content = str(text_content)

        if hasattr(metadata, 'read'):
            metadata = metadata.read().decode('utf-8')
        else:
            metadata = str(metadata)

        documents.append({
            "doc_id": str(doc_id),
            "content": text_content,
            "metadata": metadata
        })
        doc_ids.append(str(doc_id))

    print(f"✓ Extracted {len(documents)} documents")

    # Get entities for these documents
    doc_ids_str = "', '".join(doc_ids)
    cursor.execute(f"""
        SELECT
            entity_id,
            entity_name,
            entity_type,
            source_doc_id
        FROM RAG.Entities
        WHERE source_doc_id IN ('{doc_ids_str}')
    """)

    entities_by_doc = {}
    for entity_id, entity_name, entity_type, source_doc_id in cursor.fetchall():
        doc_id = str(source_doc_id)
        if doc_id not in entities_by_doc:
            entities_by_doc[doc_id] = []

        entities_by_doc[doc_id].append({
            "entity_id": str(entity_id),
            "name": str(entity_name),
            "type": str(entity_type)
        })

    total_entities = sum(len(ents) for ents in entities_by_doc.values())
    print(f"✓ Extracted {total_entities} entities")

    # Get relationships for these documents
    cursor.execute(f"""
        SELECT
            r.relationship_id,
            r.source_entity_id,
            r.target_entity_id,
            r.relationship_type,
            r.confidence
        FROM RAG.EntityRelationships r
        JOIN RAG.Entities e1 ON r.source_entity_id = e1.entity_id
        WHERE e1.source_doc_id IN ('{doc_ids_str}')
    """)

    relationships_by_doc = {}
    for rel_id, source_entity_id, target_entity_id, rel_type, confidence in cursor.fetchall():
        # Find which doc this relationship belongs to (via source entity)
        for doc_id, entities in entities_by_doc.items():
            if any(e['entity_id'] == str(source_entity_id) for e in entities):
                if doc_id not in relationships_by_doc:
                    relationships_by_doc[doc_id] = []

                relationships_by_doc[doc_id].append({
                    "relationship_id": str(rel_id),
                    "source": str(source_entity_id),
                    "target": str(target_entity_id),
                    "type": str(rel_type),
                    "confidence": float(confidence) if confidence else 1.0
                })
                break

    total_relationships = sum(len(rels) for rels in relationships_by_doc.values())
    print(f"✓ Extracted {total_relationships} relationships")

    # Combine into fixture format
    fixture_documents = []
    for doc in documents:
        doc_id = doc["doc_id"]
        fixture_documents.append({
            "doc_id": doc_id,
            "content": doc["content"],
            "metadata": doc["metadata"],
            "expected_entities": entities_by_doc.get(doc_id, []),
            "expected_relationships": relationships_by_doc.get(doc_id, [])
        })

    cursor.close()
    conn.close()

    return {
        "description": f"Knowledge graph fixture with {num_docs} documents from production database",
        "total_documents": len(fixture_documents),
        "total_entities": total_entities,
        "total_relationships": total_relationships,
        "documents": fixture_documents
    }


def main():
    parser = argparse.ArgumentParser(description="Generate knowledge graph test fixture")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--docs", type=int, default=50, help="Number of documents to extract")

    args = parser.parse_args()

    # Extract data
    fixture_data = extract_knowledge_graph(num_docs=args.docs)

    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(fixture_data, f, indent=2)

    print(f"\n✅ Fixture saved to: {output_path}")
    print(f"   Documents: {fixture_data['total_documents']}")
    print(f"   Entities: {fixture_data['total_entities']}")
    print(f"   Relationships: {fixture_data['total_relationships']}")
    print(f"   Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
