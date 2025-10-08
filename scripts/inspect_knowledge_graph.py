#!/usr/bin/env python3
"""
Graph Inspector Diagnostic Script

Inspects IRIS knowledge graph state for GraphRAG pipeline.
Checks table existence, entity/relationship/community counts, and data quality.

Contract: specs/032-investigate-graphrag-data/contracts/graph_inspector_contract.md

Exit Codes:
  0: Success - Knowledge graph populated with data
  1: Empty graph - Tables exist but contain no data
  2: Tables missing - Schema not initialized for GraphRAG
  3: Database connection error
"""

import json
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

# Try importing iris_rag components
try:
    from iris_rag.config.manager import ConfigurationManager
    from iris_rag.core.connection import ConnectionManager
except ImportError as e:
    # Critical error - cannot import framework
    output = {
        "check_name": "knowledge_graph_inspection",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "execution_time_ms": 0,
        "tables_exist": {"entities": False, "relationships": False, "communities": False},
        "counts": {"entities": 0, "relationships": 0, "communities": 0},
        "sample_entities": [],
        "document_links": {"total_entities": 0, "linked": 0, "orphaned": 0},
        "data_quality": {"entities_with_embeddings": 0, "completeness_score": 0.0},
        "diagnosis": {
            "severity": "critical",
            "message": f"Cannot import iris_rag framework: {e}",
            "suggestions": [
                "Verify iris_rag package installed: pip install -e .",
                "Activate virtual environment: source .venv/bin/activate",
                "Run: uv sync to install dependencies",
            ],
            "next_steps": [
                "Check package installation",
                "Verify Python environment is correct",
            ],
        },
    }
    print(json.dumps(output, indent=2))
    sys.exit(3)


def check_table_exists(cursor, table_name: str) -> bool:
    """Check if a table exists in the database."""
    try:
        # Try to query the table (IRIS-specific syntax)
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        cursor.fetchone()
        return True
    except Exception:
        return False


def get_row_count(cursor, table_name: str) -> int:
    """Get count of rows in a table."""
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        result = cursor.fetchone()
        return result[0] if result else 0
    except Exception:
        return 0


def get_sample_entities(cursor, limit: int = 5) -> List[Dict[str, Any]]:
    """Get up to 5 sample entities with their details."""
    try:
        cursor.execute(f"""
            SELECT id, name, type, document_id
            FROM RAG.Entities
            LIMIT {limit}
        """)
        rows = cursor.fetchall()

        entities = []
        for row in rows:
            entities.append({
                "id": row[0],
                "name": row[1],
                "type": row[2],
                "document_id": row[3] if row[3] else None,
            })
        return entities
    except Exception:
        return []


def get_document_link_stats(cursor, total_entities: int) -> Dict[str, int]:
    """Calculate entity-to-document link statistics."""
    try:
        # Count entities with document_id set
        cursor.execute("""
            SELECT COUNT(*) FROM RAG.Entities
            WHERE document_id IS NOT NULL AND document_id != ''
        """)
        linked = cursor.fetchone()[0] if cursor.fetchone() else 0

        orphaned = total_entities - linked

        return {
            "total_entities": total_entities,
            "linked": linked,
            "orphaned": orphaned,
        }
    except Exception:
        return {
            "total_entities": total_entities,
            "linked": 0,
            "orphaned": total_entities,
        }


def get_data_quality_metrics(cursor, total_entities: int) -> Dict[str, Any]:
    """Calculate data quality metrics."""
    try:
        # Count entities with embeddings (non-null vector column)
        cursor.execute("""
            SELECT COUNT(*) FROM RAG.Entities
            WHERE embedding IS NOT NULL
        """)
        entities_with_embeddings = cursor.fetchone()[0] if cursor.fetchone() else 0

        # Calculate completeness score
        completeness_score = (
            entities_with_embeddings / total_entities if total_entities > 0 else 0.0
        )

        return {
            "entities_with_embeddings": entities_with_embeddings,
            "completeness_score": round(completeness_score, 2),
        }
    except Exception:
        return {
            "entities_with_embeddings": 0,
            "completeness_score": 0.0,
        }


def generate_diagnosis(
    tables_exist: Dict[str, bool],
    counts: Dict[str, int],
    data_quality: Dict[str, Any],
) -> tuple[Dict[str, Any], int]:
    """Generate diagnosis and determine exit code."""
    # Check if tables are missing
    if not all(tables_exist.values()):
        missing_tables = [name for name, exists in tables_exist.items() if not exists]
        return (
            {
                "severity": "critical",
                "message": f"Knowledge graph schema not initialized - {', '.join(f'RAG.{t.capitalize()}' for t in missing_tables)} tables missing",
                "suggestions": [
                    "Run schema initialization for GraphRAG pipeline",
                    "Ensure SchemaManager.create_schema_manager('graphrag', ...) is called",
                    "Check database permissions for table creation",
                ],
                "next_steps": [
                    "Review schema_manager.py implementation",
                    "Verify GraphRAG pipeline initialization workflow",
                ],
            },
            2,  # Exit code 2: Tables missing
        )

    # Check if graph is empty
    if all(count == 0 for count in counts.values()):
        return (
            {
                "severity": "error",
                "message": "Knowledge graph is empty - no entities, relationships, or communities found",
                "suggestions": [
                    "Run entity extraction on loaded documents using GraphRAG pipeline",
                    "Check entity_extraction.enabled configuration in config file",
                    "Verify ontology is loaded for entity type detection",
                ],
                "next_steps": [
                    "Run verify_entity_extraction.py to check extraction service status",
                    "Review GraphRAG load_documents workflow for extraction invocation",
                ],
            },
            1,  # Exit code 1: Empty graph
        )

    # Graph has data - success
    return (
        {
            "severity": "info",
            "message": f"Knowledge graph is healthy - {counts['entities']} entities, {counts['relationships']} relationships, {counts['communities']} communities",
            "suggestions": [],
            "next_steps": [],
        },
        0,  # Exit code 0: Success
    )


def main():
    """Main inspection workflow."""
    start_time = time.time()

    try:
        # Initialize configuration and connection managers
        config_manager = ConfigurationManager()
        connection_manager = ConnectionManager(config_manager)

        # Get database connection
        try:
            conn = connection_manager.get_connection()
            cursor = conn.cursor()
        except Exception as e:
            # Connection error
            execution_time = round((time.time() - start_time) * 1000, 1)
            output = {
                "check_name": "knowledge_graph_inspection",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "execution_time_ms": execution_time,
                "tables_exist": {"entities": False, "relationships": False, "communities": False},
                "counts": {"entities": 0, "relationships": 0, "communities": 0},
                "sample_entities": [],
                "document_links": {"total_entities": 0, "linked": 0, "orphaned": 0},
                "data_quality": {"entities_with_embeddings": 0, "completeness_score": 0.0},
                "diagnosis": {
                    "severity": "critical",
                    "message": f"Database connection failed: {e}",
                    "suggestions": [
                        "Check IRIS database is running: docker ps | grep iris",
                        "Start IRIS: docker-compose up -d",
                        "Verify connection settings in .env file",
                    ],
                    "next_steps": [
                        "Verify IRIS container status",
                        "Check database connection parameters",
                    ],
                },
            }
            print(json.dumps(output, indent=2))
            sys.exit(3)

        # Check table existence
        tables_exist = {
            "entities": check_table_exists(cursor, "RAG.Entities"),
            "relationships": check_table_exists(cursor, "RAG.EntityRelationships"),
            "communities": check_table_exists(cursor, "RAG.Communities"),
        }

        # Get row counts
        counts = {
            "entities": get_row_count(cursor, "RAG.Entities") if tables_exist["entities"] else 0,
            "relationships": get_row_count(cursor, "RAG.EntityRelationships") if tables_exist["relationships"] else 0,
            "communities": get_row_count(cursor, "RAG.Communities") if tables_exist["communities"] else 0,
        }

        # Get sample entities (up to 5)
        sample_entities = get_sample_entities(cursor, limit=5) if tables_exist["entities"] else []

        # Get document link statistics
        document_links = get_document_link_stats(cursor, counts["entities"]) if tables_exist["entities"] else {
            "total_entities": 0,
            "linked": 0,
            "orphaned": 0,
        }

        # Get data quality metrics
        data_quality = get_data_quality_metrics(cursor, counts["entities"]) if tables_exist["entities"] else {
            "entities_with_embeddings": 0,
            "completeness_score": 0.0,
        }

        # Generate diagnosis and exit code
        diagnosis, exit_code = generate_diagnosis(tables_exist, counts, data_quality)

        # Calculate execution time
        execution_time = round((time.time() - start_time) * 1000, 1)

        # Build output
        output = {
            "check_name": "knowledge_graph_inspection",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "execution_time_ms": execution_time,
            "tables_exist": tables_exist,
            "counts": counts,
            "sample_entities": sample_entities,
            "document_links": document_links,
            "data_quality": data_quality,
            "diagnosis": diagnosis,
        }

        # Output JSON
        print(json.dumps(output, indent=2))

        # Exit with appropriate code
        sys.exit(exit_code)

    except Exception as e:
        # Unexpected error
        execution_time = round((time.time() - start_time) * 1000, 1)
        output = {
            "check_name": "knowledge_graph_inspection",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "execution_time_ms": execution_time,
            "tables_exist": {"entities": False, "relationships": False, "communities": False},
            "counts": {"entities": 0, "relationships": 0, "communities": 0},
            "sample_entities": [],
            "document_links": {"total_entities": 0, "linked": 0, "orphaned": 0},
            "data_quality": {"entities_with_embeddings": 0, "completeness_score": 0.0},
            "diagnosis": {
                "severity": "critical",
                "message": f"Unexpected error during inspection: {e}",
                "suggestions": [
                    "Check error logs for details",
                    "Verify database connection and schema",
                    "Report issue if error persists",
                ],
                "next_steps": [
                    "Review error stacktrace",
                    "Check database status",
                ],
            },
        }
        print(json.dumps(output, indent=2))
        sys.exit(3)


if __name__ == "__main__":
    main()
