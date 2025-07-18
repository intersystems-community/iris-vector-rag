import pytest

# This import will initially fail, which is expected for TDD,
# as scripts/schema_definition.py and its contents won't exist yet.

def test_schema_definition_structure():
    """
    Tests that the schema definition (e.g., a Python dictionary)
    has the correct structure and required fields.
    This test does not connect to the DB, it only checks the definition.
    """
    try:
        from scripts.schema_definition import EXPECTED_SCHEMA_DEFINITION
    except ImportError:
        pytest.fail("Could not import EXPECTED_SCHEMA_DEFINITION from scripts.schema_definition.py. Create the file and definition.")

    assert "table_name" in EXPECTED_SCHEMA_DEFINITION, "Schema definition must include 'table_name'"
    assert EXPECTED_SCHEMA_DEFINITION["table_name"] == "RAG.SourceDocuments", "Table name should be RAG.SourceDocuments"
    assert "columns" in EXPECTED_SCHEMA_DEFINITION, "Schema definition must include 'columns'"
    
    columns = EXPECTED_SCHEMA_DEFINITION["columns"]
    assert isinstance(columns, list), "'columns' should be a list"
    assert len(columns) == 3, "Schema should have exactly doc_id, text_content, and embedding columns for this minimal definition"

    expected_fields_spec = {
        "doc_id": {"type_prefix": "VARCHAR", "nullable": False, "primary_key": True},
        "text_content": {"type_exact": "CLOB", "nullable": False},
        "embedding": {"type_exact": "CLOB", "nullable": True} # Chosen CLOB for embeddings
    }

    defined_columns_map = {col["name"]: col for col in columns}

    for field_name, spec in expected_fields_spec.items():
        assert field_name in defined_columns_map, f"Field '{field_name}' is missing in schema definition"
        
        col_def = defined_columns_map[field_name]
        assert "type" in col_def, f"Field '{field_name}' must have a 'type' defined"
        assert "nullable" in col_def, f"Field '{field_name}' must have 'nullable' status defined"

        if "type_prefix" in spec:
            assert col_def["type"].upper().startswith(spec["type_prefix"]), \
                f"{field_name} type should start with {spec['type_prefix']}, got {col_def['type']}"
        if "type_exact" in spec:
            assert col_def["type"].upper() == spec["type_exact"], \
                f"{field_name} type should be {spec['type_exact']}, got {col_def['type']}"
        
        assert col_def["nullable"] == spec["nullable"], \
            f"{field_name} nullable status should be {spec['nullable']}, got {col_def['nullable']}"

        if spec.get("primary_key"):
            assert col_def.get("primary_key", False), f"{field_name} should be primary key"

def test_sql_ddl_exists_and_is_valid():
    """
    Tests that the SQL DDL string for creating the table exists and contains key elements.
    """
    try:
        from scripts.schema_definition import SOURCE_DOCUMENTS_TABLE_SQL
    except ImportError:
        pytest.fail("Could not import SOURCE_DOCUMENTS_TABLE_SQL from scripts.schema_definition.py. Create the file and DDL string.")
    
    assert isinstance(SOURCE_DOCUMENTS_TABLE_SQL, str), "SOURCE_DOCUMENTS_TABLE_SQL should be a string."
    assert len(SOURCE_DOCUMENTS_TABLE_SQL) > 0, "SOURCE_DOCUMENTS_TABLE_SQL should not be empty."
    
    # Basic checks for content
    assert "CREATE TABLE RAG.SourceDocuments" in SOURCE_DOCUMENTS_TABLE_SQL, "DDL should create RAG.SourceDocuments table."
    assert "doc_id VARCHAR(255) NOT NULL PRIMARY KEY" in SOURCE_DOCUMENTS_TABLE_SQL, "DDL should define doc_id correctly."
    assert "text_content CLOB NOT NULL" in SOURCE_DOCUMENTS_TABLE_SQL, "DDL should define text_content correctly."
    assert "embedding CLOB NULL" in SOURCE_DOCUMENTS_TABLE_SQL, "DDL should define embedding as CLOB NULL."
    # Check for the comment regarding TO_VECTOR
    assert "TO_VECTOR()" in SOURCE_DOCUMENTS_TABLE_SQL, "DDL comment should mention TO_VECTOR()."