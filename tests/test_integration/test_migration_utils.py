"""
Integration tests for PersonalAssistantMigrationUtils.
"""
import pytest
import json
import os
from iris_rag.utils.migration import PersonalAssistantMigrationUtils

@pytest.fixture
def migration_util():
    """Fixture for PersonalAssistantMigrationUtils instance."""
    return PersonalAssistantMigrationUtils()

@pytest.fixture
def legacy_pa_config_data():
    """Sample legacy Personal Assistant configuration data."""
    return {
        "pa_database_host": "legacy.db.example.com",
        "pa_database_port": 12345,
        "pa_database_namespace": "LEGACY_NS",
        "pa_database_user": "legacy_user",
        "pa_database_password": "legacy_password",
        "pa_embedding_model": "legacy-text-embedder",
        "pa_llm_model": "legacy-language-model",
        "pa_api_token": "abc123xyz789",
        "some_other_setting": "value123",
        "timeout_ms": 5000
    }

@pytest.fixture
def rag_config_mapping_rules():
    """Mapping rules for config conversion."""
    return {
        "pa_database_host": "iris_host",
        "pa_database_port": "iris_port",
        "pa_database_namespace": "iris_namespace",
        "pa_database_user": "iris_user",
        "pa_database_password": "iris_password",
        "pa_embedding_model": "embedding_model_name",
        "pa_llm_model": "llm_model_name",
        "pa_api_token": "llm_api_key"
        # "timeout_ms" is intentionally not mapped to test unmapped key handling
    }

@pytest.fixture
def legacy_pa_data_records():
    """Sample legacy Personal Assistant data records."""
    return [
        {"document_id": "pa_doc_001", "text_content": "First PA document.", "source": "PA_System_Alpha", "created_at": "2023-01-01"},
        {"document_id": "pa_doc_002", "text_content": "Second PA document with details.", "author": "PA User", "tags": ["important", "review"]},
        {"doc_unique_id": "pa_doc_003", "content_body": "Third document, different keys.", "status": "final"}
    ]

@pytest.fixture
def rag_data_mapping_rules():
    """Mapping rules for data record conversion."""
    return {
        "document_id": "doc_id",
        "text_content": "content",
        "source": "metadata.source_system", # Example: nesting under metadata
        "created_at": "metadata.creation_date",
        "author": "metadata.author",
        "tags": "metadata.labels",
        # For the third record with different keys
        "doc_unique_id": "doc_id",
        "content_body": "content",
        "status": "metadata.processing_status"
    }

@pytest.fixture
def temp_legacy_config_file(tmp_path, legacy_pa_config_data):
    """Creates a temporary legacy config JSON file."""
    file_path = tmp_path / "legacy_config.json"
    with open(file_path, "w") as f:
        json.dump(legacy_pa_config_data, f)
    return str(file_path)

@pytest.fixture
def temp_legacy_data_file(tmp_path, legacy_pa_data_records):
    """Creates a temporary legacy data JSON file."""
    file_path = tmp_path / "legacy_data.json"
    with open(file_path, "w") as f:
        json.dump(legacy_pa_data_records, f)
    return str(file_path)


def test_convert_legacy_config_with_rules(migration_util, legacy_pa_config_data, rag_config_mapping_rules):
    """Test configuration conversion with specified mapping rules."""
    converted_config = migration_util.convert_legacy_config_to_rag_config(
        legacy_pa_config_data,
        mapping_rules=rag_config_mapping_rules
    )

    assert converted_config["iris_host"] == legacy_pa_config_data["pa_database_host"]
    assert converted_config["iris_port"] == legacy_pa_config_data["pa_database_port"]
    assert converted_config["embedding_model_name"] == legacy_pa_config_data["pa_embedding_model"]
    assert converted_config["llm_api_key"] == legacy_pa_config_data["pa_api_token"]
    
    # Check that unmapped keys from legacy_pa_config_data are not present if not carried over
    # (Current implementation logs and skips unmapped keys)
    assert "some_other_setting" not in converted_config
    assert "timeout_ms" not in converted_config
    assert "pa_database_password" in converted_config # Check one that should be there

def test_convert_legacy_config_with_default_rules(migration_util, legacy_pa_config_data):
    """Test configuration conversion with default internal mapping rules."""
    # Modify legacy_pa_config_data to match default rule keys if necessary, or rely on its current state
    # The default rules in PersonalAssistantMigrationUtils are:
    # "pa_database_host": "iris_host", "pa_database_port": "iris_port", etc.
    
    converted_config = migration_util.convert_legacy_config_to_rag_config(legacy_pa_config_data) # No rules passed

    assert converted_config.get("iris_host") == legacy_pa_config_data["pa_database_host"]
    assert converted_config.get("iris_port") == legacy_pa_config_data["pa_database_port"]
    assert converted_config.get("embedding_model_name") == legacy_pa_config_data["pa_embedding_model"]
    # "pa_api_token" is mapped to "llm_api_key" by default rules in the class
    assert converted_config.get("llm_api_key") == legacy_pa_config_data["pa_api_token"] 
    
    assert "some_other_setting" not in converted_config # Unmapped by default rules

def test_migrate_legacy_data(migration_util, legacy_pa_data_records, rag_data_mapping_rules):
    """Test data record migration."""
    migrated_data = migration_util.migrate_legacy_data_format(
        legacy_pa_data_records,
        data_mapping_rules=rag_data_mapping_rules
    )
    assert len(migrated_data) == len(legacy_pa_data_records)

    # Check first record
    assert migrated_data[0]["doc_id"] == legacy_pa_data_records[0]["document_id"]
    assert migrated_data[0]["content"] == legacy_pa_data_records[0]["text_content"]
    assert migrated_data[0]["metadata"]["source_system"] == legacy_pa_data_records[0]["source"]
    assert migrated_data[0]["metadata"]["creation_date"] == legacy_pa_data_records[0]["created_at"]

    # Check second record (metadata structure)
    assert migrated_data[1]["doc_id"] == legacy_pa_data_records[1]["document_id"]
    assert migrated_data[1]["metadata"]["author"] == legacy_pa_data_records[1]["author"]
    assert migrated_data[1]["metadata"]["labels"] == legacy_pa_data_records[1]["tags"]

    # Check third record (different original keys)
    assert migrated_data[2]["doc_id"] == legacy_pa_data_records[2]["doc_unique_id"]
    assert migrated_data[2]["content"] == legacy_pa_data_records[2]["content_body"]
    assert migrated_data[2]["metadata"]["processing_status"] == legacy_pa_data_records[2]["status"]

def test_migrate_legacy_data_unmapped_fields_to_metadata(migration_util, legacy_pa_data_records):
    """Test that unmapped fields in data records go to metadata by default."""
    simple_data_mapping = {
        "document_id": "doc_id",
        "text_content": "content"
    }
    legacy_record_with_extra = [{"document_id": "d1", "text_content": "c1", "extra_field": "extra_val"}]
    migrated = migration_util.migrate_legacy_data_format(legacy_record_with_extra, simple_data_mapping)
    
    assert "metadata" in migrated[0]
    assert migrated[0]["metadata"]["extra_field"] == "extra_val"

def test_load_legacy_config_from_file_success(migration_util, temp_legacy_config_file, legacy_pa_config_data):
    """Test loading legacy config from a valid JSON file."""
    loaded_config = migration_util.load_legacy_config_from_file(temp_legacy_config_file)
    assert loaded_config == legacy_pa_config_data

def test_load_legacy_config_from_file_not_found(migration_util, tmp_path):
    """Test loading legacy config from a non-existent file."""
    non_existent_file = tmp_path / "not_found.json"
    with pytest.raises(FileNotFoundError):
        migration_util.load_legacy_config_from_file(str(non_existent_file))

def test_load_legacy_config_from_file_invalid_json(migration_util, tmp_path):
    """Test loading legacy config from a file with invalid JSON."""
    invalid_json_file = tmp_path / "invalid.json"
    with open(invalid_json_file, "w") as f:
        f.write("{'key': 'value',,}") # Invalid JSON
    with pytest.raises(json.JSONDecodeError):
        migration_util.load_legacy_config_from_file(str(invalid_json_file))

def test_load_legacy_data_from_file_success(migration_util, temp_legacy_data_file, legacy_pa_data_records):
    """Test loading legacy data from a valid JSON file."""
    loaded_data = migration_util.load_legacy_data_from_file(temp_legacy_data_file)
    assert loaded_data == legacy_pa_data_records

def test_load_legacy_data_from_file_not_list(migration_util, tmp_path):
    """Test loading legacy data from a file that doesn't contain a list."""
    not_list_file = tmp_path / "not_list.json"
    with open(not_list_file, "w") as f:
        json.dump({"key": "value"}, f) # JSON object, not list
    with pytest.raises(ValueError, match="Legacy data file should contain a JSON list of records."):
        migration_util.load_legacy_data_from_file(str(not_list_file))


# To run these tests: pytest tests/test_integration/test_migration_utils.py