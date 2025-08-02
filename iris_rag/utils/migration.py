"""
Utilities for migrating data and configurations from existing
Personal Assistant setups to the RAG templates system.
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

class PersonalAssistantMigrationUtils:
    """
    Provides methods for migrating data and configurations from a legacy
    Personal Assistant system to the RAG templates format.
    """

    def __init__(self):
        logger.info("PersonalAssistantMigrationUtils initialized.")

    def convert_legacy_config_to_rag_config(
        self,
        legacy_pa_config: Dict[str, Any],
        mapping_rules: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Converts a legacy Personal Assistant configuration dictionary to the
        RAG templates configuration format.

        Args:
            legacy_pa_config: The legacy configuration dictionary.
            mapping_rules: Optional dictionary defining direct key mappings.
                           Example: {"old_db_host": "iris_host", "apiKey": "llm_api_key"}

        Returns:
            A configuration dictionary compatible with RAG templates.
        """
        rag_config = {}
        if mapping_rules is None:
            # Default basic mapping rules (customize as needed)
            mapping_rules = {
                "pa_database_host": "iris_host",
                "pa_database_port": "iris_port",
                "pa_database_namespace": "iris_namespace",
                "pa_database_user": "iris_user",
                "pa_database_password": "iris_password",
                "pa_embedding_model": "embedding_model_name",
                "pa_llm_model": "llm_model_name",
                "pa_api_token": "llm_api_key", # Example, adjust if PA uses different auth
                # Add more default mappings here based on common PA config keys
            }

        logger.info(f"Starting configuration conversion with mapping rules: {mapping_rules}")

        for legacy_key, legacy_value in legacy_pa_config.items():
            # Direct mapping
            if legacy_key in mapping_rules:
                rag_key = mapping_rules[legacy_key]
                rag_config[rag_key] = legacy_value
                logger.debug(f"Mapped legacy key '{legacy_key}' to RAG key '{rag_key}'.")
            else:
                # Handle keys not in direct mapping rules
                # Option 1: Carry over if not explicitly mapped (could be risky)
                # rag_config[legacy_key] = legacy_value
                # logger.debug(f"Carried over legacy key '{legacy_key}' as is.")
                # Option 2: Log and skip
                logger.warning(f"Legacy key '{legacy_key}' not found in mapping rules and was not carried over.")
                # Option 3: Attempt intelligent mapping or transformation (more complex)
                # For now, we'll log and skip unmapped keys to avoid polluting RAG config.

        # Add any default RAG template values if they are missing and required
        # For example:
        # if "log_level" not in rag_config:
        #     rag_config["log_level"] = "INFO"

        logger.info(f"Converted legacy PA config: {legacy_pa_config} to RAG config: {rag_config}")
        return rag_config

    def migrate_legacy_data_format(
        self,
        legacy_data: List[Dict[str, Any]],
        data_mapping_rules: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """
        Converts a list of legacy data records (e.g., documents) to the
        RAG templates data format (e.g., List[Document] or List[Dict]
        compatible with RAG ingestion).

        This is a placeholder and needs to be highly customized based on the
        actual legacy data structure and the target RAG Document model.

        Args:
            legacy_data: A list of dictionaries, where each dictionary
                         represents a legacy data record.
            data_mapping_rules: A dictionary mapping legacy field names to
                                RAG template Document field names.
                                Example: {"document_text": "content", "source_id": "doc_id"}

        Returns:
            A list of dictionaries in the RAG templates data format.
        """
        migrated_data = []
        logger.info(f"Starting data migration for {len(legacy_data)} records with rules: {data_mapping_rules}")

        for i, legacy_record in enumerate(legacy_data):
            rag_record = {}
            metadata = {} # For fields that go into Document.metadata

            for legacy_field, legacy_value in legacy_record.items():
                if legacy_field in data_mapping_rules:
                    rag_field = data_mapping_rules[legacy_field]
                    # Assuming RAG Document has 'doc_id', 'content', and 'metadata'
                    if rag_field in ["doc_id", "content"]: # Direct fields in a potential Document model
                        rag_record[rag_field] = legacy_value
                    else: # Handle metadata fields with dot notation
                        if rag_field.startswith("metadata."):
                            # Extract the actual metadata field name (remove "metadata." prefix)
                            metadata_field = rag_field[9:]  # Remove "metadata." (9 characters)
                            metadata[metadata_field] = legacy_value
                        else:
                            # Regular metadata field without dot notation
                            metadata[rag_field] = legacy_value
                else:
                    # Unmapped fields could also go into metadata by default
                    metadata[legacy_field] = legacy_value
                    logger.debug(f"Record {i}: Unmapped legacy field '{legacy_field}' added to metadata.")
            
            if metadata:
                rag_record["metadata"] = metadata
            
            # Basic validation (example) - Allow records with only metadata if they have some content
            if "content" not in rag_record and "doc_id" not in rag_record and not metadata:
                logger.warning(f"Record {i} (Legacy: {legacy_record}) is missing essential fields ('content' or 'doc_id') and has no metadata after mapping. Skipping.")
                continue
            
            migrated_data.append(rag_record)
            logger.debug(f"Migrated record {i}: {rag_record}")

        logger.info(f"Successfully migrated {len(migrated_data)} data records.")
        return migrated_data

    def load_legacy_config_from_file(self, file_path: str) -> Dict[str, Any]:
        """Loads a legacy configuration from a JSON file."""
        try:
            with open(file_path, 'r') as f:
                legacy_config = json.load(f)
            logger.info(f"Successfully loaded legacy config from {file_path}")
            return legacy_config
        except FileNotFoundError:
            logger.error(f"Legacy config file not found: {file_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from legacy config file {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading legacy config from {file_path}: {e}")
            raise

    def load_legacy_data_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Loads legacy data from a JSON file (assuming a list of records)."""
        try:
            with open(file_path, 'r') as f:
                legacy_data = json.load(f)
            if not isinstance(legacy_data, list):
                raise ValueError("Legacy data file should contain a JSON list of records.")
            logger.info(f"Successfully loaded {len(legacy_data)} legacy data records from {file_path}")
            return legacy_data
        except FileNotFoundError:
            logger.error(f"Legacy data file not found: {file_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from legacy data file {file_path}: {e}")
            raise
        except ValueError as e:
            logger.error(f"Data format error in {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading legacy data from {file_path}: {e}")
            raise

# Example Usage (for illustration)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    migration_util = PersonalAssistantMigrationUtils()

    # --- Config Migration Example ---
    print("\n--- Configuration Migration Example ---")
    mock_legacy_pa_config = {
        "pa_database_host": "old-server.local",
        "pa_database_port": 54321,
        "pa_database_namespace": "PA_NAMESPACE",
        "pa_embedding_model": "text-embedding-ada-002",
        "pa_llm_model": "gpt-4",
        "pa_custom_setting": "custom_value", # This will be unmapped by default rules
        "apiKey": "legacy_api_key_value" # This will be unmapped by default rules
    }
    # Custom mapping rules if defaults are not sufficient
    custom_config_mapping = {
        "pa_database_host": "iris_host",
        "pa_database_port": "iris_port",
        "pa_database_namespace": "iris_namespace",
        "pa_embedding_model": "embedding_model_name",
        "pa_llm_model": "llm_model_name",
        "apiKey": "llm_api_key" # Map 'apiKey' to 'llm_api_key'
    }
    converted_rag_config = migration_util.convert_legacy_config_to_rag_config(
        mock_legacy_pa_config,
        mapping_rules=custom_config_mapping
    )
    print(f"Original Legacy PA Config: {mock_legacy_pa_config}")
    print(f"Converted RAG Config: {converted_rag_config}")

    # --- Data Migration Example ---
    print("\n--- Data Migration Example ---")
    mock_legacy_data = [
        {"document_id": "doc1", "text_content": "This is the first document.", "source_system": "SystemA"},
        {"document_id": "doc2", "text_content": "Content for the second document.", "author": "Jane Doe"},
        {"id": "doc3", "main_text": "Third piece of text.", "category": "General"} # Different field names
    ]
    # Define how legacy data fields map to RAG Document fields (content, doc_id, metadata)
    custom_data_mapping = {
        "document_id": "doc_id",  # Maps to Document.doc_id
        "text_content": "content", # Maps to Document.content
        "source_system": "source", # Will go into Document.metadata.source
        "author": "author_name",   # Will go into Document.metadata.author_name
        # For the third record with different field names:
        "id": "doc_id",
        "main_text": "content",
        "category": "topic"        # Will go into Document.metadata.topic
    }
    migrated_docs_data = migration_util.migrate_legacy_data_format(
        mock_legacy_data,
        data_mapping_rules=custom_data_mapping
    )
    print(f"Original Legacy Data: {json.dumps(mock_legacy_data, indent=2)}")
    print(f"Migrated RAG-compatible Data: {json.dumps(migrated_docs_data, indent=2)}")

    # Example: Loading from dummy files
    # Create dummy legacy config file
    # with open("dummy_legacy_config.json", "w") as f:
    #     json.dump(mock_legacy_pa_config, f)
    # loaded_legacy_cfg = migration_util.load_legacy_config_from_file("dummy_legacy_config.json")
    # print(f"Loaded legacy config from file: {loaded_legacy_cfg}")

    # Create dummy legacy data file
    # with open("dummy_legacy_data.json", "w") as f:
    #     json.dump(mock_legacy_data, f)
    # loaded_legacy_data_list = migration_util.load_legacy_data_from_file("dummy_legacy_data.json")
    # print(f"Loaded legacy data from file (first record): {loaded_legacy_data_list[0] if loaded_legacy_data_list else 'Empty'}")