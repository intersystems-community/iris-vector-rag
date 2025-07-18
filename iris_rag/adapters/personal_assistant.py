"""
Adapter for integrating Personal Assistant with RAG templates.
"""

import logging
from typing import Any, Dict, Optional

from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager
from iris_rag.pipelines.basic import BasicRAGPipeline # Assuming BasicRAGPipeline is the one to be used

# Placeholder for the actual RAG pipeline initialization function if different
# from rag_templates.pipelines.basic_rag import initialize_pipeline as initialize_basic_rag_pipeline
# For now, we'll assume BasicRAGPipeline can be instantiated directly or has a compatible initializer

logger = logging.getLogger(__name__)

class PersonalAssistantAdapter:
    """
    Adapts the RAG templates system for use by a Personal Assistant.

    This adapter provides a compatible interface for `initialize_iris_rag_pipeline()`
    and handles configuration translation between the Personal Assistant's
    format and the RAG templates format.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the PersonalAssistantAdapter.

        Args:
            config: Optional configuration dictionary. If provided, it will be
                    used to initialize the ConfigurationManager.
        """
        self.config_manager = ConfigurationManager(config=config)
        self.connection_manager = ConnectionManager(config_manager=self.config_manager)
        self.rag_pipeline: Optional[BasicRAGPipeline] = None
        logger.info("PersonalAssistantAdapter initialized.")

    def _translate_config(self, pa_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translates Personal Assistant configuration to RAG templates configuration.
        This is a placeholder and needs to be implemented based on actual
        differences in configuration formats.

        Args:
            pa_config: Configuration dictionary from the Personal Assistant.

        Returns:
            Configuration dictionary compatible with RAG templates.
        """
        # Example translation (customize as needed):
        rag_config = pa_config.copy() # Start with a copy

        # Potentially map PA-specific keys to RAG template keys
        # if "pa_db_host" in rag_config:
        #     rag_config["iris_host"] = rag_config.pop("pa_db_host")
        # if "pa_api_key" in rag_config:
        #     rag_config["llm_api_key"] = rag_config.pop("pa_api_key")

        logger.debug(f"Translated PA config: {pa_config} to RAG config: {rag_config}")
        return rag_config

    def initialize_iris_rag_pipeline(
        self,
        config_path: Optional[str] = None,
        pa_specific_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> BasicRAGPipeline:
        """
        Initializes and returns a RAG pipeline instance, compatible with
        the Personal Assistant's expected `initialize_iris_rag_pipeline()` interface.

        This method handles:
        1. Loading configuration (from path or dictionary).
        2. Translating PA-specific configuration to RAG templates format.
        3. Setting up ConnectionManager and ConfigurationManager.
        4. Initializing the BasicRAGPipeline (or a specified RAG pipeline).

        Args:
            config_path: Optional path to a configuration file.
            pa_specific_config: Optional dictionary containing Personal Assistant-specific
                                configuration. This will be translated.
            **kwargs: Additional keyword arguments to pass to the RAG pipeline.

        Returns:
            An initialized RAG pipeline instance.

        Raises:
            Exception: If pipeline initialization fails.
        """
        logger.info(f"Initializing IRIS RAG pipeline via PersonalAssistantAdapter. Config path: {config_path}, PA config provided: {pa_specific_config is not None}")

        if config_path:
            self.config_manager.load_config(config_path)
            logger.info(f"Configuration loaded from path: {config_path}")

        if pa_specific_config:
            iris_rag_config = self._translate_config(pa_specific_config)
            # Merge translated config with existing config, translated taking precedence
            self.config_manager.update_config(iris_rag_config)
            logger.info("Personal Assistant specific configuration translated and merged.")

        # Ensure connection manager uses the latest config
        self.connection_manager.iris_connector = None # Reset to force re-initialization with new config if any

        try:
            # Assuming BasicRAGPipeline is the target.
            # If a different pipeline or a factory function is needed, adjust here.
            self.rag_pipeline = BasicRAGPipeline(
                connection_manager=self.connection_manager,
                config_manager=self.config_manager,
                **kwargs
            )
            logger.info("BasicRAGPipeline initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}", exc_info=True)
            # Potentially fall back to SurvivalModeRAGService if applicable,
            # or re-raise depending on desired error handling.
            # For now, re-raising to make failure explicit.
            raise

        return self.rag_pipeline

    def query(self, query_text: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Submits a query to the initialized RAG pipeline.

        Args:
            query_text: The query string.
            **kwargs: Additional arguments for the pipeline's query method.

        Returns:
            The result from the RAG pipeline.

        Raises:
            RuntimeError: If the RAG pipeline is not initialized.
        """
        if not self.rag_pipeline:
            logger.error("RAG pipeline not initialized. Call initialize_iris_rag_pipeline() first.")
            raise RuntimeError("RAG pipeline not initialized. Call initialize_iris_rag_pipeline() first.")

        logger.info(f"Submitting query to RAG pipeline: {query_text}")
        try:
            result = self.rag_pipeline.query(query_text, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Error during RAG pipeline query: {e}", exc_info=True)
            # Implement fallback or error handling as per requirements
            # For example, could return a predefined error structure or attempt survival mode.
            return {"error": str(e), "answer": "Sorry, I encountered an error processing your request."}

# Example usage (for illustration, typically this would be used by the PA)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Mock PA configuration
    mock_pa_config = {
        "iris_host": "localhost",
        "iris_port": 1972,
        "iris_namespace": "USER",
        "iris_user": "testuser",
        "iris_password": "testpassword",
        "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "llm_model_name": "gpt-3.5-turbo",
        # "pa_db_host": "pa_db_server", # Example of a PA-specific key
        # "pa_api_key": "pa_secret_key" # Example of a PA-specific key
    }

    adapter = PersonalAssistantAdapter()

    try:
        # Initialize pipeline using the adapter
        # In a real scenario, the PA would call this.
        # We pass the config directly for this example.
        pipeline = adapter.initialize_iris_rag_pipeline(pa_specific_config=mock_pa_config)
        print("RAG Pipeline initialized through adapter.")

        # Example query
        # response = adapter.query("What is the capital of France?")
        # print(f"Query Response: {response}")

    except Exception as e:
        print(f"An error occurred: {e}")

    # Example of initializing with a config file (if one exists)
    # try:
    #     adapter_from_file = PersonalAssistantAdapter()
    #     # Create a dummy config file for this example if needed
    #     # with open("dummy_config.json", "w") as f:
    #     #     import json
    #     #     json.dump(mock_pa_config, f)
    #     pipeline_from_file = adapter_from_file.initialize_iris_rag_pipeline(config_path="dummy_config.json")
    #     print("RAG Pipeline initialized from file through adapter.")
    #     response_file = adapter_from_file.query("Tell me about IRIS.")
    #     print(f"Query Response (from file config): {response_file}")
    # except Exception as e:
    #     print(f"An error occurred with file config: {e}")