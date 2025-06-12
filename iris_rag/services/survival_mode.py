"""
SurvivalModeRAGService for minimal configuration and fallback scenarios.
"""

import logging
from typing import Any, Dict, Optional, List

from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager
from iris_rag.pipelines.basic import BasicRAGPipeline # Assuming this is the primary RAG pipeline
from iris_rag.core.models import Document

logger = logging.getLogger(__name__)

class SurvivalModeRAGService:
    """
    Provides RAG capabilities with a focus on resilience and graceful degradation.

    In "survival mode," this service attempts to use a fully configured RAG
    pipeline (e.g., BasicRAGPipeline). If the primary pipeline is unavailable
    or encounters errors, it can fall back to simpler, more resilient mechanisms,
    such as returning predefined responses, or attempting a very basic retrieval
    if possible, or simply indicating that the advanced RAG features are temporarily
    unavailable.
    """

    def __init__(
        self,
        connection_manager: Optional[ConnectionManager] = None,
        config_manager: Optional[ConfigurationManager] = None,
        primary_pipeline: Optional[BasicRAGPipeline] = None
    ):
        """
        Initializes the SurvivalModeRAGService.

        Args:
            connection_manager: An instance of ConnectionManager.
                                If None, a new one will be created.
            config_manager: An instance of ConfigurationManager.
                            If None, a new one will be created.
            primary_pipeline: An optional pre-initialized primary RAG pipeline.
        """
        self.config_manager = config_manager or ConfigurationManager()
        self.connection_manager = connection_manager or ConnectionManager(config_manager=self.config_manager)
        self.primary_pipeline: Optional[BasicRAGPipeline] = primary_pipeline
        self.is_primary_pipeline_healthy = True # Assume healthy initially

        if not self.primary_pipeline:
            try:
                # Attempt to initialize the primary pipeline with current config
                self.primary_pipeline = BasicRAGPipeline(
                    connection_manager=self.connection_manager,
                    config_manager=self.config_manager
                )
                logger.info("SurvivalModeRAGService: Primary BasicRAGPipeline initialized successfully.")
            except Exception as e:
                logger.warning(f"SurvivalModeRAGService: Failed to initialize primary BasicRAGPipeline: {e}. Operating in fallback mode.", exc_info=True)
                self.primary_pipeline = None
                self.is_primary_pipeline_healthy = False
        
        logger.info("SurvivalModeRAGService initialized.")

    def _check_primary_pipeline_health(self) -> bool:
        """
        Performs a basic health check on the primary RAG pipeline.
        This is a placeholder and can be expanded with actual health check logic.
        """
        if self.primary_pipeline is None:
            self.is_primary_pipeline_healthy = False
            return False
        
        # Add more sophisticated health checks if needed, e.g., pinging DB, LLM
        # For now, just check if it's instantiated.
        # A more robust check might try a dummy query or check connections.
        try:
            # Example: Check if connection manager can get a connection
            if self.connection_manager.get_iris_connection() is None:
                 logger.warning("SurvivalModeRAGService: Primary pipeline health check failed - no IRIS connection.")
                 self.is_primary_pipeline_healthy = False
                 return False
        except Exception as e:
            logger.warning(f"SurvivalModeRAGService: Primary pipeline health check failed: {e}")
            self.is_primary_pipeline_healthy = False
            return False
        
        # If we made it here, assume healthy for now
        # self.is_primary_pipeline_healthy = True # This might be too optimistic
        return self.is_primary_pipeline_healthy


    def query(self, query_text: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Processes a query, attempting to use the primary RAG pipeline first,
        then falling back to survival mechanisms if necessary.

        Args:
            query_text: The query string.
            **kwargs: Additional arguments for the pipeline's query method.

        Returns:
            A dictionary containing the answer and other relevant information.
            The structure might vary based on whether the primary pipeline
            succeeded or a fallback was used.
        """
        logger.info(f"SurvivalModeRAGService processing query: {query_text}")

        if self.is_primary_pipeline_healthy and self.primary_pipeline:
            try:
                logger.debug("Attempting query with primary RAG pipeline.")
                result = self.primary_pipeline.query(query_text, **kwargs)
                # Check if the result indicates an issue that should trigger fallback
                if result.get("error"): # or some other indicator of failure
                    logger.warning(f"Primary pipeline returned an error: {result.get('error')}. Attempting fallback.")
                    self.is_primary_pipeline_healthy = False # Mark as unhealthy for subsequent queries
                    return self._fallback_query(query_text, original_error=result.get("error"))
                return result
            except Exception as e:
                logger.error(f"Error querying primary RAG pipeline: {e}. Switching to fallback.", exc_info=True)
                self.is_primary_pipeline_healthy = False # Mark as unhealthy
                return self._fallback_query(query_text, original_error=str(e))
        else:
            logger.warning("Primary RAG pipeline is not available or unhealthy. Using fallback.")
            return self._fallback_query(query_text)

    def _fallback_query(self, query_text: str, original_error: Optional[str] = None) -> Dict[str, Any]:
        """
        Provides a fallback response when the primary RAG pipeline is unavailable.

        Args:
            query_text: The original query text.
            original_error: The error message from the primary pipeline, if any.

        Returns:
            A dictionary with a fallback answer.
        """
        logger.info(f"Executing fallback query for: {query_text}")
        
        # Basic fallback: acknowledge the issue and provide a generic response.
        # This can be made more sophisticated, e.g., by trying a keyword search
        # against a local cache or a very simple database query if IRIS is up
        # but the LLM/embedding models are down.

        fallback_message = "The advanced information retrieval system is temporarily unavailable. "
        if original_error:
            fallback_message += f"Details: {original_error}. "
        
        # Attempt a very simple keyword search if connection manager is available
        # This is a very basic example and would need proper implementation
        retrieved_docs: List[Document] = []
        try:
            if self.connection_manager and self.connection_manager.get_iris_connection():
                # This is a placeholder for a very simple retrieval logic
                # For example, a direct SQL query if a table with documents exists
                # and can be queried without complex embeddings.
                # conn = self.connection_manager.get_iris_connection()
                # cursor = conn.cursor()
                # simplified_query = f"%{query_text.split()[0]}%" # very naive
                # cursor.execute("SELECT TOP 3 DocId, Content FROM RAG.SourceDocuments WHERE Content LIKE ?", (simplified_query,))
                # rows = cursor.fetchall()
                # for row in rows:
                #     retrieved_docs.append(Document(doc_id=str(row[0]), content=str(row[1])))
                # if retrieved_docs:
                #     fallback_message += "I found some potentially related information based on keywords: "
                #     fallback_message += " ".join([doc.content[:100] + "..." for doc in retrieved_docs])
                # else:
                #     fallback_message += "I could not find information using a simple keyword search."
                # logger.info(f"Fallback keyword search retrieved {len(retrieved_docs)} documents.")
                pass # Placeholder for actual simple retrieval
        except Exception as e:
            logger.warning(f"Error during fallback simple retrieval attempt: {e}", exc_info=True)
            fallback_message += "An attempt to perform a basic search also failed. "
        
        fallback_message += "Please try again later or contact support."

        return {
            "query": query_text,
            "answer": fallback_message,
            "retrieved_documents": [], # Or retrieved_docs if the simple search above is implemented
            "source": "SurvivalModeFallback",
            "error": original_error or "Primary RAG pipeline unavailable.",
            "status": "degraded"
        }

    def reinitialize_primary_pipeline(self) -> bool:
        """
        Attempts to re-initialize the primary RAG pipeline.
        This can be called if an external change might have fixed the underlying issue.
        """
        logger.info("Attempting to re-initialize primary RAG pipeline.")
        try:
            self.primary_pipeline = BasicRAGPipeline(
                connection_manager=self.connection_manager,
                config_manager=self.config_manager
            )
            self.is_primary_pipeline_healthy = True
            logger.info("Primary BasicRAGPipeline re-initialized successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to re-initialize primary BasicRAGPipeline: {e}. Still in fallback mode.", exc_info=True)
            self.primary_pipeline = None
            self.is_primary_pipeline_healthy = False
            return False

# Example Usage (for illustration)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Scenario 1: Primary pipeline initializes and works
    print("\n--- Scenario 1: Primary pipeline works ---")
    # Mock a config that allows BasicRAGPipeline to initialize (even if it can't fully connect)
    mock_config_working = {
        "iris_host": "localhost", "iris_port": 1972, "iris_namespace": "USER",
        "iris_user": "user", "iris_password": "password",
        "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2", # Mock, won't load
        "llm_model_name": "mock-llm" # Mock
    }
    cfg_manager_working = ConfigurationManager(config=mock_config_working)
    conn_manager_working = ConnectionManager(config_manager=cfg_manager_working)
    
    # Mock BasicRAGPipeline's query method for this test
    class MockBasicRAGPipeline(BasicRAGPipeline):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Override actual initializations that might fail if IRIS/models not present
            self.embedding_model = None 
            self.llm = None
            self.iris_connector = conn_manager_working.get_iris_connection() # Simulate getting it

        def query(self, query_text: str, **kwargs: Any) -> Dict[str, Any]:
            if query_text == "error_trigger":
                raise ValueError("Simulated pipeline error")
            return {"query": query_text, "answer": f"Primary answer for: {query_text}", "retrieved_documents": [], "source": "PrimaryRAG"}

    primary_pipeline_mock = MockBasicRAGPipeline(connection_manager=conn_manager_working, config_manager=cfg_manager_working)
    
    survival_service_ok = SurvivalModeRAGService(
        connection_manager=conn_manager_working,
        config_manager=cfg_manager_working,
        primary_pipeline=primary_pipeline_mock
    )
    response_ok = survival_service_ok.query("What is RAG?")
    print(f"Response (OK): {response_ok}")

    # Scenario 2: Primary pipeline fails during query
    print("\n--- Scenario 2: Primary pipeline fails during query ---")
    response_query_fail = survival_service_ok.query("error_trigger")
    print(f"Response (Query Fail): {response_query_fail}")
    # Subsequent query should also use fallback
    response_after_fail = survival_service_ok.query("Another query")
    print(f"Response (After Fail): {response_after_fail}")


    # Scenario 3: Primary pipeline fails to initialize
    print("\n--- Scenario 3: Primary pipeline fails to initialize ---")
    mock_config_broken = {"error_on_init": True} # Config that would cause BasicRAGPipeline to fail
    cfg_manager_broken = ConfigurationManager(config=mock_config_broken)
    # We expect BasicRAGPipeline init to fail here
    # For the test, we'll pass None as primary_pipeline and let SurvivalModeRAGService try to init
    
    # To truly test this, BasicRAGPipeline would need to raise an error on init with bad config
    # For now, we simulate by not providing a working primary_pipeline
    # and assuming its internal init would fail.
    # The current SurvivalModeRAGService constructor already tries to init BasicRAGPipeline.
    # We need a way for that internal init to fail for this scenario.
    # Let's assume ConfigurationManager or ConnectionManager would raise error with "error_on_init"
    
    class FailingInitBasicRAGPipeline(BasicRAGPipeline):
        def __init__(self, connection_manager, config_manager, **kwargs):
            if config_manager.get_config("error_on_init"):
                raise ValueError("Simulated initialization failure")
            super().__init__(connection_manager, config_manager, **kwargs)

    # Monkey patch BasicRAGPipeline for this specific test context
    original_basic_rag = survival_mode.BasicRAGPipeline # Save original
    survival_mode.BasicRAGPipeline = FailingInitBasicRAGPipeline # Patch

    survival_service_init_fail = SurvivalModeRAGService(
        config_manager=cfg_manager_broken # This config will cause FailingInitBasicRAGPipeline to fail
    )
    response_init_fail = survival_service_init_fail.query("Hello?")
    print(f"Response (Init Fail): {response_init_fail}")

    survival_mode.BasicRAGPipeline = original_basic_rag # Restore original

    # Attempt reinitialization (assuming the "problem" is fixed)
    print("\n--- Attempting reinitialization (simulating fix) ---")
    # For this to work, the config needs to be "fixed"
    cfg_manager_broken.update_config({"error_on_init": False}) # "Fix" the config
    # And we need to patch BasicRAGPipeline back to a working one for the re-init call
    survival_mode.BasicRAGPipeline = MockBasicRAGPipeline
    
    if survival_service_init_fail.reinitialize_primary_pipeline():
        print("Reinitialization successful.")
        response_after_reinit = survival_service_init_fail.query("Are you back?")
        print(f"Response (After Reinit): {response_after_reinit}")
    else:
        print("Reinitialization failed.")

    survival_mode.BasicRAGPipeline = original_basic_rag # Restore original fully