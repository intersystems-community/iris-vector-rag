"""
IRIS Global GraphRAG Web Interface

Provides a web interface for the IRIS Global GraphRAG pipeline with interactive
graph visualization and side-by-side comparison capabilities.

This integrates the visualization components from the IRIS-Global-GraphRAG project
with our RAG framework's pipeline system.
"""

import logging
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, render_template, request, send_from_directory

from ..pipelines.iris_global_graphrag import (
    IRISGlobalGraphRAGException,
    IRISGlobalGraphRAGPipeline,
)

logger = logging.getLogger(__name__)


class IRISGlobalGraphRAGInterface:
    """
    Web interface for IRIS Global GraphRAG pipeline with visualization.

    Features:
    - Interactive graph visualization using 3D Force Graph
    - Side-by-side comparison of LLM vs RAG vs GraphRAG
    - Academic paper search and retrieval
    - Real-time graph network visualization
    """

    def __init__(self, pipeline: IRISGlobalGraphRAGPipeline):
        self.pipeline = pipeline
        self.template_dir = Path(__file__).parent / "iris_global_graphrag"
        self.static_dir = self.template_dir / "css"

        # Validate template directory exists
        if not self.template_dir.exists():
            raise IRISGlobalGraphRAGException(
                f"Template directory not found: {self.template_dir}. "
                "Please ensure visualization assets are properly installed."
            )

    def create_flask_app(self, app_name: str = "iris_global_graphrag") -> Flask:
        """
        Create and configure Flask application with routes.

        Args:
            app_name: Name of the Flask application

        Returns:
            Configured Flask application
        """
        app = Flask(
            app_name,
            template_folder=str(self.template_dir),
            static_folder=str(self.static_dir),
        )

        # Configure routes
        self._setup_routes(app)

        return app

    def _setup_routes(self, app: Flask):
        """Setup Flask routes for the interface."""

        @app.route("/")
        @app.route("/graphrag")
        def graphrag_page():
            """Main GraphRAG interface page."""
            return render_template("graphrag.html")

        @app.route("/rag")
        def rag_page():
            """RAG-only interface page."""
            return render_template("rag.html")

        @app.route("/llm")
        def llm_page():
            """LLM-only interface page."""
            return render_template("llm.html")

        @app.route("/llm-vs-rag")
        def llm_vs_rag_page():
            """LLM vs RAG comparison page."""
            return render_template("llm_vs_rag.html")

        @app.route("/rag-vs-graphrag")
        def rag_vs_graphrag_page():
            """RAG vs GraphRAG comparison page."""
            return render_template("rag_vs_graphrag.html")

        @app.route("/multi-pipeline-comparison")
        def multi_pipeline_comparison_page():
            """Multi-pipeline comparison page (IRIS Global vs Hybrid vs Standard GraphRAG)."""
            return render_template("multi_pipeline_comparison.html")

        # API Routes

        @app.route("/api/ask", methods=["POST"])
        def api_ask():
            """Standard RAG query endpoint."""
            try:
                data = request.get_json(force=True, silent=True) or {}
                query = (data.get("query") or "").strip()
                top_k = int(data.get("top_k") or 5)

                if not query:
                    return jsonify({"error": "Missing 'query'"}), 400

                # Use RAG mode
                result = self.pipeline.query(query, mode="rag", top_k=top_k)

                return jsonify(
                    {
                        "answer": result["answer"],
                        "processing_time": result.get("processing_time", 0),
                    }
                )

            except Exception as e:
                logger.error(f"RAG query failed: {e}")
                return jsonify({"error": str(e)}), 500

        @app.route("/api/graphrag", methods=["POST"])
        def api_graphrag():
            """GraphRAG query endpoint with visualization data."""
            try:
                data = request.get_json(force=True, silent=True) or {}
                query = (data.get("query") or "").strip()
                top_k = int(data.get("top_k") or 5)

                if not query:
                    return jsonify({"error": "Missing 'query'"}), 400

                # Use GraphRAG mode with visualization
                result = self.pipeline.query(
                    query, mode="graphrag", top_k=top_k, enable_visualization=True
                )

                response = {
                    "answer": result["answer"],
                    "processing_time": result.get("processing_time", 0),
                }

                # Add graph data for visualization
                if "graph_data" in result:
                    response["graph"] = result["graph_data"]

                if "retrieved_papers" in result:
                    response["papers"] = result["retrieved_papers"]

                return jsonify(response)

            except Exception as e:
                logger.error(f"GraphRAG query failed: {e}")
                return jsonify({"error": str(e)}), 500

        @app.route("/api/compare", methods=["POST"])
        def api_compare():
            """Compare LLM, RAG, and GraphRAG responses."""
            try:
                data = request.get_json(force=True, silent=True) or {}
                query = (data.get("query") or "").strip()
                top_k = int(data.get("top_k") or 5)

                if not query:
                    return jsonify({"error": "Missing 'query'"}), 400

                # Use comparison method
                result = self.pipeline.compare_modes(query, top_k=top_k)

                return jsonify(result)

            except Exception as e:
                logger.error(f"Comparison failed: {e}")
                return jsonify({"error": str(e)}), 500

        @app.route("/api/llm", methods=["POST"])
        def api_llm():
            """LLM-only query endpoint."""
            try:
                data = request.get_json(force=True, silent=True) or {}
                query = (data.get("query") or "").strip()

                if not query:
                    return jsonify({"error": "Missing 'query'"}), 400

                # Send directly to LLM without retrieval
                llm_response = self.pipeline.global_graphrag_module.send_to_llm(
                    [
                        {
                            "role": "user",
                            "content": f"Answer this question concisely: {query}",
                        }
                    ]
                )

                answer = llm_response.choices[0].message.content

                return jsonify({"answer": answer, "mode": "llm_only"})

            except Exception as e:
                logger.error(f"LLM query failed: {e}")
                return jsonify({"error": str(e)}), 500

        @app.route("/api/pipeline/info", methods=["GET"])
        def api_pipeline_info():
            """Get pipeline information and status."""
            try:
                info = self.pipeline.get_pipeline_info()
                return jsonify(info)
            except Exception as e:
                logger.error(f"Failed to get pipeline info: {e}")
                return jsonify({"error": str(e)}), 500

        @app.route("/api/compare/multi-pipeline", methods=["POST"])
        def api_multi_pipeline_compare():
            """Compare multiple pipeline types including HybridGraphRAG and GraphRAG."""
            try:
                from .multi_pipeline_comparator import MultiPipelineComparator

                data = request.get_json(force=True, silent=True) or {}
                query = (data.get("query") or "").strip()
                pipeline_types = data.get(
                    "pipeline_types", None
                )  # None = all available
                include_llm = data.get("include_llm_baseline", True)
                top_k = int(data.get("top_k") or 5)
                parallel = data.get("parallel_execution", True)

                if not query:
                    return jsonify({"error": "Missing 'query'"}), 400

                # Create comparator
                comparator = MultiPipelineComparator(self.pipeline.config_manager)

                # Run comparison
                result = comparator.compare_pipelines(
                    query=query,
                    pipeline_types=pipeline_types,
                    include_llm_baseline=include_llm,
                    top_k=top_k,
                    parallel_execution=parallel,
                )

                return jsonify(result)

            except Exception as e:
                logger.error(f"Multi-pipeline comparison failed: {e}")
                return jsonify({"error": str(e)}), 500

        @app.route("/api/compare/available-pipelines", methods=["GET"])
        def api_available_pipelines():
            """Get list of available pipelines for comparison."""
            try:
                from .multi_pipeline_comparator import MultiPipelineComparator

                comparator = MultiPipelineComparator(self.pipeline.config_manager)
                available = comparator.get_available_pipelines()

                return jsonify(available)

            except Exception as e:
                logger.error(f"Failed to get available pipelines: {e}")
                return jsonify({"error": str(e)}), 500

        @app.route("/css/<path:filename>")
        def serve_css(filename):
            """Serve CSS files."""
            return send_from_directory(self.static_dir, filename)

    def run_standalone(
        self, host: str = "0.0.0.0", port: int = 8000, debug: bool = False
    ):
        """
        Run the interface as a standalone Flask application.

        Args:
            host: Host address to bind to
            port: Port to listen on
            debug: Enable Flask debug mode
        """
        app = self.create_flask_app()

        logger.info(f"Starting IRIS Global GraphRAG interface on {host}:{port}")
        logger.info(f"GraphRAG interface: http://{host}:{port}/")
        logger.info(f"RAG interface: http://{host}:{port}/rag")
        logger.info(f"LLM interface: http://{host}:{port}/llm")
        logger.info(
            f"Comparison interfaces: http://{host}:{port}/llm-vs-rag, http://{host}:{port}/rag-vs-graphrag"
        )

        app.run(host=host, port=port, debug=debug)


def create_interface_from_config(
    config_path: Optional[str] = None,
) -> IRISGlobalGraphRAGInterface:
    """
    Create IRIS Global GraphRAG interface from configuration.

    Args:
        config_path: Path to configuration file (optional)

    Returns:
        Configured interface instance
    """
    from ..config.manager import ConfigurationManager
    from ..pipelines.factory import create_pipeline

    # Load configuration
    config_manager = (
        ConfigurationManager(config_path) if config_path else ConfigurationManager()
    )

    # Create pipeline
    pipeline = create_pipeline(
        pipeline_type="IRISGlobalGraphRAG",
        config_manager=config_manager,
        validate_requirements=True,
    )

    # Create interface
    interface = IRISGlobalGraphRAGInterface(pipeline)

    return interface


def main():
    """Main entry point for standalone execution."""
    import argparse

    parser = argparse.ArgumentParser(description="IRIS Global GraphRAG Web Interface")
    parser.add_argument("--host", default="0.0.0.0", help="Host address to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--config", help="Path to configuration file")

    args = parser.parse_args()

    try:
        # Create interface
        interface = create_interface_from_config(args.config)

        # Run standalone
        interface.run_standalone(host=args.host, port=args.port, debug=args.debug)

    except Exception as e:
        logger.error(f"Failed to start interface: {e}")
        raise


if __name__ == "__main__":
    main()
