#!/usr/bin/env python3
"""
Smoke test for HybridGraphRAG pipeline import and initialization.

This test verifies:
- Module import succeeds
- Pipeline class can be instantiated with default managers
- iris_graph_core availability is detected and integration is optional
- No DB calls are executed (no query), so it is safe without IRIS running

Exit codes:
- 0: Success
- 1: Import/instantiation failure
"""

import json
import logging
import sys
from pathlib import Path

# Ensure project root on path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("hybrid_graphrag_smoke")


def main() -> int:
    try:
        from iris_vector_rag.config.manager import ConfigurationManager
        from iris_vector_rag.core.connection import ConnectionManager
        from iris_vector_rag.pipelines.hybrid_graphrag import HybridGraphRAGPipeline

        logger.info("✅ Import successful: HybridGraphRAGPipeline")

        # Instantiate with default managers (should not connect until query)
        conn_mgr = ConnectionManager()
        config_mgr = ConfigurationManager()
        pipeline = HybridGraphRAGPipeline(conn_mgr, config_mgr)

        logger.info("✅ Instantiation successful")

        # Get hybrid status using new API
        hybrid_status = pipeline.get_hybrid_status()

        result = {
            "import_success": True,
            "instantiation_success": True,
            "hybrid_enabled": pipeline.is_hybrid_enabled(),
            "status_details": hybrid_status,
            "discovery_available": pipeline.discovery is not None,
            "retrieval_methods_available": pipeline.retrieval_methods is not None,
            "note": "No queries executed; safe without IRIS DB running.",
        }

        logger.info(f"🔍 Hybrid enabled: {result['hybrid_enabled']}")
        logger.info(
            f"📊 Graph core path: {hybrid_status.get('graph_core_path', 'Not found')}"
        )

        print(json.dumps(result, indent=2))
        return 0

    except Exception as e:
        logger.exception("❌ Smoke test failed")
        print(
            json.dumps(
                {
                    "import_success": False,
                    "instantiation_success": False,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                indent=2,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
