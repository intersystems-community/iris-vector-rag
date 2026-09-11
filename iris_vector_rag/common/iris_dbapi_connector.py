"""
Connector for InterSystems IRIS using Python DBAPI.

Since 0.15 this is an adapter over `iris_vector_rag.session`: the session decides
which driver to use and says so, instead of every module racing to import one at
import time. Prefer `session_for(...)` in new code.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def get_iris_dbapi_connection(config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Unified entry point for IRIS DBAPI connections.
    """
    from iris_vector_rag.session import SessionIdentity, session_for

    identity = SessionIdentity.from_mapping(config) if config else None
    return session_for(identity=identity).connection_for_adapter()


def get_iris_dbapi_module():
    """
    Access the IRIS DBAPI module.
    """
    from iris_vector_rag.common.iris_connection import _get_iris_dbapi_module

    return _get_iris_dbapi_module()


def __getattr__(name: str) -> Any:
    """`irisdbapi` stays available, but importing this module imports no driver.

    The eager module-level lookup this replaces decided the process's driver as a
    side effect of an import, which is why nothing could report the choice.
    """
    if name == "irisdbapi":
        return get_iris_dbapi_module()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
