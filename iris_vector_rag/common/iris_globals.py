"""
iris_globals — thin wrappers for iris.gset / iris.gget globals access.

All functions degrade gracefully when the ``iris`` DBAPI module is not installed
(e.g. unit test environments, external Python runtimes).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def gset(*path: str, value: str) -> None:
    """
    Set an IRIS global at *path* to *value*.

    No-op (with a debug log) when the ``iris`` module is not installed.

    Example::

        gset("IVR", "Programs", "MyProgram", value="compiled")
    """
    try:
        import iris  # type: ignore[import]

        iris.gset(*path, value)
    except ImportError:
        logger.debug(
            "iris module not available; gset(%s, value=%r) skipped", path, value
        )
    except Exception as exc:
        logger.warning("iris.gset failed: %s", exc)


def gget(*path: str) -> str | None:
    """
    Get an IRIS global at *path*.

    Returns ``None`` (with a debug log) when the ``iris`` module is not installed
    or the global does not exist.

    Example::

        value = gget("IVR", "Programs", "MyProgram")
    """
    try:
        import iris  # type: ignore[import]

        return iris.gget(*path)
    except ImportError:
        logger.debug("iris module not available; gget(%s) returns None", path)
        return None
    except Exception as exc:
        logger.warning("iris.gget failed: %s", exc)
        return None
