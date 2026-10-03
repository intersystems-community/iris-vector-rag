"""Which instance, which namespace, and as whom.

An identity is the answer to "are these two requests for the same session?".
The password is not part of that answer: a session that cannot connect with the
password it holds is a failure, not a second session.

Feature: 083-iris-session
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from iris_vector_rag.common.exceptions import ValidationError

_NAMESPACE = re.compile(r"^[A-Za-z0-9_]+$")


def _env_defaults() -> Dict[str, Any]:
    """Connection parameters as the environment describes them."""
    from iris_vector_rag.common.iris_connection import auto_detect_iris_port

    port_env = os.environ.get("IRIS_PORT")
    port = int(port_env) if port_env else (auto_detect_iris_port() or 1972)
    return {
        "host": os.environ.get("IRIS_HOST", "localhost"),
        "port": port,
        "namespace": os.environ.get("IRIS_NAMESPACE", "USER"),
        "username": os.environ.get(
            "IRIS_USER", os.environ.get("IRIS_USERNAME", "_SYSTEM")
        ),
        "password": os.environ.get("IRIS_PASSWORD", "SYS"),
    }


@dataclass(frozen=True)
class SessionIdentity:
    """The instance, port, namespace and account that name one session."""

    host: str
    port: int
    namespace: str
    username: str
    password: str = field(default="SYS", compare=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.host, str) or not self.host.strip():
            raise ValidationError(
                "host", self.host, "non-empty string", "Host cannot be empty"
            )
        if (
            isinstance(self.port, bool)
            or not isinstance(self.port, int)
            or not 1 <= self.port <= 65535
        ):
            raise ValidationError(
                "port", self.port, "1-65535", f"Invalid port: {self.port}"
            )
        if not isinstance(self.namespace, str) or not _NAMESPACE.match(self.namespace):
            raise ValidationError(
                "namespace",
                self.namespace,
                "alphanumeric + underscores",
                f"Invalid namespace: {self.namespace}",
            )
        if not isinstance(self.username, str) or not self.username.strip():
            raise ValidationError(
                "username",
                self.username,
                "non-empty string",
                "Username cannot be empty",
            )

    @property
    def key(self) -> Tuple[str, int, str, str]:
        """What makes two requests the same session."""
        return (self.host, self.port, self.namespace, self.username)

    @property
    def label(self) -> str:
        """How this instance is named in every log line and every error."""
        return f"{self.host}:{self.port}/{self.namespace}"

    @classmethod
    def from_env(cls) -> "SessionIdentity":
        return cls(**_env_defaults())

    @classmethod
    def from_mapping(cls, values: Optional[Dict[str, Any]] = None) -> "SessionIdentity":
        """Resolve from plain connection parameters, the environment filling gaps."""
        resolved = _env_defaults()
        given = values or {}
        for name in ("host", "port", "namespace", "username", "password"):
            value = given.get(name)
            if value is None and name == "host":
                value = given.get("hostname")  # the driver's own spelling
            if value is not None:
                resolved[name] = value
        if not isinstance(resolved["port"], int):
            try:
                resolved["port"] = int(resolved["port"])
            except (TypeError, ValueError):
                pass
        return cls(**resolved)

    @classmethod
    def from_config(cls, config_manager: Any) -> "SessionIdentity":
        """Resolve from a ConfigurationManager, filling gaps from the environment."""
        try:
            configured = config_manager.get_database_config() or {}
        except Exception:  # pragma: no cover - a config manager that cannot answer
            configured = {}
        return cls.from_mapping(configured)

    @classmethod
    def resolve(
        cls,
        config_manager: Optional[Any] = None,
        identity: Optional["SessionIdentity"] = None,
    ) -> "SessionIdentity":
        """The one place that decides which identity a caller meant."""
        if identity is not None:
            return identity
        if config_manager is not None:
            return cls.from_config(config_manager)
        return cls.from_env()
