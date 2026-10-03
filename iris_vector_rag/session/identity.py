"""Which instance, which namespace, and as whom.

An identity is the answer to "are these two requests for the same session?".
The password is not part of that answer: a session that cannot connect with the
password it holds is a failure, not a second session.

Feature: 083-iris-session
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from iris_vector_rag.common.exceptions import ValidationError
from iris_vector_rag.config.connection_settings import (
    ConnectionSettings,
    resolve_connection_settings,
)

_NAMESPACE = re.compile(r"^[A-Za-z0-9_]+$")


@dataclass(frozen=True)
class SessionIdentity:
    """The instance, port, namespace and account that name one session."""

    host: str
    port: int
    namespace: str
    username: str
    password: str = field(default="SYS", compare=False, repr=False)
    # True when every value came from a default: nothing was configured. Errors use
    # it to say so; it is not part of what makes two requests the same session.
    from_defaults: bool = field(default=False, compare=False, repr=False)

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
    def from_settings(cls, settings: ConnectionSettings) -> "SessionIdentity":
        return cls(
            host=settings.host,
            port=settings.port,
            namespace=settings.namespace,
            username=settings.username,
            password=settings.password,
            from_defaults=settings.all_defaults,
        )

    @classmethod
    def from_env(cls) -> "SessionIdentity":
        return cls.from_settings(resolve_connection_settings())

    @classmethod
    def from_mapping(cls, values: Optional[Dict[str, Any]] = None) -> "SessionIdentity":
        """Resolve from plain connection parameters, the resolver filling gaps."""
        return cls.from_settings(resolve_connection_settings(values or {}))

    @classmethod
    def from_config(cls, config_manager: Any) -> "SessionIdentity":
        """Resolve from a ConfigurationManager, the resolver filling gaps."""
        get_settings = getattr(config_manager, "get_connection_settings", None)
        if get_settings is not None:
            return cls.from_settings(get_settings())
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
