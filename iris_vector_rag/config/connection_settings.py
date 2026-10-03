"""The one place that decides which IRIS a process talks to.

Precedence, highest first:

1. explicit settings (arguments, or the user-written ``database.iris`` config section);
2. the process environment;
3. a ``.env`` file in the working directory (read, never loaded into ``os.environ``);
4. the pinned container: ``IRIS_CONTAINER_NAME`` names one container, and that
   container's published port fills in a port nothing above supplied;
5. the ``database.iris`` section of the YAML configuration file (the pin arrives by
   environment, which beats the file, so it sits above it);
6. defaults: ``localhost:1972``, namespace ``USER``, ``_SYSTEM`` / ``SYS``.

Variable names are iris-devtester's: ``IRIS_HOST``, ``IRIS_PORT``, ``IRIS_NAMESPACE``,
``IRIS_USERNAME``, ``IRIS_PASSWORD``, ``IRIS_TIMEOUT``, ``IRIS_CONTAINER_NAME``.
``IRIS_USER`` is a deprecated alias of ``IRIS_USERNAME``.

Nothing here scans Docker. The only Docker call is ``docker port <name> 1972/tcp`` for
the exact container the caller pinned.

Tickets: G5 (connection defaults policy), G11 (password recovery).
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Set, Tuple, Union

from iris_vector_rag.common.exceptions import ValidationError

logger = logging.getLogger(__name__)

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 1972
DEFAULT_NAMESPACE = "USER"
DEFAULT_USERNAME = "_SYSTEM"
DEFAULT_PASSWORD = "SYS"
DEFAULT_TIMEOUT = 30

SOURCE_EXPLICIT = "explicit"
SOURCE_ENVIRONMENT = "environment"
SOURCE_DOTENV = "dotenv"
SOURCE_CONTAINER = "container"
SOURCE_CONFIG_FILE = "config_file"
SOURCE_DEFAULT = "default"

_FIELDS = ("host", "port", "namespace", "username", "password", "timeout")
_ENV_NAMES = {
    "host": "IRIS_HOST",
    "port": "IRIS_PORT",
    "namespace": "IRIS_NAMESPACE",
    "username": "IRIS_USERNAME",
    "password": "IRIS_PASSWORD",
    "timeout": "IRIS_TIMEOUT",
}
_DEFAULTS = {
    "host": DEFAULT_HOST,
    "port": DEFAULT_PORT,
    "namespace": DEFAULT_NAMESPACE,
    "username": DEFAULT_USERNAME,
    "password": DEFAULT_PASSWORD,
    "timeout": DEFAULT_TIMEOUT,
}

# What has been logged already, so the line appears once per distinct target.
_logged: Set[Tuple[str, int, str, str]] = set()


class ConnectionSettingsError(ValidationError):
    """A connection setting cannot be used. Names the variable and the value."""

    def __init__(self, variable: str, value: Any, expected: str, message: str):
        super().__init__(variable, value, expected, message)


@dataclass(frozen=True)
class ConnectionSettings:
    """Resolved connection settings, with where each value came from.

    Field names are iris-devtester's ``IRISConfig`` field names.
    """

    host: str
    port: int
    namespace: str
    username: str
    password: str = field(repr=False)
    timeout: int = DEFAULT_TIMEOUT
    container_name: Optional[str] = None
    sources: Dict[str, str] = field(default_factory=dict, compare=False)

    @property
    def all_defaults(self) -> bool:
        """True when nothing was configured: no value, no pin, only defaults."""
        return self.container_name is None and all(
            source == SOURCE_DEFAULT for source in self.sources.values()
        )

    @property
    def label(self) -> str:
        return f"{self.host}:{self.port}"

    def to_dict(self) -> Dict[str, Any]:
        """The connection parameters, as the driver and the config manager spell them."""
        return {
            "host": self.host,
            "port": self.port,
            "namespace": self.namespace,
            "username": self.username,
            "password": self.password,
            "timeout": self.timeout,
        }


def _docker_port(name: str) -> Optional[int]:
    """Host port Docker publishes for container ``name``'s 1972, by exact name."""
    try:
        result = subprocess.run(
            ["docker", "port", name, "1972/tcp"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    match = re.search(r":(\d+)\s*$", result.stdout.strip().splitlines()[0])
    return int(match.group(1)) if match else None


def _present(value: Any) -> bool:
    return value is not None and value != ""


def _explicit_value(explicit: Any, name: str) -> Any:
    if explicit is None:
        return None
    if isinstance(explicit, Mapping):
        value = explicit.get(name)
        if value is None and name == "host":
            value = explicit.get("hostname")  # the driver's own spelling
        return value
    return getattr(explicit, name, None)  # an iris-devtester IRISConfig, as is


def _read_dotenv(dotenv_path: Union[str, Path, None]) -> Dict[str, str]:
    path = Path(dotenv_path) if dotenv_path is not None else Path.cwd() / ".env"
    if not path.is_file():
        return {}
    from dotenv import dotenv_values

    return {k: v for k, v in dotenv_values(path).items() if v is not None}


def _coerce(name: str, value: Any, variable: str) -> Any:
    if name == "port":
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = None
        if number is None or isinstance(value, bool) or not 1 <= number <= 65535:
            raise ConnectionSettingsError(
                variable,
                value,
                "an integer from 1 to 65535",
                f"{variable} must be an integer from 1 to 65535, got: {value!r}",
            )
        return number
    if name == "timeout":
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = 0
        if number < 1:
            raise ConnectionSettingsError(
                variable,
                value,
                "a positive number of seconds",
                f"{variable} must be a positive integer, got: {value!r}",
            )
        return number
    return str(value)


def resolve_connection_settings(
    explicit: Any = None,
    *,
    config_file: Optional[Mapping[str, Any]] = None,
    environ: Optional[Mapping[str, str]] = None,
    dotenv_path: Union[str, Path, None] = None,
    docker_port: Optional[Callable[[str], Optional[int]]] = None,
) -> ConnectionSettings:
    """Resolve the connection settings. See the module docstring for precedence.

    ``explicit`` is a mapping or an object with iris-devtester ``IRISConfig`` field
    names (``host``, ``port``, ``namespace``, ``username``, ``password``, ``timeout``,
    ``container_name``); ``None`` means "not given". ``config_file`` is the
    user-written connection section of the YAML configuration, same keys.
    """
    env: Mapping[str, str] = os.environ if environ is None else environ
    filed = _read_dotenv(dotenv_path)

    def lookup(variable: str) -> Tuple[Optional[str], Optional[str]]:
        if _present(env.get(variable)):
            return env[variable], SOURCE_ENVIRONMENT
        if _present(filed.get(variable)):
            return filed[variable], SOURCE_DOTENV
        return None, None

    values: Dict[str, Any] = {}
    sources: Dict[str, str] = {}
    for name in _FIELDS:
        variable = _ENV_NAMES[name]
        given = _explicit_value(explicit, name)
        # An explicit "" is a value, and the caller hears that it is invalid; only
        # None means "not given". In the environment "" means unset, as in devtester.
        if given is not None:
            values[name] = _coerce(name, given, name)
            sources[name] = SOURCE_EXPLICIT
            continue
        raw, source = lookup(variable)
        if raw is None and name == "username":
            raw, source = lookup("IRIS_USER")
            if raw is not None:
                warnings.warn(
                    "IRIS_USER is deprecated and is removed after 0.16; "
                    "use IRIS_USERNAME.",
                    DeprecationWarning,
                    stacklevel=3,
                )
        if raw is not None:
            values[name] = _coerce(name, raw, variable)
            sources[name] = source  # type: ignore[assignment]

    container = _explicit_value(explicit, "container_name")
    if not _present(container):
        container, _ = lookup("IRIS_CONTAINER_NAME")
    container = container if _present(container) else None

    if "port" not in values and container:
        port = (docker_port or _docker_port)(container)
        if port is None:
            raise ConnectionSettingsError(
                "IRIS_CONTAINER_NAME",
                container,
                "a running container publishing 1972/tcp",
                f"IRIS_CONTAINER_NAME={container!r} names no running container that "
                f"publishes port 1972. Start it (docker start {container}) or set "
                "IRIS_PORT.",
            )
        values["port"] = port
        sources["port"] = SOURCE_CONTAINER

    for name in _FIELDS:
        if name in values:
            continue
        filed_value = _explicit_value(config_file, name)
        if _present(filed_value):
            values[name] = _coerce(name, filed_value, f"database.iris.{name}")
            sources[name] = SOURCE_CONFIG_FILE

    for name in _FIELDS:
        if name not in values:
            values[name] = _DEFAULTS[name]
            sources[name] = SOURCE_DEFAULT

    settings = ConnectionSettings(container_name=container, sources=sources, **values)
    _log_once(settings)
    return settings


def _log_once(settings: ConnectionSettings) -> None:
    key = (settings.host, settings.port, settings.namespace, settings.username)
    if key in _logged:
        return
    _logged.add(key)
    origin = (
        "all defaults"
        if settings.all_defaults
        else ", ".join(
            f"{name} from {source}"
            for name, source in settings.sources.items()
            if name in ("host", "port", "username")
        )
    )
    logger.info(
        "IRIS connection settings: %s as user %s (%s)",
        settings.label,
        settings.username,
        origin,
    )
