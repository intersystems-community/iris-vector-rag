"""The answer model from an AI Hub ConfigStore entry (spec 093).

An IRIS administrator defines ``AI.LLM.<name>`` once, in ``%ConfigStore.Configuration``,
with the key as a Wallet reference (``secret://<Collection>.<Secret>#<field>``). This
module reads the entry over the pipeline's own IRIS connection, lets IRIS resolve the
reference, and hands the entry to LangChain's ``init_chat_model``, as
``langchain-intersystems`` does, without depending on it.

The read uses ``Get`` and ``CopyDetails`` through the released driver's Native API
(``iris.createIRIS``). ``GetDetails`` is not used: on 2026.3 it reports every ``AI.LLM``
entry as not valid.

Every failure is an ``LLMConfigError`` raised here, at build time. Messages are built
from the entry's unresolved form, so a resolved key never reaches one.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, Iterator, List

from iris_vector_rag.exceptions import LLMConfigError

CATEGORY = "AI.LLM."
NAME_RULE = r"^[A-Za-z][A-Za-z0-9_-]*$"
REQUIRED_KEYS = ("model", "model_provider")
SECRET_PREFIX = "secret://"

WALLET_FIX = (
    "grant the connecting user %Admin_Wallet:USE, or READ on the Wallet collection's "
    "UseResource"
)


def _iris() -> Any:
    import iris

    return iris


def _init_chat_model() -> Callable[..., Any]:
    from langchain.chat_models import init_chat_model

    return init_chat_model


def _references(value: Any) -> Iterator[str]:
    """Every ``secret://`` string in an entry, however deep."""
    if isinstance(value, str):
        if value.startswith(SECRET_PREFIX):
            yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _references(item)
    elif isinstance(value, list):
        for item in value:
            yield from _references(item)


def _wallet_error(name: str, references: List[str], detail: str = "") -> LLMConfigError:
    named = ", ".join(f"`{ref}`" for ref in references) or "a Wallet reference"
    problem = f"refers to the Wallet ({named}), which the connecting user cannot read"
    if detail:
        problem += f" ({detail})"
    return LLMConfigError(name, problem, WALLET_FIX)


class _Reader:
    """One read of one entry, mapping each server failure to its LLMConfigError."""

    def __init__(self, name: str, connection: Any):
        self.name = name
        self.iris = _iris()
        self.native = self.iris.createIRIS(connection)

    def error_text(self, status: Any) -> str:
        return str(
            self.native.classMethodValue("%SYSTEM.Status", "GetErrorText", status)
        )

    def version(self) -> str:
        return str(self.native.classMethodValue("%SYSTEM.Version", "GetNumber"))

    def get(self) -> Any:
        found = self.iris.IRISReference(None)
        try:
            status = self.native.classMethodValue(
                "%ConfigStore.Configuration", "Get", CATEGORY + self.name, found
            )
        except Exception as exc:
            raise self.native_failure(exc) from exc
        if status != 1:
            text = self.error_text(status)
            if "#5809" in text:
                raise LLMConfigError(
                    self.name,
                    "does not exist",
                    "check the name (names are case-sensitive), or ask the IRIS "
                    "administrator to create the entry",
                )
            raise self.unreadable(text)
        return found.getValue()

    def unreadable(self, text: str) -> LLMConfigError:
        return LLMConfigError(
            self.name,
            f"could not be read: {text}",
            "check the entry in %ConfigStore.Configuration",
        )

    def native_failure(self, exc: Exception) -> Exception:
        text = str(exc)
        if "<PROTECT>" in text and "%Native_ClassExecution" in text:
            return LLMConfigError(
                self.name,
                "cannot be read: the connecting user may not run class methods",
                "grant the connecting user a role with %Native_ClassExecution:USE",
            )
        if "<CLASS DOES NOT EXIST>" in text:
            return LLMConfigError(
                self.name,
                f"cannot be read: this IRIS ({self.version()}) has no "
                f"%ConfigStore.Configuration",
                "use IRIS 2026.3 or later with AI Hub",
            )
        return exc

    def details(self, configuration: Any, resolve: bool) -> Dict[str, Any]:
        copy = self.iris.IRISReference(None)
        args = (copy,) if resolve else (copy, 0)
        status = configuration.invoke("CopyDetails", *args)
        if status != 1:
            if resolve:
                stored = self.details(configuration, resolve=False)
                raise _wallet_error(
                    self.name, list(_references(stored)), self.error_text(status)
                )
            raise self.unreadable(self.error_text(status))
        return json.loads(copy.getValue().invoke("%ToJSON"))

    def read(self) -> Dict[str, Any]:
        configuration = self.get()
        if configuration.get("Enabled") in (0, "0", False):
            raise LLMConfigError(
                self.name,
                "is disabled",
                "enable the entry in ConfigStore, or name another one",
            )
        entry = self.details(configuration, resolve=True)
        for key in REQUIRED_KEYS:
            if key not in entry:
                raise LLMConfigError(
                    self.name, f"has no `{key}` key", f"add `{key}` to the entry"
                )
        leftover = list(_references(entry))
        if leftover:
            raise _wallet_error(self.name, leftover)
        return entry


def read_llm_entry(name: str, connection: Any) -> Dict[str, Any]:
    """The entry ``AI.LLM.<name>`` as a dict, with its Wallet references resolved.

    Raises:
        LLMConfigError: for an invalid name (before IRIS is asked), a missing or
            disabled entry, a missing ``model`` or ``model_provider``, a Wallet
            reference the user cannot read, a missing privilege, or an IRIS without
            ConfigStore.
    """
    if not isinstance(name, str) or not re.match(NAME_RULE, name):
        raise LLMConfigError(
            str(name),
            "is not a valid ConfigStore name",
            f"use a name matching {NAME_RULE}",
        )
    if connection is None:
        raise LLMConfigError(
            name,
            "cannot be read without an IRIS connection",
            "build the pipeline with a connection to the IRIS that holds the entry",
        )
    return _Reader(name, connection).read()


def _chat_model(name: str, entry: Dict[str, Any]) -> Any:
    try:
        return _init_chat_model()(**entry)
    except ImportError as exc:
        found = re.search(r"pip install (?:-U )?([\w.-]+)", str(exc))
        package = found.group(1) if found else f"langchain-{entry['model_provider']}"
        raise LLMConfigError(
            name,
            f"names provider `{entry['model_provider']}`, whose package is not installed",
            f"pip install {package}",
        ) from exc


def configstore_llm(name: str, connection_manager: Any) -> Callable[[str], str]:
    """The answer function for entry ``AI.LLM.<name>``, read once, now.

    Pass the result as ``llm_func`` to any pipeline. The LangChain model it calls is
    available as its ``chat_model`` attribute.

    Raises:
        LLMConfigError: if the entry gives no usable model; see `read_llm_entry`.
    """
    entry = read_llm_entry(name, connection_manager.get_connection())
    chat = _chat_model(name, entry)

    def answer(prompt: str) -> str:
        return chat.invoke(prompt).content

    answer.chat_model = chat  # type: ignore[attr-defined]
    return answer
