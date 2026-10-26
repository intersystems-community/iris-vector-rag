"""The answer model from an AI Hub ConfigStore entry (spec 093).

An IRIS administrator defines ``AI.LLM.<name>`` once, in ``%ConfigStore.Configuration``,
with the key as a Wallet reference (``secret://<Collection>.<Secret>#<field>``). This
module reads the entry over the pipeline's own IRIS connection, lets IRIS resolve the
reference, and hands the entry to LangChain's ``init_chat_model``, as
``langchain-intersystems`` does, without depending on it.

The read uses ``Get`` and ``CopyDetails`` through the released driver's Native API
(``iris.createIRIS``). ``GetDetails`` is not used: on 2026.3 it reports every ``AI.LLM``
entry as not valid.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict

CATEGORY = "AI.LLM."


def _iris() -> Any:
    import iris

    return iris


def _init_chat_model() -> Callable[..., Any]:
    from langchain.chat_models import init_chat_model

    return init_chat_model


def read_llm_entry(name: str, connection: Any) -> Dict[str, Any]:
    """The entry ``AI.LLM.<name>`` as a dict, with its Wallet references resolved."""
    iris = _iris()
    native = iris.createIRIS(connection)
    found = iris.IRISReference(None)
    native.classMethodValue("%ConfigStore.Configuration", "Get", CATEGORY + name, found)
    details = iris.IRISReference(None)
    found.getValue().invoke("CopyDetails", details)
    return json.loads(details.getValue().invoke("%ToJSON"))


def configstore_llm(name: str, connection_manager: Any) -> Callable[[str], str]:
    """The answer function for entry ``AI.LLM.<name>``, read once, now.

    Pass the result as ``llm_func`` to any pipeline. The LangChain model it calls is
    available as its ``chat_model`` attribute.
    """
    entry = read_llm_entry(name, connection_manager.get_connection())
    chat = _init_chat_model()(**entry)

    def answer(prompt: str) -> str:
        return chat.invoke(prompt).content

    answer.chat_model = chat  # type: ignore[attr-defined]
    return answer
