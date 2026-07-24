"""
IrisLLMDSPyAdapter — DSPy-compatible LM adapter wrapping iris_llm.ChatIris.

Follows the same pattern as DirectOpenAILM in dspy_modules/entity_extraction_module.py.
Requires iris_llm to be installed; raises ImportError on construction if absent.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class IrisLLMDSPyAdapter:
    """
    DSPy-compatible LM adapter wrapping ``iris_llm.ChatIris``.

    Usage::

        from iris_llm import Provider
        from iris_llm.langchain import ChatIris
        import dspy

        provider = Provider.new_openai(api_key="...")
        chat = ChatIris(provider=provider, model="gpt-4o-mini")
        lm = IrisLLMDSPyAdapter(chat_iris=chat)
        dspy.configure(lm=lm)

    Thread safety: create one instance per thread for DSPy parallel optimisation.
    """

    def __init__(self, chat_iris: Any, model: str = "iris_llm", **kwargs: Any) -> None:
        try:
            import dspy  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "IrisLLMDSPyAdapter requires dspy-ai. "
                "Install with: pip install dspy-ai"
            ) from exc

        # Replicate minimum DSPy BaseLM attributes without importing dspy at
        # module level — keeps the module importable without dspy installed.
        self.model = model
        self.provider = "openai"  # DSPy tool-call compatibility
        self.kwargs: dict[str, Any] = {"model": model, **kwargs}
        self._chat = chat_iris

    def __call__(
        self,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Invoke the underlying ChatIris and return ``list[str]`` for DSPy."""
        from langchain_core.messages import HumanMessage, SystemMessage

        if messages:
            lc_messages = [
                (
                    SystemMessage(content=m["content"])
                    if m.get("role") == "system"
                    else HumanMessage(content=m["content"])
                )
                for m in messages
            ]
        else:
            lc_messages = [HumanMessage(content=prompt or "")]

        response = self._chat.invoke(lc_messages)
        return [response.content]

    def basic_request(self, prompt: str, **kwargs: Any) -> list[str]:
        """Delegate to ``__call__``; present for DSPy BaseLM compatibility."""
        return self(prompt=prompt, **kwargs)
