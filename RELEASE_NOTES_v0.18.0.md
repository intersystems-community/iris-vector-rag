# Release Notes: iris-vector-rag v0.18.0

Released 2026-10-27 as `v0.18.0`.

**The answer model from AI Hub ConfigStore.** On IRIS 2026.3 with AI Hub, an administrator
defines a model once, inside IRIS, as a ConfigStore entry `AI.LLM.<name>`, with its API key kept
in the IRIS Wallet. Name the entry, and a pipeline takes its answer model from it: no key in the
environment, and no second copy of the model settings. See
[docs/LLM_FROM_CONFIGSTORE.md](docs/LLM_FROM_CONFIGSTORE.md).

```python
from iris_vector_rag import create_pipeline

pipeline = create_pipeline("basic", llm_config="opsreview")
```

## Highlights

- **Three ways to name the entry:** `create_pipeline(..., llm_config=...)`, the `llm.config_name`
  setting (`RAG_LLM__CONFIG_NAME`), or `llm_func=configstore_llm(name, connections)` for a
  pipeline you build yourself.
- **Read once, when the pipeline is built,** over the pipeline's own IRIS connection. IRIS
  resolves the Wallet reference, and the entry goes to LangChain's `init_chat_model` unchanged,
  as `langchain-intersystems` does. No key is taken from the environment.
- **Precedence:** an explicit `llm_func`, then the entry, then the YAML `llm:` settings.
- **One clear error, `LLMConfigError`,** for every way an entry can fail, naming the entry and
  the fix:
  - an invalid name, refused before IRIS is asked;
  - a missing or disabled entry;
  - a missing `model` or `model_provider`;
  - a Wallet reference the user cannot read (the message gives the reference, never a value);
  - a missing `%Native_ClassExecution:USE`;
  - an IRIS without ConfigStore, with its version;
  - a provider package that is not installed, with the package to install.

## Upgrading

Nothing changes unless you name an entry. The connecting user needs two privileges, both listed
in the guide: `%Native_ClassExecution:USE`, and `%Admin_Wallet:USE` or READ on the Wallet
collection's resource.

## Verification

- Unit: 2,376 passed, 5 skipped.
- Integration on IRIS for Health Community 2026.3.0AI.126 (a throwaway lab with a real Wallet
  secret and users without each privilege): 8 passed. The version-gate test on IRIS 2026.1:
  passed.
- The full integration suite on IRIS 2026.1: 512 passed. Its failures are the tests that need a
  real OpenAI key, and they fail the same way on 0.17.0.
- The release workflow's test steps were replayed on a fresh container before tagging.
