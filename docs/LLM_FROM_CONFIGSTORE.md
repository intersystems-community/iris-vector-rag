# The answer model from AI Hub ConfigStore

On IRIS 2026.3 or later with AI Hub, an administrator can define a model once, inside IRIS,
as a ConfigStore entry `AI.LLM.<name>`. The entry holds the model, the provider and settings
such as temperature. Its API key is a reference into the IRIS Wallet, so the key stays under
IRIS access control. Name the entry, and a pipeline takes its answer model from it: no key
in the environment, and no second copy of the model settings.

```python
from iris_vector_rag import create_pipeline

pipeline = create_pipeline("basic", llm_config="opsreview")
```

The entry is read once, when the pipeline is built, over the same IRIS connection that holds
the data. It is handed to LangChain's `init_chat_model` unchanged, as `langchain-intersystems`
does. A change to the entry takes effect in the next pipeline you build.

## Creating the entry (administrator)

In the `%SYS` namespace, create a Wallet collection and secret for the key, then the entry:

```objectscript
 // A resource that guards reading the collection's secrets
 set sc = ##class(Security.Resources).Create("OpsReviewKeys", "Read the opsreview keys", "")
 set sc = ##class(%Wallet.Collection).Create("opsreview", {"UseResource": "OpsReviewKeys", "EditResource": "%Admin_Secure"})
 set sc = ##class(%Wallet.KeyValue).Create("opsreview.openai", {"Usage": "CUSTOM", "Secret": {"api_key": "sk-..."}})

 // The entry: any key init_chat_model accepts, with the key as a Wallet reference
 set entry = {"model": "gpt-4o-mini", "model_provider": "openai", "temperature": 0.1, "api_key": "secret://opsreview.openai#api_key"}
 set sc = ##class(%ConfigStore.Configuration).Create("AI", "LLM", "", "opsreview", entry)
```

- **`model` and `model_provider` are required.** Every other key (`temperature`,
  `base_url`, `max_tokens`, …) reaches LangChain as it is.
- **Names** match `^[A-Za-z][A-Za-z0-9_-]*$` and are case-sensitive: `OpsReview` and
  `opsreview` are different entries.
- **A disabled entry is refused**, not used.

## The Wallet reference

`secret://<Collection>.<Secret>#<field>`, for example `secret://opsreview.openai#api_key`.
IRIS resolves it while the entry is read.

Always include `#field`. Without it, the reference resolves to the whole secret as JSON
text, which reaches the provider as the key and fails there.

## The two privileges the connecting user needs

| Privilege                                                      | Without it                                                   |
| -------------------------------------------------------------- | ------------------------------------------------------------ |
| `%Native_ClassExecution:USE`                                   | The entry cannot be read at all.                             |
| `%Admin_Wallet:USE`, or READ on the collection's `UseResource` | The Wallet reference cannot be resolved, so there is no key. |

Granting READ on the collection's resource (`OpsReviewKeys` above) is the narrower choice: it
opens that collection only.

## Three ways to choose the entry

1. **In code:** `create_pipeline(..., llm_config="opsreview")`.
2. **In configuration:** `llm: {config_name: opsreview}` in your YAML, or the environment
   variable `RAG_LLM__CONFIG_NAME=opsreview`. `create_pipeline` reads it when `llm_config`
   is not given.
3. **For a pipeline you build yourself:** `configstore_llm` gives the answer function every
   pipeline constructor accepts as `llm_func`.

   ```python
   from iris_vector_rag.common.configstore_llm import configstore_llm
   from iris_vector_rag.pipelines.basic import BasicRAGPipeline

   llm = configstore_llm("opsreview", connection_manager)
   pipeline = BasicRAGPipeline(connection_manager, config_manager, llm_func=llm)
   ```

   The LangChain model is `llm.chat_model`.

An explicit `llm_func` wins over an entry, and an entry wins over the YAML `llm:` model
settings. With no entry named anywhere, nothing changes.

## Errors

Every way the entry can fail to give a model raises `LLMConfigError` when the pipeline is
built, never at the first question. The message names the entry and the fix:

- the name breaks ConfigStore's naming rule (IRIS is not asked);
- there is no such entry, or it is disabled;
- it has no `model` or no `model_provider`;
- the user cannot read the Wallet: the message gives the reference, never a value;
- the user lacks `%Native_ClassExecution:USE`;
- the IRIS has no ConfigStore: the message gives its version;
- the provider's LangChain package is not installed: the message names it
  (`pip install langchain-anthropic`, say). `langchain-openai` comes with this library;
  other providers are yours to install.
