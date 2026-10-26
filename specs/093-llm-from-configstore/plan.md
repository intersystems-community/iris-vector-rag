# Implementation Plan: Answer Model from AI Hub ConfigStore

**Branch**: `093-llm-from-configstore` | **Date**: 2026-10-26 | **Spec**: [spec.md](spec.md)

## Summary

`create_pipeline(..., llm_config="opsreview")`, `llm.config_name`, or a pipeline built with
`llm_func=configstore_llm("opsreview", connections)` takes the answer model from AI Hub's
`AI.LLM.opsreview` entry. The read happens once, at build time:

1. Over ivr's own IRIS connection, `iris.createIRIS(conn)` and
   `%ConfigStore.Configuration.Get("AI.LLM.<name>")`.
2. `CopyDetails`, which resolves Wallet references.
3. Checks on the result.
4. LangChain's `init_chat_model(**entry)`, wrapped as ivr's text-in, text-out `llm_func`.

Every failure is one `LLMConfigError` that names the entry and the fix.

## Technical Context

**Language/Version**: Python 3.11
**Primary Dependencies**:
- `intersystems-irispython` (released driver; `iris.createIRIS` and `classMethodValue`, not
  the unreleased `IRISNative`);
- `langchain` (`langchain.chat_models.init_chat_model`), already a dependency.

**Storage**: none new. It reads `%ConfigStore.Configuration` on IRIS 2026.3 AI builds.
**Testing**:
- **Unit:** pytest against a fake Native object covering every failure path.
- **Integration (default):** `iris-vector-rag-iris` (2026.1) for the version-gate error.
- **Integration (AI Hub):** `ivr-aihub-lab` (2026.3.0AI.126, registered throwaway), for the
  real read, the Wallet and the privileges.

**Constraints**:
- No network call at build time.
- No key taken from the environment when `llm_config` is set.
- No change when it isn't.

## Constitution Check

| Principle                  | How                                                                                                                                                                                                                                                                                                                                     |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 IRIS-first integration   | Real reads on a 2026.3 AI build. The fixture starts the registered `ivr-aihub-lab` if it isn't running and removes it at session end. GitHub CI cannot pull internal AI images, so those tests carry the `aihub` marker and are deselected there, with the reason documented. Locally they run by default. The 2026.1 version gate runs everywhere. |
| 6 Secrets hygiene          | Keys stay in the Wallet. Errors print the `secret://` reference, never a value. Tests use the fake key `sk-probe-123` and assert it is never logged.                                                                                                                                                                                       |
| 7 Backend mode             | No new connection: the pipeline's existing one is used.                                                                                                                                                                                                                                                                                  |

Gate: **pass**, with one documented deselection in GitHub CI (internal image).

## Design

1. **`iris_vector_rag/common/configstore_llm.py`** (new, about 120 lines):
   - `read_llm_entry(name, connection) -> dict` checks the name (`^[A-Za-z][A-Za-z0-9_-]*$`)
     before any call, then:
     - `createIRIS(conn)`;
     - `classMethodValue("%ConfigStore.Configuration", "Get", "AI.LLM."+name, ref)`;
     - checks `sc != 1`, mapping #5809 to "no entry";
     - `CopyDetails(det)` (resolves secrets);
     - checks `sc`, mapping #26220 to the Wallet error, fetching the reference with
       `CopyDetails(det, 0)`;
     - `%ToJSON`, `json.loads`;
     - required keys, `Enabled`, and any remaining `secret://`.
   - `<CLASS DOES NOT EXIST>` maps to the version error, with `%SYSTEM.Version.GetNumber()`.
     `<PROTECT> … %Native_ClassExecution` maps to the native-privilege error.
   - `configstore_llm(name, connection_manager) -> Callable[[str], str]`:
     `init_chat_model(**entry)` (an `ImportError` names the provider package), wrapped as
     `lambda prompt: chat.invoke(prompt).content`.
2. **`LLMConfigError`** (new, in `exceptions.py`, a subclass of `RAGException`), with fields
   `entry` and `fix`.
3. **`create_pipeline`**: both branches resolve
   `effective_llm_func = llm_func or configstore_llm(name) if name else get_llm_func(...)`.
   `name` is `kwargs.pop("llm_config")` or `config_manager.get("llm.config_name")`. One helper
   (`_answer_model(...)`) replaces the two duplicated blocks.
4. **Docs:** a section in `docs/` on making an entry and a Wallet secret, the reference
   format, the two privileges, and the three ways to select it. Plus a CHANGELOG entry.
5. **Test lab:** `tests/fixtures/aihub/docker-compose.yml` (the probe's, without
   `ISC_DATA_DIRECTORY`) and a session fixture `aihub_iris`, which:
   - starts or reuses `ivr-aihub-lab` (ports 56972 and 56773);
   - creates a Wallet secret and the entries;
   - creates a user without Wallet privileges;
   - tears it all down.

## Complexity Tracking

| Deviation                                    | Why                                                                                                   |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| AI Hub integration tests deselected in GitHub CI | GitHub runners cannot pull `docker.iscinternal.com` images. They run locally by default, and the version gate runs everywhere. |
