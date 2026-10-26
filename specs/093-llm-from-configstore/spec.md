# Feature Specification: Answer Model from AI Hub ConfigStore

**Feature Branch**: `093-llm-from-configstore`
**Created**: 2026-10-26
**Status**: Draft
**Input**: Decided with Tom on 2026-10-26: the backlog item
`ivr-research-notes/.scratch/rakathon-later/issues/09-llm-from-configstore.md`. The facts come
from a probe on IRIS for Health Community 2026.3.0AI build 126
(`.scratch/rakathon-later/research/configstore-probe.md`).

## Background

In an IRIS shop using AI Hub, an administrator defines a named model entry once, inside IRIS:
`AI.LLM.<name>` in `%ConfigStore.Configuration`. The entry holds the model, the provider and
settings such as temperature. Its API key is a reference into the IRIS Wallet
(`secret://<Collection>.<Secret>#<field>`), so the key never leaves IRIS's access control. Apps
refer to the entry by name. ObjectScript AI Hub code and `langchain-intersystems` (internal, not
on PyPI) both read these entries.

Today iris-vector-rag takes its answer model from a Python function or from its own YAML
`llm:` settings, with the key in an environment variable. An IRIS shop therefore configures
models twice, and copies keys out of the Wallet into the environment to do it.

The probe found:

- Reading an entry from Python with `Get` and `CopyDetails` returns the secret already
  resolved.
- A caller without Wallet privileges gets an error status, not an exception and not a blank.
- `GetDetails` refuses every `AI.LLM` entry on this build.
- The newest `langchain-intersystems` imports a driver class that no released driver has.

## Clarifications

### Session 2026-10-26 (with Tom)

- **Q:** What does ivr take from ConfigStore? → **A:** The answer model only, from
  `AI.LLM.<name>`. No embedder, since AI Hub has no embedding category, and no MCP servers, since
  ivr serves MCP rather than consuming it.
- **Q:** How is the entry turned into a model? → **A:** ivr reads the entry over its own IRIS
  connection and hands it to LangChain's `init_chat_model`. That matches `langchain-intersystems`
  without depending on it.
- **Q:** What if the key cannot be resolved? → **A:** The read resolves Wallet references itself
  (probe). When it fails, ivr raises an error naming the entry, the reference and the privilege
  needed. It never falls back to an environment variable.
- **Q:** How is it named, and what takes precedence? → **A:** `llm.config_name` in configuration
  or `llm_config=` in code. The entry is read from the same IRIS that holds the data. The order is
  an explicit `llm_func`, then `llm_config`, then the YAML `llm:` model settings. Errors are raised
  when the pipeline is built.

## User Scenarios & Testing _(mandatory)_

### User Story 1 - Use the model the IRIS administrator configured (Priority: P1)

An IRIS administrator has created `AI.LLM.opsreview`, with its key in the Wallet. A developer
builds a pipeline with `create_pipeline(..., llm_config="opsreview")`, or sets
`llm.config_name: opsreview`, or passes `llm_func=configstore_llm("opsreview", connections)` to a
pipeline it builds itself, and asks a question. The answer comes from that model, and the developer never handles the key.

**Why this priority**: It is the feature: one place to configure models, and keys that stay in
IRIS.

**Independent Test**: On a 2026.3 AI build, create an entry whose key is a Wallet reference to a
fake key. Build a pipeline from it. The model object carries the resolved key, provider, model
and temperature from the entry, and no network call is made.

**Acceptance Scenarios**:

1. **Given** entry `AI.LLM.opsreview` = `{model, model_provider: openai, temperature,
   api_key: secret://…}`, **When** `create_pipeline("basic", llm_config="opsreview")` runs,
   **Then** the pipeline's answer model is an OpenAI chat model with that model name,
   temperature and the resolved key.
2. **Given** `RAG_LLM__CONFIG_NAME=opsreview` and no other LLM setting, **When** a pipeline is
   built, **Then** it uses the same entry.
3. **Given** an explicit `llm_func` and `llm_config`, **When** a pipeline is built, **Then**
   `llm_func` is used and the entry is not read.
4. **Given** `llm_config` and the YAML `llm.model_name`, **When** a pipeline is built, **Then**
   the entry wins.
5. **Given** no `llm_config` anywhere, **When** a pipeline is built, **Then** behaviour is exactly
   as before this feature.

---

### User Story 2 - A clear error instead of a bad model (Priority: P1)

Every way the entry can fail to give a usable model is reported when the pipeline is built, in
words that say what to fix. The user never sees a failure later at the first question, or an
authentication error that never mentions the Wallet.

**Why this priority**: The probe showed the server's own messages don't name the missing
privilege or the entry. Without this, the feature fails in ways that are hard to diagnose.

**Independent Test**: For each failure, build a pipeline and check the error type and text.

**Acceptance Scenarios**:

1. **Given** no entry `AI.LLM.nosuch`, **When** a pipeline is built with it, **Then** the error
   says there is no ConfigStore entry `AI.LLM.nosuch`.
2. **Given** a user without Wallet privileges, **When** the pipeline is built, **Then** the error
   names the entry, the reference (`secret://<Collection>.<Secret>#<field>`) and the privilege
   needed (`%Admin_Wallet:USE`, or READ on the collection's resource).
3. **Given** a user without `%Native_ClassExecution:USE`, **When** the pipeline is built,
   **Then** the error names that privilege.
4. **Given** a name with a dot or another character ConfigStore refuses, **When** a pipeline is
   built, **Then** the error states ConfigStore's naming rule, and IRIS is not queried.
5. **Given** an entry without `model` or `model_provider`, **When** a pipeline is built, **Then**
   the error names the missing key.
6. **Given** an IRIS without `%ConfigStore.Configuration` (older than 2026.3), **When** a
   pipeline is built, **Then** the error says the feature needs IRIS 2026.3 or later (AI Hub) and
   gives the server's version.
7. **Given** an entry that is disabled, **When** a pipeline is built, **Then** the error says it
   is disabled.
8. **Given** an entry whose provider package is not installed (e.g. `model_provider: anthropic`
   without `langchain-anthropic`), **When** a pipeline is built, **Then** the error names the
   package to install.
9. **Given** any value that still reads `secret://` after the read, **When** a pipeline is built,
   **Then** it is treated as an unresolved reference (scenario 2's error). An unresolved value is
   never handed to the model.

### Edge Cases

- **Unknown keys** in the entry (e.g. `base_url`) are passed to LangChain unchanged, as
  `langchain-intersystems` does.
- **Names are case-sensitive:** `OpsReview` and `opsreview` are different entries.
- **A reference without `#field`** resolves to the whole secret as JSON text. That reaches the
  provider as a key and fails there. The docs say to always include `#field`.
- **An entry changed after the pipeline is built** is not re-read; build a new pipeline.
- **No IRIS connection** at build time (a pipeline built without one) with `llm_config` set:
  error.

## Requirements _(mandatory)_

### Functional Requirements

- **FR-001**: The answer model MUST be selectable by the name of an AI Hub ConfigStore LLM entry:
  `create_pipeline(..., llm_config=...)`, or `llm.config_name` in configuration (environment
  `RAG_LLM__CONFIG_NAME`), which `create_pipeline` reads. For a pipeline built directly, a public
  helper `configstore_llm(name, connection_manager)` MUST return the answer function, to pass as
  the `llm_func` every pipeline constructor already accepts.
- **FR-002**: The entry `AI.LLM.<name>` MUST be read over the IRIS connection the pipeline
  already uses for its data, with the released `intersystems-irispython` driver.
- **FR-003**: Wallet references in the entry MUST be resolved by the read itself. No API key may
  be taken from an environment variable when `llm_config` is set.
- **FR-004**: The entry's `model` and `model_provider`, and every other key, MUST be passed to
  LangChain's `init_chat_model` unchanged, and the result wrapped as the pipeline's answer
  function.
- **FR-005**: Precedence MUST be: an explicit `llm_func`, then `llm_config` / `llm.config_name`,
  then the existing YAML `llm:` settings.
- **FR-006**: The entry MUST be read and the model built when the pipeline is built, not at the
  first question.
- **FR-007**: Each failure in User Story 2 MUST raise one error type (`LLMConfigError`) whose
  message names the entry and the specific fix: missing entry, missing privilege (Wallet or
  native class execution), an invalid name, a missing required key, no ConfigStore on the server
  (with the server version), a disabled entry, a missing provider package, or an unresolved
  reference.
- **FR-008**: A name MUST match ConfigStore's rule `^[A-Za-z][A-Za-z0-9_-]*$` before IRIS is
  queried.
- **FR-009**: The read MUST NOT use a validity check that refuses `AI.LLM` entries: on 2026.3
  every such entry reports itself not valid.
- **FR-010**: With no `llm_config` set anywhere, every existing behaviour MUST be unchanged.
- **FR-011**: The docs MUST show how an administrator creates an entry and a Wallet secret. They
  MUST give the reference format, list the two privileges the connecting user needs, and include
  a pipeline example.

### Key Entities

- **LLM entry**: a named configuration in IRIS (`AI.LLM.<name>`): model, provider, settings and
  a key that is usually a Wallet reference.
- **Wallet reference**: `secret://<Collection>.<Secret>#<field>`, which IRIS resolves for a
  caller with the privilege to read that secret.

## Success Criteria _(mandatory)_

### Measurable Outcomes

- **SC-001**: On a 2026.3 AI build, a pipeline built from an entry whose key is in the Wallet
  answers with the configured model, and no key appears in ivr's configuration, environment or
  logs.
- **SC-002**: Each of the 9 failure scenarios produces an error that names the entry and the fix,
  in 100 % of runs, at build time.
- **SC-003**: The existing unit and integration suites pass unchanged.

## Assumptions

- The integration test needs a 2026.3 AI build. It runs on the registered throwaway lab
  `ivr-aihub-lab`, which the probe used. The project container (2026.1) has no ConfigStore, so
  there the tests cover the version-gate error.
- LangChain's `init_chat_model` builds a model object without a network call (probe:
  `langchain` 1.4.7).
- Provider packages other than `langchain-openai` are the user's to install.
- A disabled entry is refused rather than used: an administrator who disabled it meant "don't
  use".
- The embedder and MCP servers are out of scope (decision Q1).
