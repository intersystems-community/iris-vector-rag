# Tasks: Answer Model from AI Hub ConfigStore

**Tests first in every phase.** Each user story ends with an integration gate.

- **Gate on `iris-vector-rag-iris` (2026.1):** the version-gate test.
- **Gate on `ivr-aihub-lab` (2026.3.0AI.126):** every other integration test. Registered, port 56972.

## Phase 1: Setup

- [X] T001 Add `tests/fixtures/aihub/docker-compose.yml` (the probe's lab, without `ISC_DATA_DIRECTORY`) and a
  session fixture `aihub_iris` in `tests/integration/conftest_aihub.py`:
  - start or reuse `ivr-aihub-lab`;
  - create Wallet collection `ivrprobe` with secret `openai` = `{"api_key": "sk-probe-123"}`;
  - create entries `AI.LLM.ivrtest` (key as a Wallet reference), `AI.LLM.ivrnokey` (no `model`) and
    `AI.LLM.ivroff` (disabled);
  - create a user without Wallet privileges, and one without `%Native_ClassExecution`;
  - delete all of it at session end.
- [X] T002 Register the `aihub` pytest marker. Deselect it in `.github/workflows/*.yml` test steps, with a comment
  (the internal image).

## Phase 2: Foundational

- [X] T003 [P] (FR-007) Unit test `tests/unit/test_llm_config_error.py`: `LLMConfigError` carries `entry` and
  `fix`, and its message never contains a secret value.
- [X] T004 (FR-007) Add `LLMConfigError(RAGException)` in `iris_vector_rag/exceptions.py`.

## Phase 3: US1 – use the configured model (P1)

- [X] T005 [P] [US1] (FR-002, FR-003, FR-004, FR-006) Unit tests in `tests/unit/test_configstore_llm.py`, with a
  fake `createIRIS` / `classMethodValue` / `invoke` object:
  - the entry JSON reaches `init_chat_model` unchanged, including unknown keys;
  - the result is a text-in, text-out function;
  - no environment variable is read;
  - only `Get` and `CopyDetails` are called (FR-009);
  - the read happens once, at build.
- [X] T006 [P] [US1] (FR-001, FR-005, FR-010) Unit tests in `tests/unit/test_create_pipeline_llm_config.py`:
  - precedence: `llm_func`, then `llm_config`, then `llm.config_name`, then the YAML `llm:` settings;
  - with nothing set, `get_llm_func` is called exactly as before.
- [X] T007 [US1] Integration test `tests/integration/test_configstore_llm_aihub.py` (`aihub` marker):
  - `create_pipeline("basic", llm_config="ivrtest")` builds ChatOpenAI with the resolved key `sk-probe-123`, the
    model and the temperature, with no network call;
  - `configstore_llm("ivrtest", connections)` passed as `llm_func` does the same;
  - the key never appears in captured logs.
- [X] T008 [US1] Implement `iris_vector_rag/common/configstore_llm.py`: `read_llm_entry` and `configstore_llm`.
- [X] T009 [US1] Wire `create_pipeline` in `iris_vector_rag/__init__.py`: one `_answer_model` helper replaces the
  two duplicated blocks, with `llm_config=` and `llm.config_name`.
- [X] T010 [US1] Gate: T005 to T007 pass.

## Phase 4: US2 – clear errors at build time (P1)

- [X] T011 [P] [US2] (FR-007, FR-008) Unit tests in `tests/unit/test_configstore_llm_errors.py`. Each of the 9
  failures raises `LLMConfigError` with the entry and the fix:
  - an invalid name, with no call made;
  - #5809, a missing entry;
  - #26220, the Wallet: the message names the reference from `CopyDetails(det, 0)` and the privilege;
  - `<PROTECT>`, the native-execution privilege;
  - a missing `model` or `model_provider`;
  - `<CLASS DOES NOT EXIST>`, the version, giving it;
  - a disabled entry;
  - an `ImportError` for the provider, naming the package;
  - a leftover `secret://` value.
- [X] T012 [US2] Integration tests (`aihub` marker) for a missing entry, no Wallet privilege, no native privilege,
  missing keys and a disabled entry. Plus `tests/integration/test_configstore_llm_version_gate.py` on
  `iris-vector-rag-iris` (2026.1), with no marker.
- [X] T013 [US2] Implement the error mapping in `configstore_llm.py`.
- [X] T014 [US2] Gate: T011 and T012 pass.

## Phase 5: Polish

- [X] T015 [P] (FR-011) `docs/LLM_FROM_CONFIGSTORE.md`:
  - how an administrator creates the entry and Wallet secret (ObjectScript);
  - the reference format, with `#field` required;
  - the two privileges;
  - the three ways to select it.

  Link it from the README.
- [X] T016 CHANGELOG Unreleased entry.
- [ ] T017 Full run:
  - the unit suite;
  - the integration suites, including `aihub` locally;
  - `scripts/gates/antipatterns.py`;
  - `.specify/gates/verify.sh --boundary agent`.

## Dependencies

Setup → Foundational → US1 → US2 → Polish. US2's errors build on US1's read path.
