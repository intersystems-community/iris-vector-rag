# Feature Specification: IRIS Session

**Feature Branch**: `083-iris-session`
**Created**: 2026-09-10
**Status**: Draft
**Input**: User description: "IRISSession: one session per IRIS instance and namespace owns connection lifetime, health cadence, driver selection and recovery; eleven acquisition paths become one."

## Background

A caller's relationship with an IRIS instance (a "session", see `CONTEXT.md`) is
today expressed through eleven different entry points, including two unrelated
classes that share a name. Protective behaviour added over the last releases (a
bounded connect, back-off after a hung handshake, recovery from an expired default
password, forcing TCP) landed on one of those paths only; another still connects
without any of it. Every storage call re-checks health with a network probe, and
one retrieval component receives no connection at all because it reads an attribute
no path provides. This feature gives the session one owner.

## User Scenarios & Testing _(mandatory)_

### User Story 1 - One way in, every protection everywhere (Priority: P1)

Whether a caller builds a pipeline from configuration, hands in a connection they
already hold, shares an engine between pipelines, or serves requests from the HTTP
API, they obtain the same kind of session, and every protective behaviour (bounded
connect, back-off, password recovery, transport selection) applies to all of them.

**Why this priority**: The 0.14.0 release cycle lost days to a handshake that hung
forever; the fix was applied to one acquisition path, and the others still hang. A
protection that only some callers get is not a protection.

**Independent Test**: Build a session through each supported entry point against an
unreachable instance and against an instance whose default password has expired;
observe the same bounded failure and the same recovery on all of them.

**Acceptance Scenarios**:

1. **Given** an IRIS instance that accepts TCP connections but never completes the
   driver handshake, **When** a caller acquires a session through any entry point,
   **Then** the attempt fails within the configured connect budget with an error
   naming the instance, and subsequent attempts fail immediately during the back-off
   window instead of hanging again.
2. **Given** an instance whose default account password is expired, **When** a
   caller acquires a session through any entry point, **Then** the session recovers
   the account and connects, and the recovery is reported once.
3. **Given** a caller who already holds a connection, **When** they pass it to a
   pipeline, **Then** the pipeline uses that connection and never closes it.
4. **Given** two pipelines constructed from the same configuration in one process,
   **When** both run, **Then** they share one session rather than opening two.
5. **Given** a session in use, **When** a caller asks it for the identity of the
   instance and namespace it is attached to, **Then** it answers without opening a
   connection.

---

### User Story 2 - Health is checked on a cadence, not on every call (Priority: P1)

A pipeline issuing many storage operations pays for at most one health probe per
interval, not one per operation, and a connection that has genuinely died is
replaced without the caller writing retry code.

**Why this priority**: Every storage call currently issues a probe query before its
real query, which doubles round trips on Community Edition where connections are
scarce; and the probe is not even used to recover, so the cost buys nothing.

**Independent Test**: Run N storage operations through one session and count the
probe queries; then kill the underlying connection server-side and confirm the next
operation succeeds after one transparent reconnect.

**Acceptance Scenarios**:

1. **Given** a healthy session, **When** 100 storage operations run inside one
   health interval, **Then** at most one health probe is issued.
2. **Given** a session whose underlying connection has been dropped by the server,
   **When** the next operation runs, **Then** the session reconnects once and the
   operation succeeds, without the caller catching an error.
3. **Given** a session whose instance is unreachable, **When** the next operation
   runs, **Then** it fails with an error naming the instance, and no more than one
   reconnect is attempted per operation.
4. **Given** a caller that wants to know the state without paying for a probe,
   **When** they ask the session whether it is currently connected, **Then** the
   answer comes from recorded state.

---

### User Story 3 - Statements are safe to run and safe to read (Priority: P2)

A caller runs a statement and gets rows they can use after the statement is over,
without managing a cursor, without leaking one on failure, and without holding a
reference to a row that the driver has invalidated.

**Why this priority**: Rows returned by this driver stop being readable once their
cursor closes; this has caused silent empty results and truncated fields. Making
the safe form the only convenient form removes the whole class.

**Independent Test**: Run a statement that returns rows, close everything, and read
the returned values; run a statement that raises mid-iteration and confirm no
cursor remains open.

**Acceptance Scenarios**:

1. **Given** a statement that returns rows, **When** a caller runs it through the
   session, **Then** the values remain readable after the underlying cursor is
   closed.
2. **Given** a statement that fails during execution, **When** the failure
   propagates, **Then** no cursor and no transaction is left open by the session.
3. **Given** a statement expected to return a single row, **When** it returns none,
   **Then** the caller receives an explicit empty answer rather than an error about
   row objects.
4. **Given** a caller in a thread that has not used the session before, **When**
   they run a statement, **Then** they get a connection that is not shared with any
   other thread.

---

### User Story 4 - Driver choice is a decision, not an import accident (Priority: P2)

An operator running inside IRIS gets the embedded driver; an operator running
outside gets the network driver; both can see which was chosen and why, and a test
can supply neither.

**Why this priority**: Driver selection is currently spread across import-time
try/except blocks in several modules, so the same process can end up using
different drivers depending on which module loaded first, and no log says which.

**Independent Test**: Construct a session in an environment where both drivers are
importable, one, and neither; inspect the reported choice in each case.

**Acceptance Scenarios**:

1. **Given** an environment where the embedded driver is available, **When** a
   session is created, **Then** the embedded driver is used and the choice is
   reported.
2. **Given** an environment with only the network driver, **When** a session is
   created, **Then** the network driver is used with the transport the environment
   requires.
3. **Given** an environment with no usable driver, **When** a session is created,
   **Then** it fails with one error naming what was tried, not an import error from
   an unrelated module.
4. **Given** a test that supplies its own driver, **When** it creates a session,
   **Then** no real driver is imported and no network call occurs.

---

### User Story 5 - The old names keep working for one release (Priority: P3)

An existing user upgrading to 0.15 finds their current code working unchanged, with
a warning that names the replacement; by 0.16 the old names are gone.

**Why this priority**: The removed names are public and appear in downstream code
and in this project's own examples; a silent break would be worse than the current
duplication.

**Independent Test**: Run the previous release's usage examples against the new
release; confirm identical results plus one deprecation warning each.

**Acceptance Scenarios**:

1. **Given** code using a superseded connection entry point, **When** it runs on
   0.15, **Then** it behaves as before and emits one deprecation warning naming the
   session as the replacement.
2. **Given** the HTTP API, **When** it serves concurrent requests on 0.15, **Then**
   it behaves as before while obtaining its connections from sessions.
3. **Given** an adapter that needs the underlying connection object, **When** it
   asks the session for it, **Then** it receives it through a single documented
   accessor marked for adapter use only.

---

### Edge Cases

- Two threads acquire the same session simultaneously on first use: exactly one
  connection is opened per thread, and no thread observes a half-built session.
- A caller-supplied connection is already broken: the session reports the failure on
  first use and does not attempt to replace a connection it does not own.
- The instance restarts between two operations: the next operation reconnects; a
  transaction that was open at the time is reported as lost rather than silently
  retried.
- Connect back-off is in effect and the instance becomes healthy inside the window:
  the caller waits at most the back-off window, not longer.
- A process forks after a session exists: the child does not inherit and reuse the
  parent's connection.
- Interpreter shutdown with sessions open: connections close without raising from a
  finaliser.
- Configuration differing only in namespace: two sessions, not one.

## Requirements _(mandatory)_

### Functional Requirements

- **FR-001**: The library MUST expose exactly one way to obtain a session, and every
  internal component MUST obtain its connectivity through it.
- **FR-002**: A session MUST be identified by the instance and namespace it attaches
  to; two requests with the same identity in one process MUST yield the same
  session.
- **FR-003**: Connections MUST NOT be shared between threads; each thread using a
  session MUST get its own connection from that session.
- **FR-004**: The connect attempt MUST be bounded by a configurable budget, and a
  failed or timed-out handshake MUST put that session identity into a back-off
  window during which further attempts fail immediately.
- **FR-005**: Recovery from an expired default account password MUST be attempted
  once per session identity and reported when it happens.
- **FR-006**: The session MUST select its driver explicitly, preferring the embedded
  driver when it is usable, and MUST report which driver it selected.
- **FR-007**: Driver selection MUST be substitutable so that tests can run without
  any real driver and without any network call.
- **FR-008**: A session MUST verify health at most once per configured interval, and
  MUST NOT probe as part of ordinary operations inside that interval.
- **FR-009**: A session that owns its connection MUST replace it transparently after
  a connection-level failure, at most once per operation.
- **FR-010**: A session MUST NOT close or replace a connection supplied by the
  caller; it MUST report such failures instead.
- **FR-011**: The session MUST offer statement execution that returns values which
  remain valid after the underlying cursor closes.
- **FR-012**: Every statement path MUST release its cursor on both success and
  failure.
- **FR-013**: Access to the underlying connection object MUST be through a single
  accessor documented as being for adapters only, and no library component outside
  adapters may use it.
- **FR-014**: The retrieval component that currently receives no connection MUST
  receive its session, and the engine attribute that reports a fixed embedding
  dimension MUST report the real one.
- **FR-015**: The superseded entry points MUST keep their current signatures and
  behaviour in 0.15, emitting one deprecation warning each that names the session as
  the replacement; they are removed in 0.16.
- **FR-016**: Entry points that exist only as re-exports with no callers MUST be
  removed in this feature rather than deprecated.
- **FR-017**: The HTTP API MUST continue to serve concurrent requests with its
  current behaviour while obtaining connections from sessions.
- **FR-018**: All connection acquisition MUST go through the bounded, backed-off
  path; a direct driver connect anywhere else in the library MUST fail the quality
  gates.
- **FR-019**: Failure modes (unreachable instance, expired password, no usable
  driver, dead connection, caller-supplied connection broken) MUST be distinct,
  descriptive errors naming the instance and namespace.
- **FR-020**: An engine MUST own exactly one session, and pipelines sharing an
  engine MUST share that session.

### Key Entities

- **Session**: a caller's relationship with one instance and namespace: its identity,
  its driver choice, its connection lifetime, its health cadence, its back-off
  state.
- **Session identity**: the instance, port, namespace and account that determine
  whether two requests are for the same session.
- **Driver**: the substitutable means of reaching IRIS (embedded or network), with a
  fixed set of capabilities the session relies on: availability, connect, ping,
  close, password state, password recovery.
- **Connection**: one thread's live handle inside a session; owned by the session or
  supplied by the caller.
- **Statement result**: rows copied out of the driver's cursor so they outlive it.

## Assumptions

- One session per identity per process is sufficient; Community Edition's connection
  scarcity is addressed by not opening more, not by queuing.
- The existing configurable connect budget and back-off window keep their current
  defaults and environment variable names.
- The health interval default is on the order of tens of seconds; an exact value is
  a tuning decision recorded in the plan, not a requirement.
- Transaction management stays with callers; the session guarantees cursor cleanup,
  not transaction semantics.
- The pooling behaviour the HTTP API relies on is a property of that adapter, not of
  the session; no general-purpose pool is introduced.
- Table shape resolution (feature 082) is the first consumer of the session and lands
  first; this feature does not restate its requirements.

## Success Criteria _(mandatory)_

### Measurable Outcomes

- **SC-001**: One entry point provides connectivity; the number of public
  connection-acquisition paths goes from eleven to one plus deprecated adapters, and
  no two classes in the library share a name.
- **SC-002**: A run of 100 storage operations issues at most one health probe, down
  from at least one per operation.
- **SC-003**: Every acquisition path fails within the connect budget against a
  non-completing handshake, and enters back-off; verified by one test per path.
- **SC-004**: Zero direct driver connects exist outside the session, enforced by an
  automated check in the quality gates.
- **SC-005**: The reported embedding dimension matches the configured model on every
  engine, and the retrieval component reports a live session rather than none.
- **SC-006**: Statement results are readable after their cursor closes in every
  library call site; no call site holds a driver row object.
- **SC-007**: Existing contract, unit and end-to-end suites pass with no behaviour
  change; the only new failures permitted are deprecation warnings.
- **SC-008**: Storage unit tests construct a session with a substituted driver and
  make no network call; test setup for connectivity is one line.
- **SC-009**: Release 0.15 breaks no documented public call; release 0.16 removes the
  deprecated names with the removal listed in the changelog.
