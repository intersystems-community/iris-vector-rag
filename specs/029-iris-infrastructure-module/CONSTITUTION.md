# IRIS DevTools Constitution

**Version**: 1.0.0
**Status**: Foundational
**Last Updated**: 2025-10-05

## Preamble

This constitution codifies the hard-won lessons, blind alleys avoided, and battle-tested practices from years of InterSystems IRIS development. Every principle herein represents real production experience, real failures overcome, and real solutions that work.

## Core Principles

### 1. AUTOMATIC REMEDIATION OVER MANUAL INTERVENTION

**The Principle**: Infrastructure problems must be automatically detected and remediated without developer intervention.

**Why It Matters**:
- Password expiration errors have wasted hundreds of developer hours
- Manual remediation breaks CI/CD pipelines
- "Works on my machine" scenarios damage team productivity

**Implementation Requirements**:
- ✅ Password change required → automatic reset
- ✅ Container not found → automatic start
- ✅ Port conflicts → automatic port reassignment
- ✅ Stale schema → automatic reset
- ✅ Connection failures → automatic retry with backoff

**Forbidden**:
- ❌ Error messages without remediation steps
- ❌ Requiring manual Docker commands
- ❌ Silent failures without auto-recovery attempts

**Example**:
```python
# WRONG: Manual intervention required
raise ConnectionError("Password change required. Run: docker exec...")

# RIGHT: Automatic remediation
if "Password change required" in str(error):
    reset_password_automatically()
    retry_connection()
```

### 2. DBAPI FIRST, JDBC FALLBACK

**The Principle**: Always prefer `intersystems-irispython` (DBAPI) over JDBC connections.

**Why It Matters**:
- DBAPI is 3x faster than JDBC for typical workloads
- DBAPI has better Python integration (no JVM overhead)
- JDBC requires external jar files (deployment complexity)
- DBAPI works in restrictive environments (no Java required)

**Implementation Requirements**:
- ✅ Attempt DBAPI connection first
- ✅ Fall back to JDBC only if DBAPI unavailable
- ✅ Log connection type used
- ✅ Support both connection types in parallel

**Forbidden**:
- ❌ JDBC-only implementations
- ❌ Silently falling back without logging
- ❌ Requiring JDBC when DBAPI available

**Evidence**:
```
Benchmark Results (1000 simple queries):
- DBAPI: 2.3 seconds
- JDBC:  7.1 seconds
- Speedup: 3.09x
```

### 3. ISOLATION BY DEFAULT

**The Principle**: Every test suite gets its own isolated IRIS instance unless explicitly shared.

**Why It Matters**:
- Shared databases cause test pollution
- Parallel test execution requires isolation
- Cleanup failures cascade to other tests
- "Works alone but fails in suite" mysteries

**Implementation Requirements**:
- ✅ Testcontainers for test isolation
- ✅ Unique namespaces per test class
- ✅ test_run_id tracking for data cleanup
- ✅ Automatic cleanup even on test failure

**Forbidden**:
- ❌ Shared databases without cleanup
- ❌ Assuming tests run sequentially
- ❌ Leaving test data behind

**Scope Guidelines**:
```python
# Module scope: Fast, shared state acceptable
@pytest.fixture(scope="module")
def iris_db_fast():
    # One container for all tests in module
    # Use when: Tests are read-only or properly isolated

# Function scope: Slower, maximum isolation
@pytest.fixture(scope="function")
def iris_db_isolated():
    # New container for each test
    # Use when: Tests modify schema or require clean state
```

### 4. ZERO CONFIGURATION VIABLE

**The Principle**: `pip install iris-devtools && pytest` must work without configuration.

**Why It Matters**:
- Reduces onboarding friction
- Enables quick experimentation
- Makes examples self-contained
- CI/CD "just works"

**Implementation Requirements**:
- ✅ Sensible defaults for all configuration
- ✅ Auto-discovery of IRIS instances
- ✅ Community edition defaults
- ✅ Environment variable overrides
- ✅ Explicit configuration always possible

**Forbidden**:
- ❌ Required configuration files
- ❌ Mandatory environment variables
- ❌ Undocumented prerequisites

**Configuration Hierarchy** (highest priority first):
1. Explicit constructor arguments
2. Environment variables
3. .env files in project root
4. Docker container inspection
5. Sensible defaults (localhost:1972, etc.)

### 5. FAIL FAST WITH GUIDANCE

**The Principle**: Errors must be detected immediately with clear remediation steps.

**Why It Matters**:
- Debugging time is expensive
- Stack traces without context are useless
- Developers need actionable guidance

**Implementation Requirements**:
- ✅ Detect errors at initialization, not first use
- ✅ Include "What went wrong" explanation
- ✅ Include "How to fix it" remediation
- ✅ Include "Why it matters" context (when helpful)
- ✅ Link to relevant documentation

**Forbidden**:
- ❌ Generic error messages
- ❌ Stack traces without explanation
- ❌ "Contact administrator" without details

**Example**:
```python
# WRONG
raise ConnectionError("Failed to connect")

# RIGHT
raise ConnectionError(
    "Failed to connect to IRIS at localhost:1972\n"
    "\n"
    "What went wrong:\n"
    "  The IRIS database is not running or not accessible.\n"
    "\n"
    "How to fix it:\n"
    "  1. Start IRIS: docker-compose up -d\n"
    "  2. Wait 30 seconds for startup\n"
    "  3. Verify: docker logs iris_db_rag_templates\n"
    "\n"
    "Alternative: Use testcontainers for automatic IRIS management:\n"
    "  from iris_devtools.containers import IRISContainer\n"
    "  with IRISContainer.community() as iris:\n"
    "      conn = iris.get_connection()\n"
    "\n"
    "Documentation: https://iris-devtools.readthedocs.io/troubleshooting/\n"
)
```

### 6. ENTERPRISE READY, COMMUNITY FRIENDLY

**The Principle**: Support both Community and Enterprise editions equally well.

**Why It Matters**:
- Different projects have different needs
- Development often uses Community, production uses Enterprise
- License management is complex
- Mirror configurations are enterprise-only

**Implementation Requirements**:
- ✅ Community edition as default
- ✅ Enterprise edition via license_key parameter
- ✅ Auto-discovery of license files
- ✅ Support for all Enterprise features (Mirrors, Sharding, etc.)
- ✅ Clear documentation of edition differences

**Forbidden**:
- ❌ Hardcoded edition assumptions
- ❌ Enterprise-only code paths without community fallback
- ❌ Obscure license errors

**License Discovery Order**:
1. Explicit `license_key` parameter
2. `IRIS_LICENSE_KEY` environment variable
3. `~/.iris/iris.key`
4. `./iris.key` in project root
5. Auto-discovered from Docker volume mounts

### 7. MEDICAL-GRADE RELIABILITY

**The Principle**: All code must be battle-tested in production scenarios with comprehensive error handling.

**Why It Matters**:
- Healthcare applications require 99.9%+ uptime
- Silent failures are unacceptable
- Diagnostic data saves hours of debugging
- Idempotency prevents cascading failures

**Implementation Requirements**:
- ✅ 95%+ test coverage
- ✅ All error paths tested
- ✅ Idempotent operations (safe to retry)
- ✅ Comprehensive logging
- ✅ Performance monitoring
- ✅ Graceful degradation
- ✅ Health check endpoints

**Forbidden**:
- ❌ Untested error paths
- ❌ Non-idempotent operations
- ❌ Silent failures
- ❌ Assumptions about state

**Reliability Checklist**:
```python
# Every operation must answer:
- [ ] What happens if this fails?
- [ ] Can it be retried safely?
- [ ] How do we detect failure?
- [ ] What diagnostics are logged?
- [ ] How do we recover?
- [ ] What's the user impact?
```

### 8. DOCUMENT THE BLIND ALLEYS

**The Principle**: Failed approaches must be documented to prevent repetition.

**Why It Matters**:
- Developers waste time rediscovering "why not"
- Institutional knowledge must be preserved
- Context for design decisions is valuable
- Prevents regression to worse solutions

**Implementation Requirements**:
- ✅ `docs/learnings/` directory for deep-dives
- ✅ "Why not X?" sections in documentation
- ✅ ADR (Architecture Decision Records) for major choices
- ✅ Performance comparisons
- ✅ Case studies from production

**Documented Blind Alleys**:
- **Why not JDBC-only?** → DBAPI is 3x faster, see benchmark
- **Why not shared test databases?** → Data pollution, see case study
- **Why not manual password resets?** → CI/CD breaks, see incident report
- **Why not port 1972 hardcoded?** → Conflicts in parallel tests, see issue #42

**Example Documentation**:
```markdown
## Why Not Use Docker Compose for Tests?

**What we tried**: Using docker-compose.yml for test database
**Why it didn't work**:
- Parallel tests conflicted on ports
- Cleanup required manual intervention
- CI/CD required docker-compose installation
- Container lifecycle not tied to test lifecycle

**What we use instead**: Testcontainers
**Evidence**: 287 test failures → 0 after migration
**Date tried**: 2024-09-15
**Decision**: Codified in constitution v1.0
```

## Governance

### Amendment Process

Principles may be amended when:
1. New evidence contradicts existing principle
2. Technology landscape changes materially
3. Production experience reveals gap
4. Community consensus emerges

**Amendment requires**:
- Concrete evidence (benchmarks, case studies, incident reports)
- Backwards compatibility analysis
- Migration guide for existing code
- Updated documentation
- Version bump (major if breaking)

### Enforcement

**Pre-commit hooks** validate:
- [ ] No hardcoded passwords or credentials
- [ ] All database operations are idempotent
- [ ] Error messages include remediation
- [ ] Test isolation via testcontainers or unique namespaces

**CI/CD validates**:
- [ ] 95%+ test coverage
- [ ] All platforms (Linux, Mac, Windows)
- [ ] Both Community and Enterprise editions
- [ ] Performance benchmarks (no regressions)

**Code review checklist**:
- [ ] Follows DBAPI-first principle
- [ ] Automatic remediation implemented
- [ ] Comprehensive error handling
- [ ] Documentation updated
- [ ] Blind alleys documented if applicable

## Version History

### v1.0.0 (2025-10-05)
- Initial constitution
- 8 core principles established
- Based on rag-templates production experience
- Incorporates learnings from Features 026, 028

## References

- [Testcontainers Best Practices](https://testcontainers.com/guides/)
- [InterSystems IRIS Docker Guide](https://docs.intersystems.com/irislatest/csp/docbook/DocBook.UI.Page.cls?KEY=ADOCK)
- [rag-templates Feature 026](../026-fix-critical-issues/)
- [rag-templates Feature 028](../028-obviously-these-failures/)
- [12 Factor App Methodology](https://12factor.net/)
- [Medical Device Software Standards](https://www.fda.gov/medical-devices/digital-health-center-excellence/software-medical-device-samd)

---

**Remember**: Every principle here was paid for with real debugging time, real production incidents, real developer frustration. Honor these learnings by building on them, not repeating them.
