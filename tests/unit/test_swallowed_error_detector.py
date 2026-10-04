"""The `swallowed-error` detector: a broad `except` that eats the error.

The repo shipped dozens of handlers that logged (or did not) and carried on with an
empty result: a failed insert that reported success, a model that failed to load
and became zero vectors. The detector is what makes "fail loudly" stick: an
instance absent from the baseline fails the gate, and the baseline only shrinks.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
GATE = ROOT / "scripts/gates/antipatterns.py"
FILE = "iris_vector_rag/storage/example.py"


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location("antipatterns_gate_swallow", GATE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _found(gate, source):
    return gate.swallowed_errors(FILE, source)


PASS_BODY = """
def load():
    try:
        work()
    except Exception:
        pass
"""

RETURN_DEFAULT = """
def load():
    try:
        return work()
    except Exception as exc:
        return []
"""

LOG_AND_CONTINUE = """
def load(items):
    for item in items:
        try:
            work(item)
        except Exception as exc:
            logger.warning("failed: %s", exc)
            continue
"""

BARE = """
def load():
    try:
        work()
    except:
        return None
"""

TUPLE_WITH_EXCEPTION = """
def load():
    try:
        work()
    except (ValueError, Exception):
        pass
"""


@pytest.mark.parametrize(
    "source",
    [PASS_BODY, RETURN_DEFAULT, LOG_AND_CONTINUE, BARE, TUPLE_WITH_EXCEPTION],
    ids=["pass", "return-default", "log-and-continue", "bare", "tuple"],
)
def test_a_broad_except_that_swallows_is_reported(gate, source):
    findings = _found(gate, source)
    assert len(findings) == 1, findings
    assert findings[0].check == "swallowed-error"
    assert findings[0].location.startswith(f"{FILE}::load")


RERAISE = """
def load():
    try:
        work()
    except Exception as exc:
        logger.error("x")
        raise
"""

RAISE_TYPED = """
def load():
    try:
        work()
    except Exception as exc:
        raise LoadError("could not load") from exc
"""

NARROW = """
def load():
    try:
        work()
    except (KeyError, ValueError):
        return None
"""

RAISE_IN_NESTED_FUNCTION_ONLY = """
def load():
    try:
        work()
    except Exception:
        def later():
            raise RuntimeError
        return later
"""


@pytest.mark.parametrize(
    "source", [RERAISE, RAISE_TYPED, NARROW], ids=["reraise", "typed-raise", "narrow"]
)
def test_handlers_that_raise_or_catch_narrowly_are_not_reported(gate, source):
    assert _found(gate, source) == []


def test_a_raise_inside_a_nested_function_does_not_count(gate):
    assert len(_found(gate, RAISE_IN_NESTED_FUNCTION_ONLY)) == 1


def test_the_location_survives_unrelated_edits_above(gate):
    """Keyed by function and ordinal, not line, so another branch adding an import
    does not turn every baseline entry stale."""
    shifted = "import os\nimport sys\n\n" + PASS_BODY
    assert [f.location for f in _found(gate, PASS_BODY)] == [
        f.location for f in _found(gate, shifted)
    ]


def test_two_swallows_in_one_function_are_two_findings(gate):
    source = PASS_BODY + """
def two():
    try:
        a()
    except Exception:
        pass
    try:
        b()
    except Exception:
        pass
"""
    locations = [f.location for f in _found(gate, source)]
    assert len(set(locations)) == 3


def test_methods_are_named_by_class_and_method(gate):
    source = """
class Store:
    def put(self):
        try:
            work()
        except Exception:
            pass
"""
    (finding,) = _found(gate, source)
    assert finding.location == f"{FILE}::Store.put#1"


def test_a_file_that_does_not_parse_is_an_error_not_a_pass(gate):
    with pytest.raises(SyntaxError):
        _found(gate, "def broken(:\n")


def test_the_check_is_registered_and_baselined(gate):
    assert "swallowed-error" in gate.CHECKS
    assert "swallowed-error" not in gate.NO_BASELINE


def test_the_tree_is_green_against_the_baseline_and_a_new_swallow_is_not(gate):
    """Today's instances are baselined; one more is a new finding."""
    baseline = gate.load_baseline()
    findings = gate.CHECKS["swallowed-error"]()
    assert findings, "the package has swallows today; a detector finding none is broken"
    assert {f.key() for f in findings} == {
        k for k in baseline if k.startswith("swallowed-error\t")
    }
    extra = gate.swallowed_errors("iris_vector_rag/new_module.py", PASS_BODY)
    assert extra[0].key() not in baseline


def test_only_the_library_is_scanned(gate):
    for finding in gate.CHECKS["swallowed-error"]():
        assert finding.location.startswith("iris_vector_rag/")
