"""Unit tests for scripts/dev/tcp_setup.py: unexpire passwords over TCP with no `iris session`.

Offline: a fake Native API handle keeps a model of the helper class loading and
password unexpiry calls. Contract test reads the class source to catch method mismatches.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import sys
from contextlib import contextmanager

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/dev/tcp_setup.py"
CLASS = ROOT / "scripts/dev/objectscript/IVRDev/Setup.cls"


@pytest.fixture(scope="module")
def ts():
    spec = importlib.util.spec_from_file_location("tcp_setup_ivr", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeIRIS:
    def __init__(self, loaded_hash: str | None = None):
        self.loaded_hash = loaded_hash
        self.calls = []
        self.loads = []
        self.locks = []
        self.fail = {}

    def classMethodValue(self, cls, method, *args):
        self.calls.append((method, *args))
        if method == "SourceHash":
            return self.loaded_hash
        if method == "UnExpireUserPasswords":
            if "UnExpireUserPasswords" in self.fail:
                return None  # will trigger FAIL path
            return 1  # success
        if method == "LoadStream":
            stream, flags = args
            # Track which class was loaded (the class name from the stream content)
            # For simplicity, we know it's IVRDev.Setup
            self.loads.append(("Setup", flags))
            self.loaded_hash = "current"
            return 1
        if method == "GetErrorText":
            return "mock error"
        return None

    def classMethodObject(self, cls, method):
        return FakeStream()

    def lock(self, a, timeout, name):
        self.locks.append(("lock", name))

    def unlock(self, a, name):
        self.locks.append(("unlock", name))


class FakeStream:
    def __init__(self):
        self.text = ""

    def invoke(self, method, *args):
        if method == "Write":
            self.text += args[0]
        return None


def class_methods() -> set[str]:
    """All ClassMethod names defined in the helper class source."""
    text = CLASS.read_text()
    return {m.group(1) for m in re.finditer(r"ClassMethod\s+(\w+)\(", text)}


# --------------------------------------------------------------------------- helper class


def test_every_method_the_module_calls_is_in_the_class():
    """The module names IVRDev.Setup methods as strings; each must be defined."""
    called = set(re.findall(r'_call\(\s*\w+,\s*"(\w+)"', SCRIPT.read_text()))
    called |= set(re.findall(r'HELPER,\s*"(\w+)"', SCRIPT.read_text()))
    assert called, "found no IVRDev.Setup calls in tcp_setup.py"
    assert called <= class_methods(), called - class_methods()


def test_the_class_lives_outside_shipped_sources():
    assert "iris_src" not in str(CLASS)
    assert "iris_vector_rag" not in str(CLASS)
    assert re.search(r"^Class IVRDev\.Setup\b", CLASS.read_text(), re.M)


def test_source_hash_length(ts):
    assert len(ts.source_hash()) >= 16


def test_helper_source_stamps_the_hash(ts):
    text = ts.helper_source()
    assert f'Parameter SOURCEHASH = "{ts.source_hash()}";' in text


def test_helper_loads_when_absent(ts):
    native = FakeIRIS(loaded_hash=None)
    assert ts.ensure_helper(native) is True
    assert native.loaded_hash == "current"
    assert native.loads == [("Setup", "ck-d")]
    assert native.locks == [("lock", "^IVRDev.Setup"), ("unlock", "^IVRDev.Setup")]


def test_helper_reloads_when_source_changed(ts):
    native = FakeIRIS(loaded_hash="stale")
    assert ts.ensure_helper(native) is True
    assert native.loads == [("Setup", "ck-d")]


def test_helper_not_reloaded_when_current(ts):
    native = FakeIRIS(loaded_hash=None)
    native.loaded_hash = ts.source_hash()
    assert ts.ensure_helper(native) is False
    assert native.loads == []


# --------------------------------------------------------------------------- unexpire


def test_unexpire_calls_the_security_users_method(ts):
    native = FakeIRIS(loaded_hash=ts.source_hash())
    ts.unexpire_passwords(native)
    assert ("UnExpireUserPasswords", "*") in native.calls


def test_unexpire_with_pattern(ts):
    native = FakeIRIS(loaded_hash=ts.source_hash())
    ts.unexpire_passwords(native, "Admin*")
    assert ("UnExpireUserPasswords", "Admin*") in native.calls


# --------------------------------------------------------------------------- connection


def test_connects_to_sys_on_the_configured_port(ts, monkeypatch):
    seen = {}

    class Conn:
        def close(self):
            seen["closed"] = True

    fake_iris = type(sys)("iris")
    fake_iris.createConnection = lambda *a, **k: seen.update(args=a, kwargs=k) or Conn()
    fake_iris.createIRIS = lambda conn: FakeIRIS(loaded_hash=ts.source_hash())
    monkeypatch.setitem(sys.modules, "iris", fake_iris)
    monkeypatch.setenv("IRIS_PORT", "51972")
    monkeypatch.setenv("IRIS_USERNAME", "_SYSTEM")
    monkeypatch.setenv("IRIS_PASSWORD", "SYS")

    with ts.sys_native() as native:
        assert isinstance(native, FakeIRIS)
    assert seen["kwargs"]["namespace"] == "%SYS"
    assert seen["kwargs"]["port"] == 51972
    assert seen["kwargs"]["hostname"] == "127.0.0.1"
    assert seen["kwargs"]["username"] == "_SYSTEM"
    assert seen["kwargs"]["password"] == "SYS"
    assert seen["closed"]


# --------------------------------------------------------------------------- CLI


def test_cli_unexpire(ts, monkeypatch):
    unexpired = []
    monkeypatch.setattr(
        ts,
        "sys_native",
        _fake_ctx(FakeIRIS(loaded_hash="x")),
    )
    monkeypatch.setattr(
        ts, "unexpire_passwords", lambda native, pattern="*": unexpired.append(pattern)
    )
    assert ts.main(["unexpire"]) == 0
    assert unexpired == ["*"]


def test_cli_unexpire_with_pattern(ts, monkeypatch):
    unexpired = []
    monkeypatch.setattr(
        ts,
        "sys_native",
        _fake_ctx(FakeIRIS(loaded_hash="x")),
    )
    monkeypatch.setattr(
        ts, "unexpire_passwords", lambda native, pattern="*": unexpired.append(pattern)
    )
    assert ts.main(["unexpire", "Admin*"]) == 0
    assert unexpired == ["Admin*"]


def test_cli_rejects_unknown_command(ts):
    assert ts.main(["create", "NS"]) == 2
    assert ts.main([]) == 2


def _fake_ctx(native):
    @contextmanager
    def ctx(*a, **k):
        yield native

    return ctx
