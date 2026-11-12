#!/usr/bin/env python3
"""Unexpire IRIS passwords over TCP, with no `iris session`.

    python scripts/dev/tcp_setup.py unexpire

Cortex XDR kills processes that pipe ObjectScript into `docker exec iris session`,
so password unexpiry runs through the Native API and a helper class (IVRDev.Setup)
loaded into %SYS over TCP.

Connection: 127.0.0.1:IRIS_PORT (env var, default 51972) as IRIS_USERNAME/IRIS_PASSWORD
(default _SYSTEM/SYS).
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Iterator, Optional

HELPER = "IVRDev.Setup"
HELPER_SOURCE = (
    Path(__file__).resolve().parent / "objectscript" / "IVRDev" / "Setup.cls"
)
HOST = "127.0.0.1"
LOCK_TIMEOUT = 120
_HASH_LINE = 'Parameter SOURCEHASH = "";'


def source_hash() -> str:
    """Hash of the helper class source: a change reloads it."""
    return hashlib.sha256(HELPER_SOURCE.read_bytes()).hexdigest()[:24]


def helper_source() -> str:
    """The class source with SOURCEHASH parameter set to source_hash()."""
    text = HELPER_SOURCE.read_text()
    if _HASH_LINE not in text:
        raise RuntimeError(f"{HELPER_SOURCE}: no `{_HASH_LINE}` line to stamp")
    return text.replace(_HASH_LINE, f'Parameter SOURCEHASH = "{source_hash()}";', 1)


def _loaded_hash(native) -> Optional[str]:
    try:
        return str(native.classMethodValue(HELPER, "SourceHash"))
    except RuntimeError as exc:
        if "CLASS DOES NOT EXIST" not in str(exc):
            raise
        return None


def ensure_helper(native) -> bool:
    """Load and compile IVRDev.Setup into `native`'s namespace (%SYS) unless already current.
    Returns True when it loaded. Under a lock so concurrent calls don't conflict."""
    if _loaded_hash(native) == source_hash():
        return False
    native.lock("", LOCK_TIMEOUT, "^IVRDev.Setup")
    try:
        if _loaded_hash(native) == source_hash():
            return False
        stream = native.classMethodObject("%Stream.GlobalCharacter", "%New")
        stream.invoke("Write", helper_source())
        stream.invoke("Rewind")
        status = native.classMethodValue("%SYSTEM.OBJ", "LoadStream", stream, "ck-d")
        if not (status == 1 or str(status) == "1"):
            text = native.classMethodValue("%SYSTEM.Status", "GetErrorText", status)
            raise RuntimeError(f"loading {HELPER} into %SYS failed: {text}")
    finally:
        native.unlock("", "^IVRDev.Setup")
    return True


@contextlib.contextmanager
def sys_native(port: Optional[int] = None, host: str = HOST) -> Iterator:
    """A Native API handle in %SYS with the helper loaded; connection closed after."""
    import iris

    conn = iris.createConnection(
        hostname=host,
        port=int(port or os.environ.get("IRIS_PORT", "51972")),
        namespace="%SYS",
        username=os.environ.get("IRIS_USERNAME", "_SYSTEM"),
        password=os.environ.get("IRIS_PASSWORD", "SYS"),
    )
    try:
        native = iris.createIRIS(conn)
        ensure_helper(native)
        yield native
    finally:
        conn.close()


def _call(native, method: str, *args, what: str = "") -> str:
    """IVRDev.Setup.<method>(args); raises naming `what` when it answers FAIL."""
    out = native.classMethodValue(HELPER, method, *args)
    out = "" if out is None else str(out)
    if out.startswith("FAIL"):
        label = what or method
        raise RuntimeError(f"{label}: {out}")
    return out


def unexpire_passwords(native, pattern: str = "*") -> None:
    """Unexpire all user passwords matching `pattern`."""
    _call(native, "UnExpireUserPasswords", pattern)


def run(native, namespace: str, code: str) -> str:
    """Run ObjectScript code in namespace and return captured output."""
    result = _call(native, "Run", namespace, code, what=f"run/{namespace}")
    return result


def main(argv: list) -> int:
    if argv[:1] == ["unexpire"] and len(argv) <= 2:
        pattern = argv[1] if len(argv) == 2 else "*"
        with sys_native() as native:
            unexpire_passwords(native, pattern)
        print(f"passwords unexpired: {pattern}")
        return 0

    if argv[:1] == ["run"]:
        # tcp_setup.py run [--namespace NS] [code]
        # Reads code from stdin if not provided as args
        namespace = "USER"
        args_rest = argv[1:]

        if len(args_rest) >= 2 and args_rest[0] == "--namespace":
            namespace = args_rest[1]
            args_rest = args_rest[2:]

        if args_rest:
            code = " ".join(args_rest)
        else:
            code = sys.stdin.read()

        with sys_native() as native:
            output = run(native, namespace, code)
        print(output, end="")
        return 0

    print("usage: tcp_setup.py unexpire [pattern]")
    print("       tcp_setup.py run [--namespace NS] [code]")
    print("                    (reads code from stdin if not provided)")
    print(f"refused: {argv}")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
