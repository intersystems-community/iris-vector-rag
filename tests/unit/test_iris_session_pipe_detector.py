"""The `iris-session-pipe` detector: piping input into `docker exec iris session`.

Cortex XDR kills processes that pipe ObjectScript into `iris session` over docker
exec or ssh. The detector catches every pattern before they reach production.

Shipped instances (all fixed in this branch):
- scripts/ci/start-iris.sh:137
- tests/support/iris_environment.py:251
- tests/contract/test_objectscript_sdk_fix.py:66
- tests/e2e/test_session_e2e.py:138
- tests/e2e/test_sdk_canonical_vector_type.py:85
- tests/utils/iris_password_reset.py:96-104
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
GATE = ROOT / "scripts/gates/antipatterns.py"


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location("antipatterns_gate_session", GATE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_detects_docker_exec_pipe_in_shell(gate):
    source = """
    printf 'do ##class(Security.Users).UnExpireUserPasswords("*")' \\
        | docker exec -i "$NAME" iris session IRIS -U%SYS
    """
    findings = gate.iris_session_pipes("example.sh", source)
    assert len(findings) == 1
    assert "iris session" in findings[0].message.lower()


def test_detects_docker_exec_dash_T_pipe(gate):
    source = """
    db_test=$(docker-compose exec -T iris_db iris session iris -U%SYS << 'EOF'
    write 1
    halt
EOF
    """
    findings = gate.iris_session_pipes("example.sh", source)
    assert len(findings) == 1


def test_detects_subprocess_with_iris_session(gate):
    source = """
    out = subprocess.run(
        ["docker", "exec", "-i", name, "iris", "session", "IRIS", "-U%SYS"],
        input=script,
        text=True,
    )
    """
    findings = gate.iris_session_pipes("example.py", source)
    assert len(findings) == 1


def test_detects_iris_session_in_popen_list(gate):
    source = """
    done = subprocess.run(
        ["docker", "exec", "-i", self.container, "iris", "session", "IRIS"],
        input=f'zn "USER"\\n{script}\\nhalt\\n',
        capture_output=True,
    )
    """
    findings = gate.iris_session_pipes("test_foo.py", source)
    assert len(findings) == 1


def test_ignores_comment_mentioning_iris_session(gate):
    source = """
    # We used to run iris session here, now we use TCP
    conn = iris.createConnection(...)
    """
    findings = gate.iris_session_pipes("example.py", source)
    assert len(findings) == 0


def test_ignores_session_in_docstring(gate):
    source = '''
    """Run ObjectScript over TCP instead of iris session."""
    def run_code(lines):
        pass
    '''
    findings = gate.iris_session_pipes("example.py", source)
    assert len(findings) == 0


def test_ignores_bare_docker_exec_without_session(gate):
    source = """
    docker exec iris irissession IRIS < script.txt
    """
    findings = gate.iris_session_pipes("example.sh", source)
    assert len(findings) == 0


def test_detects_heredoc_to_session(gate):
    source = """
    docker exec -i iris iris session IRIS -U%SYS << 'EOF'
    do ##class(Security.Users).UnExpireUserPasswords("*")
    halt
EOF
    """
    findings = gate.iris_session_pipes("start.sh", source)
    assert len(findings) == 1


def test_ignores_tcp_replacement_pattern(gate):
    # The new pattern: tcp_setup.py with no pipe
    source = """
    python scripts/dev/tcp_setup.py unexpire
    """
    findings = gate.iris_session_pipes("example.sh", source)
    assert len(findings) == 0
