"""ivg 4.2.0 runs class-method calls as the account IVR connected with, not as _SYSTEM.

`IRISGraphEngine._iris_obj()` used to open its Native API connection as a hard-coded
`_SYSTEM` / `SYS`, so BM25, vector and ledger calls ran with full rights whatever the
caller's role. Against a real IRIS, as three throwaway accounts that differ only in role:

- the connection the calls run on belongs to the account IVR gave the engine;
- a read-only database role can search but not write;
- an account without `%Native_ClassExecution` is refused, instead of being silently
  upgraded to a superuser as it was with ivg 4.1.1.

The accounts and roles are created in %SYS and removed at the end. See
`docs/research/least-privilege-spike.md` in the ivg repository for the minimum rights.
"""

from __future__ import annotations

import uuid

import pytest

from iris_vector_rag.integrations.ivg import create_graph_engine

pytestmark = pytest.mark.integration

DATABASE_RESOURCE = "%DB_USER"
PASSWORD = "Ivr-lp-9x!"


def _sys_object(target):
    import iris
    import iris.dbapi

    connection = iris.dbapi.connect(
        hostname=target.host,
        port=target.port,
        namespace="%SYS",
        username=target.username,
        password=target.password,
    )
    return connection, iris.createIRIS(connection)


@pytest.fixture(scope="module")
def accounts(iris_target, ivg_deployed):
    """name -> role resources; the accounts exist for the module, then are removed."""
    assert iris_target.namespace == "USER", "the database resource below is USER's"
    suffix = uuid.uuid4().hex[:6].upper()
    definitions = {
        "reader": f"{DATABASE_RESOURCE}:R,%Native_ClassExecution:U",
        "writer": f"{DATABASE_RESOURCE}:RW,%Native_ClassExecution:U",
        "no_exec": f"{DATABASE_RESOURCE}:RW",
    }
    created = {}  # kind -> user name
    roles = []
    connection, sys_object = _sys_object(iris_target)
    try:
        for kind, resources in definitions.items():
            role = f"IVRLP_R_{kind.upper()}_{suffix}"
            user = (
                f"IVRLP_U_{kind.upper()}_{suffix}"  # a user cannot share a role's name
            )
            status = sys_object.classMethodValue(
                "Security.Roles", "Create", role, "IVR least-privilege test", resources
            )
            assert str(status) == "1", f"role {role}: {status!r}"
            roles.append(role)
            status = sys_object.classMethodValue(
                "Security.Users", "Create", user, role, PASSWORD, user
            )
            assert str(status) == "1", f"user {user}: {status!r}"
            created[kind] = user
        yield created
    finally:
        for cls, names in (("Users", created.values()), ("Roles", roles)):
            for name in names:
                try:
                    sys_object.classMethodValue(f"Security.{cls}", "Delete", name)
                except Exception:
                    pass
        connection.close()


def _engine_as(target, username, password=PASSWORD):
    import iris.dbapi

    credentials = {
        "host": target.host,
        "port": target.port,
        "namespace": target.namespace,
        "username": username,
        "password": password,
    }
    connection = iris.dbapi.connect(
        hostname=target.host,
        port=target.port,
        namespace=target.namespace,
        username=username,
        password=password,
    )
    return connection, create_graph_engine(connection, credentials=credentials)


def _close(connection, *engines):
    for engine in engines:
        native_connection = getattr(engine, "_native_conn", None)
        if native_connection is not None:
            native_connection.close()
    connection.close()


def _user_name(engine):
    """The IRIS account the engine's class-method connection is logged in as."""
    engine._iris_obj()  # opens the dedicated connection
    cursor = engine._native_conn.cursor()
    try:
        cursor.execute("SELECT USER")
        return str(cursor.fetchone()[0])
    finally:
        cursor.close()


def test_class_method_calls_run_as_the_account_the_engine_was_given(
    iris_target, accounts
):
    connection, engine = _engine_as(iris_target, accounts["writer"])
    try:
        assert _user_name(engine) == accounts["writer"]
        assert _user_name(engine) != "_SYSTEM"
        # A connection of its own, not the one cursor SQL uses.
        assert engine._native_conn is not None
        assert engine._native_conn is not connection
    finally:
        _close(connection, engine)


def test_a_read_role_can_search_a_bm25_index(iris_target, accounts):
    name = f"ivrlp_{uuid.uuid4().hex[:8]}"
    admin_connection, admin = _engine_as(
        iris_target, iris_target.username, iris_target.password
    )
    reader_connection, reader = _engine_as(iris_target, accounts["reader"])
    try:
        admin.bm25_insert(name, "d1", "timeout while connecting to the primary")
        admin.bm25_insert(name, "d2", "disk usage is normal")

        hits = reader.bm25_search(name, "timeout", 5)

        assert [doc_id for doc_id, _ in hits] == ["d1"]
    finally:
        admin.bm25_drop(name)
        _close(reader_connection, reader)
        _close(admin_connection, admin)


def test_a_read_role_cannot_write_the_index(iris_target, accounts):
    name = f"ivrlp_{uuid.uuid4().hex[:8]}"
    connection, reader = _engine_as(iris_target, accounts["reader"])
    try:
        with pytest.raises(Exception, match="PROTECT"):
            reader.bm25_insert(name, "d1", "nobody may write this")
    finally:
        _close(connection, reader)


def test_a_write_role_can_write_and_search(iris_target, accounts):
    name = f"ivrlp_{uuid.uuid4().hex[:8]}"
    connection, writer = _engine_as(iris_target, accounts["writer"])
    try:
        assert writer.bm25_insert(name, "d1", "replica lag is rising")
        assert [i for i, _ in writer.bm25_search(name, "replica", 5)] == ["d1"]
    finally:
        writer.bm25_drop(name)
        _close(connection, writer)


def test_an_account_without_class_execution_is_refused_not_upgraded(
    iris_target, accounts
):
    """ivg 4.1.1 logged in as _SYSTEM here and the call succeeded."""
    connection, engine = _engine_as(iris_target, accounts["no_exec"])
    try:
        with pytest.raises(Exception, match="PROTECT|Native_ClassExecution"):
            engine.bm25_search("anything", "query", 5)
    finally:
        _close(connection, engine)
