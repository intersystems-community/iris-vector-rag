"""A session identity says which instance, which namespace, and as whom.

Feature: 083-iris-session (FR-002, FR-019)
"""

import pytest

from iris_vector_rag.common.exceptions import ValidationError
from iris_vector_rag.session import SessionIdentity


def _identity(**overrides):
    fields = {
        "host": "localhost",
        "port": 51972,
        "namespace": "USER",
        "username": "_SYSTEM",
        "password": "SYS",
    }
    fields.update(overrides)
    return SessionIdentity(**fields)


def test_the_key_is_everything_but_the_password():
    assert _identity().key == ("localhost", 51972, "USER", "_SYSTEM")


def test_two_identities_differing_only_in_password_are_the_same_session():
    # A wrong password is a failure to connect, not a second session.
    assert _identity(password="SYS") == _identity(password="somethingelse")
    assert hash(_identity(password="SYS")) == hash(_identity(password="other"))


def test_a_different_namespace_is_a_different_session():
    assert _identity(namespace="USER") != _identity(namespace="RAG")


def test_the_password_appears_in_no_repr():
    text = repr(_identity(password="hunter2"))
    assert "hunter2" not in text
    assert "localhost" in text and "USER" in text


def test_the_label_names_the_instance_and_namespace():
    assert _identity().label == "localhost:51972/USER"


@pytest.mark.parametrize(
    "overrides,parameter",
    [
        ({"host": "  "}, "host"),
        ({"port": 0}, "port"),
        ({"port": 99999}, "port"),
        ({"port": "51972"}, "port"),
        ({"namespace": "not a namespace"}, "namespace"),
        ({"namespace": ""}, "namespace"),
        ({"username": ""}, "username"),
    ],
)
def test_an_invalid_identity_cannot_exist(overrides, parameter):
    with pytest.raises(ValidationError) as caught:
        _identity(**overrides)
    assert caught.value.parameter_name == parameter


def test_from_env_reads_the_documented_variables(monkeypatch):
    monkeypatch.setenv("IRIS_HOST", "iris.example")
    monkeypatch.setenv("IRIS_PORT", "1234")
    monkeypatch.setenv("IRIS_NAMESPACE", "RAG")
    monkeypatch.setenv("IRIS_USER", "someone")
    monkeypatch.setenv("IRIS_PASSWORD", "secret")

    identity = SessionIdentity.from_env()

    assert identity.key == ("iris.example", 1234, "RAG", "someone")
    assert identity.password == "secret"


def test_from_env_accepts_the_username_alias(monkeypatch):
    monkeypatch.delenv("IRIS_USER", raising=False)
    monkeypatch.setenv("IRIS_USERNAME", "aliased")
    monkeypatch.setenv("IRIS_PORT", "51972")

    assert SessionIdentity.from_env().username == "aliased"


def test_from_config_uses_the_configuration_manager():
    class Config:
        def get_database_config(self):
            return {
                "host": "confighost",
                "port": 4321,
                "namespace": "CFG",
                "username": "cfguser",
                "password": "cfgpass",
            }

    identity = SessionIdentity.from_config(Config())

    assert identity.key == ("confighost", 4321, "CFG", "cfguser")


def test_from_config_falls_back_to_the_environment_for_what_it_omits(monkeypatch):
    monkeypatch.setenv("IRIS_NAMESPACE", "FROMENV")
    monkeypatch.setenv("IRIS_PORT", "51972")

    class Config:
        def get_database_config(self):
            return {"host": "confighost"}

    identity = SessionIdentity.from_config(Config())

    assert identity.host == "confighost"
    assert identity.namespace == "FROMENV"


def test_the_drivers_own_spelling_of_host_is_accepted(monkeypatch):
    """Callers pass what the driver takes, and `hostname` is what the driver takes."""
    identity = SessionIdentity.from_mapping({"hostname": "elsewhere", "port": 51972})

    assert identity.host == "elsewhere"


def test_a_port_given_as_text_becomes_a_port(monkeypatch):
    identity = SessionIdentity.from_mapping({"host": "h", "port": "51972"})

    assert identity.port == 51972
    assert identity.key == ("h", 51972, identity.namespace, identity.username)


def test_a_port_that_is_not_a_number_is_refused_by_name(monkeypatch):
    # Silently substituting a default would connect to an instance nobody asked for,
    # so the value that cannot be a port is named instead.
    with pytest.raises(ValidationError) as caught:
        SessionIdentity.from_mapping({"host": "h", "port": "not-a-port"})

    assert "not-a-port" in str(caught.value)
