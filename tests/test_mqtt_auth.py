"""Tests for pipeline/mqtt_auth.py — optional broker credentials/TLS.

The notifier and the find-dogs listener previously had *no* way to
authenticate to the broker: they called connect() with host and port only, so
a deployment that put auth on its broker could not connect at all. These tests
lock down that the env vars are honoured, and — just as importantly — that the
default stays plaintext/unauthenticated so existing localhost/LAN deployments
are untouched.
"""
import importlib.util
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    path = os.path.join(REPO, "pipeline", "mqtt_auth.py")
    spec = importlib.util.spec_from_file_location("mqtt_auth_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ma = _load()


class FakeClient:
    """Records the paho calls apply_mqtt_auth is supposed to make."""

    def __init__(self):
        self.credentials = None
        self.tls = None
        self.tls_insecure = None

    def username_pw_set(self, username, password=None):
        self.credentials = (username, password)

    def tls_set(self, ca_certs=None, **kw):
        self.tls = {"ca_certs": ca_certs, **kw}

    def tls_insecure_set(self, value):
        self.tls_insecure = value


class TestDefaults:
    def test_no_env_leaves_client_untouched(self):
        c = FakeClient()
        ma.apply_mqtt_auth(c, env={})
        assert c.credentials is None
        assert c.tls is None
        assert c.tls_insecure is None

    def test_no_env_summary_is_explicit_about_being_open(self):
        assert ma.mqtt_security_summary({}) == "auth=none tls=off"

    def test_blank_username_is_treated_as_unset(self):
        c = FakeClient()
        ma.apply_mqtt_auth(c, env={"MQTT_USERNAME": "   "})
        assert c.credentials is None


class TestCredentials:
    def test_username_and_password_applied(self):
        c = FakeClient()
        ma.apply_mqtt_auth(c, env={"MQTT_USERNAME": "dw", "MQTT_PASSWORD": "s3cret"})
        assert c.credentials == ("dw", "s3cret")

    def test_username_without_password_passes_none(self):
        """paho distinguishes "no password" from "empty password"."""
        c = FakeClient()
        ma.apply_mqtt_auth(c, env={"MQTT_USERNAME": "dw"})
        assert c.credentials == ("dw", None)

    def test_password_without_username_is_ignored(self):
        c = FakeClient()
        ma.apply_mqtt_auth(c, env={"MQTT_PASSWORD": "s3cret"})
        assert c.credentials is None

    def test_summary_never_contains_the_password(self):
        env = {"MQTT_USERNAME": "dw", "MQTT_PASSWORD": "s3cret"}
        summary = ma.apply_mqtt_auth(FakeClient(), env=env)
        assert "s3cret" not in summary
        assert summary == "auth=user tls=off"


class TestTls:
    @pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on"])
    def test_truthy_values_enable_tls(self, raw):
        c = FakeClient()
        ma.apply_mqtt_auth(c, env={"MQTT_TLS": raw})
        assert c.tls == {"ca_certs": None}

    @pytest.mark.parametrize("raw", ["", "0", "false", "no", "off", "maybe"])
    def test_other_values_leave_tls_off(self, raw):
        c = FakeClient()
        ma.apply_mqtt_auth(c, env={"MQTT_TLS": raw})
        assert c.tls is None

    def test_custom_ca_is_passed_through(self):
        c = FakeClient()
        ma.apply_mqtt_auth(c, env={"MQTT_TLS": "true",
                                   "MQTT_TLS_CA_CERT": "/app/mqtt-ca.crt"})
        assert c.tls == {"ca_certs": "/app/mqtt-ca.crt"}

    def test_insecure_requires_explicit_opt_in(self):
        c = FakeClient()
        ma.apply_mqtt_auth(c, env={"MQTT_TLS": "true"})
        assert c.tls_insecure is None

    def test_insecure_is_surfaced_in_the_summary(self):
        """Skipping verification must not look the same as real TLS in a log."""
        env = {"MQTT_TLS": "true", "MQTT_TLS_INSECURE": "1"}
        c = FakeClient()
        summary = ma.apply_mqtt_auth(c, env=env)
        assert c.tls_insecure is True
        assert summary == "auth=none tls=on(unverified)"

    def test_insecure_without_tls_does_nothing(self):
        c = FakeClient()
        ma.apply_mqtt_auth(c, env={"MQTT_TLS_INSECURE": "1"})
        assert c.tls is None
        assert c.tls_insecure is None


class TestBothTogether:
    def test_auth_and_tls(self):
        env = {"MQTT_USERNAME": "dw", "MQTT_PASSWORD": "p",
               "MQTT_TLS": "true"}
        c = FakeClient()
        assert ma.apply_mqtt_auth(c, env=env) == "auth=user tls=on"
        assert c.credentials == ("dw", "p")
        assert c.tls == {"ca_certs": None}

    def test_falls_back_to_os_environ(self, monkeypatch):
        monkeypatch.setenv("MQTT_USERNAME", "from-environ")
        monkeypatch.delenv("MQTT_TLS", raising=False)
        c = FakeClient()
        ma.apply_mqtt_auth(c)
        assert c.credentials == ("from-environ", None)
