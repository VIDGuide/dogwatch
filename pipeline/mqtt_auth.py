"""mqtt_auth.py — optional broker credentials/TLS for the notifier image.

The detector side (``mqtt_publisher.Publisher``) has always accepted
``mqtt_username`` / ``mqtt_password`` / ``mqtt_tls`` from its camera config.
The notifier side had no equivalent: ``dogwatch-notify.py`` and
``find-dogs-mqtt.py`` called ``connect()`` with host and port only, so there
was no configuration that could secure them. A deployment that put auth on
its broker — the thing the detector's config invites you to do — would simply
have found the notifier unable to connect, with no way to fix it short of
editing the source.

That gap matters more on this side than on the publisher's, because these two
processes are *subscribers* that act on what they receive:
``find-dogs-mqtt.py`` shells out to a camera scan on any message to
``<base>/find-dogs/trigger/+``, and ``dogwatch-notify.py`` sends Telegram
alerts and republishes snapshots. On an unauthenticated broker, anyone who
can reach port 1883 can drive both.

Still *optional*, and still plaintext by default: the documented topology is a
broker on localhost or a trusted LAN (``MQTT_HOST`` defaults to the docker
bridge), and mandating TLS would break every existing deployment for no gain
in that topology. This exists so that securing the broker is a config change
rather than a code change.

Env vars (all optional):
    MQTT_USERNAME / MQTT_PASSWORD  broker credentials
    MQTT_TLS                       "1"/"true"/"yes" to wrap the connection
    MQTT_TLS_CA_CERT               CA bundle for a private/self-signed CA
    MQTT_TLS_INSECURE              "1" to skip hostname verification
"""
import os

__all__ = ["apply_mqtt_auth", "mqtt_security_summary"]

_TRUE = ("1", "true", "yes", "on")


def _flag(env, name):
    return env.get(name, "").strip().lower() in _TRUE


def apply_mqtt_auth(client, env=None):
    """Apply credentials/TLS from the environment to *client*.

    Call before ``connect``/``connect_async``. Returns a short human-readable
    summary of what was applied, suitable for a startup log line.
    """
    env = env if env is not None else os.environ
    username = env.get("MQTT_USERNAME", "").strip()
    password = env.get("MQTT_PASSWORD", "")
    if username:
        client.username_pw_set(username, password or None)
    if _flag(env, "MQTT_TLS"):
        ca_cert = env.get("MQTT_TLS_CA_CERT", "").strip() or None
        client.tls_set(ca_certs=ca_cert)
        if _flag(env, "MQTT_TLS_INSECURE"):
            # Explicit opt-in only. Logged by mqtt_security_summary so a
            # deployment can't quietly sit in this state believing it has TLS.
            client.tls_insecure_set(True)
    return mqtt_security_summary(env)


def mqtt_security_summary(env=None):
    """Describe the connection's security posture for the startup log."""
    env = env if env is not None else os.environ
    parts = []
    parts.append("auth=user" if env.get("MQTT_USERNAME", "").strip() else "auth=none")
    if _flag(env, "MQTT_TLS"):
        parts.append("tls=on(unverified)" if _flag(env, "MQTT_TLS_INSECURE") else "tls=on")
    else:
        parts.append("tls=off")
    return " ".join(parts)
