"""redact.py — strip embedded credentials out of anything headed for a log.

Camera URLs in this project routinely carry credentials inline
(``rtsp://user:pass@host/...``, ``http://user:pass@nvr/ISAPI/...``). Those
strings reach the logs by two routes, and the second one is easy to miss:

  1. **Directly** — e.g. ``print(f"No frame from {cfg['rtsp_url']}")``.
  2. **Via subprocess exceptions** — both
     ``subprocess.TimeoutExpired.__str__`` and
     ``subprocess.CalledProcessError.__str__`` embed the *entire argv list*,
     and the ffmpeg argv always contains the credential-bearing URL. Because
     an RTSP timeout is the ordinary failure mode for an offline or slow
     camera, ``print(f"...failed: {exc}")`` leaks credentials into
     ``docker logs`` as a matter of routine, not as an edge case.

So the rule for this codebase is: **never interpolate a URL or a caught
subprocess exception into a log line without passing it through
``redact()`` first.**

Everything here is deliberately defensive and must never raise — a
redaction helper that throws inside an error-handling path would convert a
logged warning into a crash, which is strictly worse than the leak it was
trying to prevent.
"""
import re

__all__ = ["redact", "redact_url", "PLACEHOLDER"]

PLACEHOLDER = "***:***"

# scheme://  [userinfo[:password]] @  host...
#
# Deliberately conservative about what can appear in the userinfo section:
# no '/', whitespace, or '@', so we can't run past the authority component
# and start eating path segments. The password group is optional so a bare
# "rtsp://user@host" is still redacted.
_CREDS_RE = re.compile(
    r"([a-zA-Z][a-zA-Z0-9+.\-]*://)"   # 1: scheme://
    r"([^/\s:@]+)"                     # 2: user
    r"(?::([^/\s@]*))?"                # 3: optional :password
    r"@"                               # literal @ terminating userinfo
)


def redact_url(value):
    """Return *value* as a string with any ``user:pass@`` userinfo masked.

    Non-string input is coerced with ``str()`` first, which is what makes
    this safe to wrap around an exception object:

    >>> redact_url("rtsp://bob:hunter2@cam.lan:554/Streaming/Channels/1201")
    'rtsp://***:***@cam.lan:554/Streaming/Channels/1201'
    >>> redact_url("rtsp://bob@cam.lan/stream")
    'rtsp://***:***@cam.lan/stream'
    >>> redact_url("rtsp://cam.lan/stream")
    'rtsp://cam.lan/stream'
    """
    try:
        text = value if isinstance(value, str) else str(value)
    except Exception:
        return "<unprintable>"
    try:
        return _CREDS_RE.sub(rf"\1{PLACEHOLDER}@", text)
    except Exception:
        # Regex can't realistically fail here, but a redaction helper must
        # never be the reason an error path crashes.
        return "<redaction-failed>"


# `redact` is the name to reach for at call sites: it reads correctly whether
# the argument is a URL, an exception, or an argv list.
redact = redact_url
