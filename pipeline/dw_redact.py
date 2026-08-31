"""dw_redact.py — credential redaction for the notifier/pipeline image.

This is a deliberate, behaviour-identical twin of the repo-root ``redact.py``.

Why a copy rather than a shared import: the two Docker images are built from
different contexts (the detector image builds from the repo root and does
``COPY *.py /app/``; the notifier image builds from ``pipeline/``), so a
single module cannot be reached by both without restructuring the build
contexts. Keeping each image self-contained is the lower-risk trade.

The copy is protected against drift by ``tests/test_redact.py``, which
imports *both* modules and asserts they produce identical output for the
same inputs. If you change one, change the other or CI fails.

See ``redact.py`` for the full rationale — the short version is that
``subprocess.TimeoutExpired``/``CalledProcessError`` stringify their entire
argv, and the ffmpeg argv always contains a credential-bearing camera URL,
so an ordinary camera timeout would otherwise print
``rtsp://user:pass@host`` straight into the logs.
"""
import re

__all__ = ["redact", "redact_url", "PLACEHOLDER"]

PLACEHOLDER = "***:***"

_CREDS_RE = re.compile(
    r"([a-zA-Z][a-zA-Z0-9+.\-]*://)"   # 1: scheme://
    r"([^/\s:@]+)"                     # 2: user
    r"(?::([^/\s@]*))?"                # 3: optional :password
    r"@"                               # literal @ terminating userinfo
)


def redact_url(value):
    """Return *value* as a string with any ``user:pass@`` userinfo masked.

    Safe to wrap around an exception object; never raises.
    """
    try:
        text = value if isinstance(value, str) else str(value)
    except Exception:
        return "<unprintable>"
    try:
        return _CREDS_RE.sub(rf"\1{PLACEHOLDER}@", text)
    except Exception:
        return "<redaction-failed>"


redact = redact_url
