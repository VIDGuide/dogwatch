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

# Telegram puts the bot token in the URL *path*, not in userinfo and not in a
# header — ``https://api.telegram.org/bot<token>/sendMessage`` is the only
# form the Bot API offers, so there is nothing to fix at the call site.
#
# That matters because ``requests`` embeds the full request URL in its
# exception text: a ConnectionError reads
# ``HTTPSConnectionPool(host='api.telegram.org', port=443): Max retries
# exceeded with url: /bot<TOKEN>/sendMessage (Caused by ...)``. So the routine
# "internet blipped" failure printed a live bot token — full control of the
# bot, and the same class of leak as the RTSP credentials above.
#
# A token is always ``<bot_id>:<auth_string>``. Requiring that shape after
# ``/bot`` keeps this from matching ordinary paths like ``/bots/list``.
# (``urllib`` callers don't have the leak — URLError's str() carries only the
# errno — but the pattern lives here so any future move to requests is safe.)
_TG_TOKEN_RE = re.compile(r"(/bot)(\d+:[A-Za-z0-9_\-]+)")


def redact_url(value):
    """Return *value* as a string with any embedded credential masked.

    Handles ``user:pass@`` userinfo and Telegram ``/bot<token>/`` path
    segments. Non-string input is coerced with ``str()`` first, which is what
    makes this safe to wrap around an exception object:

    >>> redact_url("rtsp://bob:hunter2@cam.lan:554/Streaming/Channels/1201")
    'rtsp://***:***@cam.lan:554/Streaming/Channels/1201'
    >>> redact_url("rtsp://bob@cam.lan/stream")
    'rtsp://***:***@cam.lan/stream'
    >>> redact_url("rtsp://cam.lan/stream")
    'rtsp://cam.lan/stream'
    >>> redact_url("https://api.telegram.org/bot8123:AAH-xY_9q/sendMessage")
    'https://api.telegram.org/bot***:***/sendMessage'
    """
    try:
        text = value if isinstance(value, str) else str(value)
    except Exception:
        return "<unprintable>"
    try:
        text = _CREDS_RE.sub(rf"\1{PLACEHOLDER}@", text)
        return _TG_TOKEN_RE.sub(rf"\1{PLACEHOLDER}", text)
    except Exception:
        # Regex can't realistically fail here, but a redaction helper must
        # never be the reason an error path crashes.
        return "<redaction-failed>"


# `redact` is the name to reach for at call sites: it reads correctly whether
# the argument is a URL, an exception, or an argv list.
redact = redact_url
