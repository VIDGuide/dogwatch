"""Unit tests for credential redaction.

Covers both the repo-root ``redact.py`` (detector image) and the
``pipeline/dw_redact.py`` twin (notifier image), and asserts they behave
identically so the deliberate duplication cannot drift.
"""
import importlib.util
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import redact as root_redact


def _load_pipeline_module():
    path = os.path.join(REPO, "pipeline", "dw_redact.py")
    spec = importlib.util.spec_from_file_location("dw_redact_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pipeline_redact = _load_pipeline_module()

BOTH = [root_redact, pipeline_redact]

CASES = [
    # (input, expected)
    ("rtsp://bob:hunter2@cam.lan:554/Streaming/Channels/1201",
     "rtsp://***:***@cam.lan:554/Streaming/Channels/1201"),
    ("http://admin:p%40ss@nvr-ip/ISAPI/Streaming/channels/1201/picture",
     "http://***:***@nvr-ip/ISAPI/Streaming/channels/1201/picture"),
    # Bare user with no password still gets masked.
    ("rtsp://bob@cam.lan/stream", "rtsp://***:***@cam.lan/stream"),
    # Empty password.
    ("rtsp://bob:@cam.lan/stream", "rtsp://***:***@cam.lan/stream"),
    # No credentials -> untouched.
    ("rtsp://cam.lan:554/stream", "rtsp://cam.lan:554/stream"),
    # Telegram carries the bot token in the URL *path*, and requests puts the
    # request URL in its exception text — so this had to become a redacted
    # case, not an untouched one.
    ("https://api.telegram.org/bot8123456:AAH-xY_9qZ/sendMessage",
     "https://api.telegram.org/bot***:***/sendMessage"),
    ("https://api.telegram.org/bot8123456:AAH-xY_9qZ/sendPhoto",
     "https://api.telegram.org/bot***:***/sendPhoto"),
    # The real leak shape: a requests ConnectionError, which embeds the path.
    ("HTTPSConnectionPool(host='api.telegram.org', port=443): Max retries "
     "exceeded with url: /bot8123456:AAH-xY_9qZ/sendMessage (Caused by ...)",
     "HTTPSConnectionPool(host='api.telegram.org', port=443): Max retries "
     "exceeded with url: /bot***:***/sendMessage (Caused by ...)"),
    # Not a token shape -> left alone, so ordinary paths survive.
    ("https://example.test/bots/list", "https://example.test/bots/list"),
    ("https://example.test/bothersome", "https://example.test/bothersome"),
    ("", ""),
]


@pytest.mark.parametrize("mod", BOTH, ids=["root", "pipeline"])
@pytest.mark.parametrize("raw,expected", CASES)
def test_redacts_userinfo(mod, raw, expected):
    assert mod.redact(raw) == expected


@pytest.mark.parametrize("raw,_expected", CASES)
def test_both_implementations_agree(raw, _expected):
    """Guard against the intentional duplication drifting apart."""
    assert root_redact.redact(raw) == pipeline_redact.redact(raw)


@pytest.mark.parametrize("mod", BOTH, ids=["root", "pipeline"])
class TestRobustness:
    def test_multiple_urls_in_one_string(self, mod):
        text = ("failed: rtsp://a:b@h1/s then http://c:d@h2/p")
        out = mod.redact(text)
        assert "a:b" not in out
        assert "c:d" not in out
        assert out.count("***:***") == 2

    def test_does_not_eat_path_after_authority(self, mod):
        # An '@' later in the path must not be treated as userinfo.
        assert mod.redact("rtsp://cam.lan/path@notcreds") == "rtsp://cam.lan/path@notcreds"

    def test_non_string_input_is_coerced(self, mod):
        assert mod.redact(None) == "None"
        assert mod.redact(1234) == "1234"

    def test_never_raises_on_weird_object(self, mod):
        class Boom:
            def __str__(self):
                raise RuntimeError("nope")

        # Must degrade, not propagate — this runs inside error handlers.
        assert mod.redact(Boom()) == "<unprintable>"


@pytest.mark.parametrize("mod", BOTH, ids=["root", "pipeline"])
class TestSubprocessExceptionLeak:
    """The leak that motivated this module: subprocess exceptions stringify
    the entire argv, and the ffmpeg argv carries the credential-bearing URL."""

    URL = "rtsp://user:s3cret@cam.lan:554/Streaming/Channels/1201"

    def test_timeout_expired_argv_is_redacted(self, mod):
        exc = subprocess.TimeoutExpired(
            cmd=["ffmpeg", "-i", self.URL, "-frames:v", "1", "out.jpg"],
            timeout=15,
        )
        raw = str(exc)
        assert "s3cret" in raw, "precondition: the exception really does leak"
        assert "s3cret" not in mod.redact(exc)
        assert "***:***" in mod.redact(exc)

    def test_called_process_error_argv_is_redacted(self, mod):
        exc = subprocess.CalledProcessError(
            returncode=1,
            cmd=["ffmpeg", "-rtsp_transport", "tcp", "-i", self.URL],
        )
        raw = str(exc)
        assert "s3cret" in raw, "precondition: the exception really does leak"
        assert "s3cret" not in mod.redact(exc)


@pytest.mark.parametrize("mod", BOTH, ids=["root", "pipeline"])
class TestTelegramTokenLeak:
    """The second leak of the same shape: ``requests`` embeds the request URL
    in its exception text, and the Telegram Bot API puts the bot token in the
    URL path. So ``print(f'TG send error: {e}')`` on a routine network failure
    published a live bot token (full control of the bot) to the log.

    find-dogs.py is the requests-based Telegram caller; dogwatch-notify.py and
    dogwatch_check.py use urllib, whose URLError carries only the errno.
    """

    TOKEN = "8123456789:AAHrealLookingTokenValue_x-9"

    def test_requests_exception_url_is_redacted(self, mod):
        requests = pytest.importorskip("requests")
        url = f"https://127.0.0.1:9/bot{self.TOKEN}/sendMessage"
        try:
            requests.post(url, json={}, timeout=1)
        except Exception as exc:  # ConnectionError in practice
            raw = str(exc)
            assert self.TOKEN in raw, "precondition: requests really does leak the URL"
            out = mod.redact(exc)
            assert self.TOKEN not in out
            assert "/bot***:***/sendMessage" in out
        else:
            pytest.fail("expected the connection to 127.0.0.1:9 to fail")

    def test_token_alone_is_not_mistaken_for_a_url(self, mod):
        # A bare token with no /bot prefix is not something we can recognise;
        # documented so the limitation is deliberate rather than a surprise.
        assert mod.redact(self.TOKEN) == self.TOKEN
