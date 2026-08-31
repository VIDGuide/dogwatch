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
    ("https://api.telegram.org/botTOKEN/sendMessage",
     "https://api.telegram.org/botTOKEN/sendMessage"),
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
