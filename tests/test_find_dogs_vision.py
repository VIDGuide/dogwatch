"""Unit tests for pipeline/find-dogs.py's vision verification fallback.

The bug these lock down: ``vision_verify_with`` returned a 2-tuple
``(None, '')`` on all three of its failure paths (unreadable image, API error,
truncated/empty response) but a 3-tuple ``(dog, activity, description)`` on
success. Its caller ``vision_verify`` unpacks three values, so the instant the
*primary* provider failed — exactly the case the fallback exists to cover — the
unpack raised ``ValueError: not enough values to unpack (expected 3, got 2)``
before the ``if dog is None:`` fallback check was ever reached.

Consequences: the fallback provider was unreachable dead code, and a transient
rate-limit on the primary turned into a hard crash for that channel instead of
a failover. ``pipeline/dogwatch_check.py``'s ``make_vision_verifier`` uses a
single-``None`` sentinel and was always correct — only this module was wrong,
and CI only ever ``compileall``-ed it, so nothing caught the mismatch.
"""
import importlib.util
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE = os.path.join(REPO, "pipeline")


def _load():
    """Import find-dogs.py by path (hyphenated name, dir isn't a package)."""
    sys.path.insert(0, PIPELINE)
    spec = importlib.util.spec_from_file_location(
        "find_dogs_under_test", os.path.join(PIPELINE, "find-dogs.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fd = _load()

KEYS = ("http://primary", "model-a", "key-a",
        "http://fallback", "model-b", "key-b")


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status = status

    @property
    def status_code(self):
        # The real requests.Response exposes status_code; find-dogs.py's
        # _reject_redirect() reads it before raise_for_status().
        return self.status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"{self.status} Too Many Requests")

    def json(self):
        return self._payload


def _reply(text):
    return {"choices": [{"message": {"content": text}}]}


def _ok_payload(dog="YES", activity="digging", description="a dog digging"):
    import json
    return _reply(json.dumps(
        {"dog": dog, "activity": activity, "description": description}))


@pytest.fixture
def image(tmp_path):
    p = tmp_path / "frame.jpg"
    p.write_bytes(b"\xff\xd8\xff\xd9")   # minimal JPEG-ish bytes
    return str(p)


# --------------------------------------------------------------------------
# vision_verify_with: every return path must be a 3-tuple
# --------------------------------------------------------------------------

class TestReturnArity:
    def test_success_returns_three_values(self, image, monkeypatch):
        monkeypatch.setattr(fd.requests, "post",
                            lambda *a, **k: FakeResponse(_ok_payload()))
        assert fd.vision_verify_with(image, "u", "m", "k", "primary") == (
            "YES", "digging", "a dog digging")

    def test_unreadable_image_returns_three_values(self):
        result = fd.vision_verify_with(
            "/nonexistent/path/frame.jpg", "u", "m", "k", "primary")
        assert result == (None, "", "")

    def test_api_error_returns_three_values(self, image, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("429 rate limited")
        monkeypatch.setattr(fd.requests, "post", boom)
        assert fd.vision_verify_with(image, "u", "m", "k", "primary") == (
            None, "", "")

    def test_http_error_status_returns_three_values(self, image, monkeypatch):
        monkeypatch.setattr(fd.requests, "post",
                            lambda *a, **k: FakeResponse({}, status=429))
        assert fd.vision_verify_with(image, "u", "m", "k", "primary") == (
            None, "", "")

    def test_empty_response_returns_three_values(self, image, monkeypatch):
        monkeypatch.setattr(fd.requests, "post",
                            lambda *a, **k: FakeResponse(_reply("")))
        assert fd.vision_verify_with(image, "u", "m", "k", "primary") == (
            None, "", "")

    def test_truncated_response_returns_three_values(self, image, monkeypatch):
        monkeypatch.setattr(fd.requests, "post",
                            lambda *a, **k: FakeResponse(_reply("YE")))
        assert fd.vision_verify_with(image, "u", "m", "k", "primary") == (
            None, "", "")

    @pytest.mark.parametrize("payload_text", ["", "YE", '{"dog": "YES"}'])
    def test_every_path_returns_the_same_arity_as_success(
            self, image, monkeypatch, payload_text):
        # The invariant that was violated: success and failure must agree on
        # shape, because the caller unpacks unconditionally.
        monkeypatch.setattr(fd.requests, "post",
                            lambda *a, **k: FakeResponse(_reply(payload_text)))
        assert len(fd.vision_verify_with(image, "u", "m", "k", "p")) == 3


# --------------------------------------------------------------------------
# vision_verify: the fallback must actually be reachable
# --------------------------------------------------------------------------

class TestFallbackReachable:
    def test_primary_failure_falls_through_to_fallback(self, image,
                                                       monkeypatch):
        # The core regression: this used to raise ValueError instead of
        # retrying against the fallback provider.
        seen = []

        def post(url, **kwargs):
            seen.append(url)
            if url == "http://primary":
                raise RuntimeError("429 rate limited")
            return FakeResponse(_ok_payload(dog="NO", activity="",
                                            description="empty yard"))

        monkeypatch.setattr(fd.requests, "post", post)

        assert fd.vision_verify(image, KEYS) == ("NO", "", "empty yard")
        assert seen == ["http://primary", "http://fallback"]

    def test_primary_success_does_not_call_fallback(self, image, monkeypatch):
        seen = []

        def post(url, **kwargs):
            seen.append(url)
            return FakeResponse(_ok_payload())

        monkeypatch.setattr(fd.requests, "post", post)

        assert fd.vision_verify(image, KEYS)[0] == "YES"
        assert seen == ["http://primary"]

    def test_both_providers_failing_degrades_to_none(self, image, monkeypatch):
        # Must not raise: the caller maps a None verdict onto "uncertain" for
        # that channel, so the rest of the scan still completes.
        def boom(*a, **k):
            raise RuntimeError("down")

        monkeypatch.setattr(fd.requests, "post", boom)

        assert fd.vision_verify(image, KEYS) == (None, "", "")

    def test_unreadable_image_does_not_crash_the_scan(self, monkeypatch):
        # grab_frame succeeded but the file vanished/is unreadable — the first
        # of the three failure paths, and the one easiest to hit locally.
        monkeypatch.setattr(fd.requests, "post",
                            lambda *a, **k: FakeResponse(_ok_payload()))
        dog, activity, description = fd.vision_verify(
            "/nonexistent/path/frame.jpg", KEYS)
        assert dog is None
        assert (activity, description) == ("", "")
