"""Unit tests for pipeline/dogwatch_check.py.

This logic was previously embedded in a bash heredoc and therefore untestable
— these tests exist to lock down the watermark, offset and dedupe behaviour
that governs whether an event gets alerted, double-alerted, or silently
dropped.
"""
import importlib.util
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE = os.path.join(REPO, "pipeline")


def _load():
    """Import dogwatch_check.py by path (its dir isn't a package)."""
    sys.path.insert(0, PIPELINE)
    spec = importlib.util.spec_from_file_location(
        "dogwatch_check_under_test", os.path.join(PIPELINE, "dogwatch_check.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dc = _load()


def write_events(path, events):
    with open(path, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def ev(ts, slug="digging", state="ON", camera="rear-east", **extra):
    d = {"ts": ts, "topic": f"dogwatch/{camera}/{slug}", "state": state,
         "camera": camera}
    d.update(extra)
    return d


class TestCheckStateRobustness:
    """The state file is the single point of failure that could previously
    stop all alerting forever, so every malformed input must degrade to
    'start from zero' rather than raise."""

    def test_missing_file_starts_at_zero(self, tmp_path):
        st = dc.CheckState(str(tmp_path / "nope.json")).load()
        assert st.ts == 0.0
        assert st.offset == 0

    def test_empty_file_starts_at_zero(self, tmp_path):
        p = tmp_path / "state.json"
        p.write_text("")
        st = dc.CheckState(str(p)).load()
        assert st.ts == 0.0

    @pytest.mark.parametrize("garbage", [
        "not json at all",
        "{truncated",
        '{"ts": "banana"}',
        '{"ts": null}',
        "[]",
        '{"ts": NaN}',
    ])
    def test_garbage_never_raises(self, tmp_path, garbage):
        # This is the regression: the old code called float() on the file's
        # contents outside any try/except, so one torn write wedged every
        # future cycle with a traceback and alerts stopped permanently.
        p = tmp_path / "state.json"
        p.write_text(garbage)
        st = dc.CheckState(str(p)).load()
        assert st.ts == 0.0

    def test_nan_and_inf_are_rejected(self, tmp_path):
        p = tmp_path / "state.json"
        p.write_text('{"ts": 1e999, "offset": 5}')
        st = dc.CheckState(str(p)).load()
        assert st.ts == 0.0  # inf would poison every future comparison

    def test_negative_offset_clamped(self, tmp_path):
        p = tmp_path / "state.json"
        p.write_text('{"ts": 5, "offset": -20}')
        st = dc.CheckState(str(p)).load()
        assert st.offset == 0

    def test_legacy_bare_float_watermark_is_inherited(self, tmp_path):
        legacy = tmp_path / "dogwatch-last-ts"
        legacy.write_text("1234.5\n")
        st = dc.CheckState(str(tmp_path / "state.json"), str(legacy)).load()
        assert st.ts == pytest.approx(1234.5)
        assert st.offset == 0

    def test_legacy_garbage_does_not_raise(self, tmp_path):
        legacy = tmp_path / "dogwatch-last-ts"
        legacy.write_text("\x00\x00partial")
        st = dc.CheckState(str(tmp_path / "state.json"), str(legacy)).load()
        assert st.ts == 0.0


class TestCheckStateSave:
    def test_round_trip(self, tmp_path):
        p = str(tmp_path / "state.json")
        st = dc.CheckState(p)
        st.advance(ts=99.5, offset=420)
        assert st.save() is True
        again = dc.CheckState(p).load()
        assert again.ts == pytest.approx(99.5)
        assert again.offset == 420

    def test_save_leaves_no_tmp_file_behind(self, tmp_path):
        p = str(tmp_path / "state.json")
        st = dc.CheckState(p)
        st.advance(ts=1.0, offset=1)
        st.save()
        assert not os.path.exists(p + ".tmp")
        assert os.listdir(tmp_path) == ["state.json"]

    def test_advance_is_monotonic(self, tmp_path):
        st = dc.CheckState(str(tmp_path / "s.json"))
        st.advance(ts=100.0)
        st.advance(ts=50.0)   # older event must not rewind the watermark
        assert st.ts == pytest.approx(100.0)

    def test_advance_ignores_unparseable_ts(self, tmp_path):
        st = dc.CheckState(str(tmp_path / "s.json"))
        st.advance(ts=10.0)
        st.advance(ts="banana")
        assert st.ts == pytest.approx(10.0)


class TestReadNewEvents:
    def test_reads_all_lines_from_offset_zero(self, tmp_path):
        p = tmp_path / "events.jsonl"
        write_events(p, [ev(1), ev(2), ev(3)])
        st = dc.CheckState(str(tmp_path / "s.json"))
        got = [e for e, _ in dc.read_new_events(str(p), st) if e]
        assert [e["ts"] for e in got] == [1, 2, 3]

    def test_offset_skips_already_consumed_lines(self, tmp_path):
        """The event log is append-only and never rotated; re-parsing it in
        full every cycle was pure waste."""
        p = tmp_path / "events.jsonl"
        write_events(p, [ev(1), ev(2)])
        st = dc.CheckState(str(tmp_path / "s.json"))
        consumed = [off for _, off in dc.read_new_events(str(p), st)][-1]

        with open(p, "a") as f:
            f.write(json.dumps(ev(3)) + "\n")

        st.advance(offset=consumed)
        got = [e for e, _ in dc.read_new_events(str(p), st) if e]
        assert [e["ts"] for e in got] == [3]

    def test_shrunk_file_is_reread_from_start(self, tmp_path):
        """Container recreate truncates /tmp; a stale large offset must not
        cause every event to be skipped forever."""
        p = tmp_path / "events.jsonl"
        write_events(p, [ev(7)])
        st = dc.CheckState(str(tmp_path / "s.json"))
        st.advance(offset=999_999)
        got = [e for e, _ in dc.read_new_events(str(p), st) if e]
        assert [e["ts"] for e in got] == [7]

    def test_partial_trailing_line_is_not_consumed(self, tmp_path):
        """A concurrent append mid-write must be left for the next cycle
        rather than logged as malformed and dropped."""
        p = tmp_path / "events.jsonl"
        with open(p, "w") as f:
            f.write(json.dumps(ev(1)) + "\n")
            f.write('{"ts": 2, "top')  # torn write, no newline
        st = dc.CheckState(str(tmp_path / "s.json"))
        results = list(dc.read_new_events(str(p), st))
        events = [e for e, _ in results if e]
        assert [e["ts"] for e in events] == [1]
        # Offset must stop before the partial line so it is re-read intact.
        assert results[-1][1] == len(json.dumps(ev(1)) + "\n")

    def test_malformed_line_is_skipped_not_fatal(self, tmp_path):
        p = tmp_path / "events.jsonl"
        with open(p, "w") as f:
            f.write("{not json}\n")
            f.write(json.dumps(ev(5)) + "\n")
        st = dc.CheckState(str(tmp_path / "s.json"))
        got = [e for e, _ in dc.read_new_events(str(p), st) if e]
        assert [e["ts"] for e in got] == [5]

    def test_missing_file_yields_nothing(self, tmp_path):
        st = dc.CheckState(str(tmp_path / "s.json"))
        assert list(dc.read_new_events(str(tmp_path / "nope.jsonl"), st)) == []


def make_cfg(tmp_path, **over):
    env = {
        "DOGWATCH_STATUS_FILE": str(tmp_path / "events.jsonl"),
        "DOGWATCH_STATE_FILE": str(tmp_path / "state.json"),
        "DOGWATCH_LAST_TS_FILE": str(tmp_path / "legacy-ts"),
        "DOGWATCH_WORKSPACE_DIR": str(tmp_path / "ws"),
        "DOGWATCH_NOTIFY_CONFIG": str(tmp_path / "notify.json"),
        "DOGWATCH_SECRETS_FILE": str(tmp_path / "secrets.json"),
        # Stands in for /tmp — the directory the notifier writes its event
        # snapshots into, and the only place (besides the workspace dir) that
        # safe_snapshot_path will accept a path from.
        "DOGWATCH_SNAPSHOT_ALLOW_DIRS": str(tmp_path / "snaps"),
        "DOGWATCH_VISION_API_KEY": "k",
        "DOGWATCH_BOT_TOKEN": "t",
        "DOGWATCH_CHAT_ID": "c",
    }
    env.update(over)
    return dc.Config(env)


def write_snapshot(tmp_path, name="dogwatch_snap_rear-east_1000.jpg"):
    """Create a snapshot file where the notifier would have written it."""
    snaps = tmp_path / "snaps"
    snaps.mkdir(exist_ok=True)
    p = snaps / name
    p.write_bytes(b"jpegdata")
    return p


class TestSnapshotPathAllowlist:
    """The ``snapshot`` field is attacker-reachable and every use of it is an
    exfiltration primitive (staged copy -> Telegram upload -> vision API), so
    it must be constrained to paths the notifier could actually have written.

    The event log's default home is /tmp/dogwatch-events.jsonl: a predictable
    name in a world-writable directory, append-only, and never authenticated.
    Appending one line naming an arbitrary file was enough to have it mailed
    out over Telegram.
    """

    def test_accepts_a_notifier_written_snapshot(self, tmp_path):
        cfg = make_cfg(tmp_path)
        snap = write_snapshot(tmp_path)
        assert dc.safe_snapshot_path(cfg, str(snap)) == os.path.realpath(str(snap))

    def test_accepts_a_fresh_capture_name(self, tmp_path):
        cfg = make_cfg(tmp_path)
        snap = write_snapshot(tmp_path, "dogwatch_check_camera_1700000000.jpg")
        assert dc.safe_snapshot_path(cfg, str(snap))

    def test_accepts_a_staged_workspace_copy(self, tmp_path):
        cfg = make_cfg(tmp_path)
        ws = tmp_path / "ws"
        ws.mkdir()
        p = ws / "dogwatch_1700000000.jpg"
        p.write_bytes(b"jpegdata")
        assert dc.safe_snapshot_path(cfg, str(p))

    @pytest.mark.parametrize("path", [
        "/etc/passwd",
        "/root/.openclaw/secrets.json",
        "../../etc/passwd",
        "",
    ])
    def test_rejects_paths_outside_the_allowlist(self, tmp_path, path):
        cfg = make_cfg(tmp_path)
        assert dc.safe_snapshot_path(cfg, path) == ""

    def test_rejects_traversal_out_of_an_allowed_dir(self, tmp_path):
        cfg = make_cfg(tmp_path)
        (tmp_path / "snaps").mkdir(exist_ok=True)
        secret = tmp_path / "secrets.json"
        secret.write_text("{}")
        # Lands on a real file, and the string even starts with an allowed dir.
        assert dc.safe_snapshot_path(
            cfg, str(tmp_path / "snaps" / ".." / "secrets.json")) == ""

    def test_rejects_a_symlink_planted_with_an_allowed_name(self, tmp_path):
        cfg = make_cfg(tmp_path)
        snaps = tmp_path / "snaps"
        snaps.mkdir(exist_ok=True)
        secret = tmp_path / "secrets.json"
        secret.write_text("{}")
        link = snaps / "dogwatch_snap_camera_1700000000.jpg"
        link.symlink_to(secret)
        assert dc.safe_snapshot_path(cfg, str(link)) == ""

    def test_rejects_a_non_jpeg_in_an_allowed_dir(self, tmp_path):
        cfg = make_cfg(tmp_path)
        p = write_snapshot(tmp_path, "dogwatch_snap_camera_1.json")
        assert dc.safe_snapshot_path(cfg, str(p)) == ""

    def test_rejects_a_foreign_filename_in_an_allowed_dir(self, tmp_path):
        cfg = make_cfg(tmp_path)
        p = write_snapshot(tmp_path, "id_rsa.jpg")
        assert dc.safe_snapshot_path(cfg, str(p)) == ""

    def test_rejects_a_directory(self, tmp_path):
        cfg = make_cfg(tmp_path)
        d = tmp_path / "snaps" / "dogwatch_snap_camera_1.jpg"
        d.mkdir(parents=True)
        assert dc.safe_snapshot_path(cfg, str(d)) == ""

    def test_rejects_a_missing_file(self, tmp_path):
        cfg = make_cfg(tmp_path)
        (tmp_path / "snaps").mkdir(exist_ok=True)
        assert dc.safe_snapshot_path(
            cfg, str(tmp_path / "snaps" / "dogwatch_snap_camera_1.jpg")) == ""

    @pytest.mark.parametrize("bad", [None, 12345, {"path": "/tmp/x.jpg"}])
    def test_rejects_non_string_values(self, tmp_path, bad):
        cfg = make_cfg(tmp_path)
        assert dc.safe_snapshot_path(cfg, bad) == ""

    def test_event_naming_a_secret_yields_no_snapshot(self, tmp_path):
        """End-to-end: the malicious log line reaches collect_pending and comes
        out with no snapshot rather than a staged copy of the secret."""
        cfg = make_cfg(tmp_path)
        secret = tmp_path / "secrets.json"
        secret.write_text('{"botToken": "hunter2"}')
        write_events(cfg.status_file, [ev(1000.0, snapshot=str(secret))])
        st = dc.CheckState(cfg.state_file).load()
        pending, _, _ = dc.collect_pending(cfg, st, {}, now=1000.0)
        assert len(pending) == 1
        assert pending[0]["snapshot"] == ""

    def test_default_allowlist_is_tmp_plus_workspace(self):
        cfg = dc.Config({"DOGWATCH_WORKSPACE_DIR": "/var/dogwatch/ws"})
        # realpath'd, so the comparison in safe_snapshot_path is symlink-stable
        # (macOS resolves /tmp and /var under /private).
        assert os.path.realpath("/tmp") in cfg.snapshot_allow_dirs
        assert os.path.realpath("/var/dogwatch/ws") in cfg.snapshot_allow_dirs


class TestCollectPendingDedupe:
    def test_single_event_is_pending(self, tmp_path):
        cfg = make_cfg(tmp_path)
        write_events(cfg.status_file, [ev(1000.0)])
        st = dc.CheckState(cfg.state_file).load()
        pending, max_seen, _ = dc.collect_pending(cfg, st, {}, now=1000.0)
        assert len(pending) == 1
        assert max_seen == pytest.approx(1000.0)

    def test_off_events_are_ignored(self, tmp_path):
        cfg = make_cfg(tmp_path)
        write_events(cfg.status_file, [ev(1000.0, state="OFF")])
        st = dc.CheckState(cfg.state_file).load()
        pending, max_seen, _ = dc.collect_pending(cfg, st, {}, now=1000.0)
        assert pending == []
        # ...but they still advance the high-water mark.
        assert max_seen == pytest.approx(1000.0)

    def test_events_at_or_below_watermark_are_skipped(self, tmp_path):
        cfg = make_cfg(tmp_path)
        write_events(cfg.status_file, [ev(1000.0)])
        st = dc.CheckState(cfg.state_file)
        st.ts = 1000.0
        pending, _, _ = dc.collect_pending(cfg, st, {}, now=1000.0)
        assert pending == []

    def test_events_older_than_max_age_are_skipped(self, tmp_path):
        cfg = make_cfg(tmp_path, DOGWATCH_MAX_EVENT_AGE="100")
        write_events(cfg.status_file, [ev(500.0)])
        st = dc.CheckState(cfg.state_file).load()
        pending, _, _ = dc.collect_pending(cfg, st, {}, now=1000.0)
        assert pending == []

    def test_repeat_within_window_is_deduped(self, tmp_path):
        cfg = make_cfg(tmp_path, DOGWATCH_DEDUPE_WINDOW="90")
        write_events(cfg.status_file, [ev(1000.0), ev(1030.0)])
        st = dc.CheckState(cfg.state_file).load()
        pending, _, _ = dc.collect_pending(cfg, st, {}, now=1030.0)
        assert len(pending) == 1
        assert pending[0]["ts"] == pytest.approx(1000.0)

    def test_dedupe_window_is_anchored_to_first_event(self, tmp_path):
        """Regression: the window used to be compared against the *latest*
        kept entry's ts and then overwritten with it, so a long incident
        emitting a repeat every ~80s slid the 90s window forward forever and
        collapsed into a single reported event."""
        cfg = make_cfg(tmp_path, DOGWATCH_DEDUPE_WINDOW="90")
        # 0, +80, +160: each gap is under 90s, but the span is 160s, so the
        # third event belongs to a new incident.
        write_events(cfg.status_file, [ev(1000.0), ev(1080.0), ev(1160.0)])
        st = dc.CheckState(cfg.state_file).load()
        pending, _, _ = dc.collect_pending(cfg, st, {}, now=1160.0)
        assert [p["ts"] for p in pending] == [pytest.approx(1000.0),
                                             pytest.approx(1160.0)]

    def test_separate_incidents_outside_window_both_kept(self, tmp_path):
        cfg = make_cfg(tmp_path, DOGWATCH_DEDUPE_WINDOW="30")
        write_events(cfg.status_file, [ev(1000.0), ev(1100.0)])
        st = dc.CheckState(cfg.state_file).load()
        pending, _, _ = dc.collect_pending(cfg, st, {}, now=1100.0)
        assert len(pending) == 2

    def test_different_cameras_are_not_deduped_together(self, tmp_path):
        cfg = make_cfg(tmp_path)
        write_events(cfg.status_file, [
            ev(1000.0, camera="rear-east"),
            ev(1001.0, camera="camera"),
        ])
        st = dc.CheckState(cfg.state_file).load()
        pending, _, _ = dc.collect_pending(cfg, st, {}, now=1001.0)
        assert len(pending) == 2

    def test_different_event_types_are_not_deduped_together(self, tmp_path):
        cfg = make_cfg(tmp_path)
        write_events(cfg.status_file, [
            ev(1000.0, slug="dog_at_fence"),
            ev(1001.0, slug="digging"),
        ])
        st = dc.CheckState(cfg.state_file).load()
        pending, _, _ = dc.collect_pending(cfg, st, {}, now=1001.0)
        assert {p["type"] for p in pending} == {"dog_at_fence", "digging"}

    def test_stored_snapshot_is_staged_into_workspace(self, tmp_path):
        cfg = make_cfg(tmp_path)
        snap = write_snapshot(tmp_path)
        write_events(cfg.status_file, [ev(1000.0, snapshot=str(snap))])
        st = dc.CheckState(cfg.state_file).load()
        pending, _, _ = dc.collect_pending(cfg, st, {}, now=1000.0)
        assert pending[0]["snapshot"].startswith(cfg.workspace_dir)
        assert os.path.exists(pending[0]["snapshot"])
        # A staged copy is not scratch — it must not be auto-deleted.
        assert pending[0]["temp_snapshot"] is False

    def test_repeat_with_snapshot_upgrades_entry_without_moving_anchor(self, tmp_path):
        cfg = make_cfg(tmp_path, DOGWATCH_DEDUPE_WINDOW="90")
        snap = write_snapshot(tmp_path)
        write_events(cfg.status_file, [
            ev(1000.0),                              # no snapshot
            ev(1030.0, snapshot=str(snap)),          # real frame arrives
        ])
        st = dc.CheckState(cfg.state_file).load()
        pending, _, _ = dc.collect_pending(cfg, st, {}, now=1030.0)
        assert len(pending) == 1
        assert pending[0]["ts"] == pytest.approx(1000.0)  # anchor unchanged
        assert os.path.exists(pending[0]["snapshot"])

    def test_events_missing_required_fields_are_skipped(self, tmp_path):
        cfg = make_cfg(tmp_path)
        with open(cfg.status_file, "w") as f:
            f.write(json.dumps({"topic": "dogwatch/digging", "state": "ON"}) + "\n")
            f.write(json.dumps(ev(1000.0)) + "\n")
        st = dc.CheckState(cfg.state_file).load()
        pending, _, _ = dc.collect_pending(cfg, st, {}, now=1000.0)
        assert len(pending) == 1


class TestSanitizeMd:
    """Model-authored prose used to go into Telegram with parse_mode=Markdown
    unescaped: one unbalanced '*' returned HTTP 400, which the send helper
    swallowed, so the verdict was silently lost."""

    @pytest.mark.parametrize("ch", ["*", "_", "[", "]", "`"])
    def test_markdown_specials_removed(self, ch):
        assert ch not in dc.sanitize_md(f"a {ch}dog{ch} digging")

    def test_newlines_collapsed(self):
        assert "\n" not in dc.sanitize_md("two dogs\ndigging")

    def test_empty_and_none_safe(self):
        assert dc.sanitize_md("") == ""
        assert dc.sanitize_md(None) == ""

    def test_long_text_is_capped(self):
        out = dc.sanitize_md("x" * 5000)
        assert len(out) <= dc.MAX_DESCRIPTION_CHARS

    def test_normal_sentence_survives_intact(self):
        text = "2 dogs digging near the fence"
        assert dc.sanitize_md(text) == text


class TestConfigDefaults:
    def test_bad_numeric_env_falls_back_to_default(self, tmp_path):
        cfg = make_cfg(tmp_path, DOGWATCH_DEDUPE_WINDOW="not-a-number")
        assert cfg.dedupe_window == 90.0

    def test_max_event_age_is_generous_by_default(self, tmp_path):
        # The old 7-minute wall-clock cutoff had to exceed the loop period plus
        # the cycle duration, and a busy cycle (30s siren follow-up per
        # confirmed digging event) silently blew past it.
        cfg = make_cfg(tmp_path)
        assert cfg.max_event_age >= 900


class TestResolveCredentials:
    def test_env_bot_token_wins(self, tmp_path):
        cfg = make_cfg(tmp_path, DOGWATCH_BOT_TOKEN="env-token")
        assert dc.resolve_bot_token(cfg, {}, {}) == "env-token"

    def test_notify_config_token_used_next(self, tmp_path):
        cfg = make_cfg(tmp_path, DOGWATCH_BOT_TOKEN="")
        assert dc.resolve_bot_token(cfg, {}, {"botToken": "cfg-token"}) == "cfg-token"

    def test_secrets_dogwatch_account_preferred_over_default(self, tmp_path):
        cfg = make_cfg(tmp_path, DOGWATCH_BOT_TOKEN="")
        secrets = {"channels": {"telegram": {"accounts": {
            "dogwatch": {"botToken": "dw"},
            "default": {"botToken": "def"},
        }}}}
        assert dc.resolve_bot_token(cfg, secrets, {}) == "dw"

    def test_missing_everything_returns_empty_not_keyerror(self, tmp_path):
        cfg = make_cfg(tmp_path, DOGWATCH_BOT_TOKEN="")
        # The old code did accounts['default']['botToken'] unguarded.
        assert dc.resolve_bot_token(cfg, {"channels": {}}, {}) == ""

    def test_provider_key_matches_endpoint(self):
        secrets = {"models": {"providers": {
            "openrouter": {"apiKey": "or-key"},
            "google": {"apiKey": "g-key"},
            "deepseek": {"apiKey": "ds-key"},
        }}}
        assert dc.resolve_provider_key(
            "https://openrouter.ai/api/v1/chat/completions", secrets) == "or-key"
        assert dc.resolve_provider_key(
            "https://generativelanguage.googleapis.com/v1beta/x", secrets) == "g-key"
        assert dc.resolve_provider_key(
            "https://api.deepseek.com/chat/completions", secrets) == "ds-key"

    def test_provider_key_missing_returns_empty(self):
        assert dc.resolve_provider_key("https://openrouter.ai/x", {}) == ""


class TestLoadSecrets:
    def test_missing_file_is_not_fatal(self, tmp_path):
        # Regression: a missing secrets file used to sys.exit(1) even when
        # every credential was supplied by environment variable.
        assert dc.load_secrets(str(tmp_path / "nope.json")) == {}

    def test_malformed_json_is_not_fatal(self, tmp_path):
        p = tmp_path / "secrets.json"
        p.write_text("{broken")
        assert dc.load_secrets(str(p)) == {}

    def test_non_dict_json_is_not_fatal(self, tmp_path):
        p = tmp_path / "secrets.json"
        p.write_text("[1,2,3]")
        assert dc.load_secrets(str(p)) == {}


class TestVisionParsing:
    def test_strict_json_parsed(self):
        dog, dig, desc = dc._parse_vision_content(
            '{"dog":"DOG","digging":"YES","description":"2 dogs digging"}', "p")
        assert (dog, dig, desc) == ("DOG", True, "2 dogs digging")

    def test_digging_no_is_false_not_none(self):
        _, dig, _ = dc._parse_vision_content(
            '{"dog":"DOG","digging":"NO","description":""}', "p")
        assert dig is False

    def test_digging_uncertain_is_none(self):
        _, dig, _ = dc._parse_vision_content(
            '{"dog":"DOG","digging":"UNCERTAIN"}', "p")
        assert dig is None

    def test_non_json_falls_back_to_keyword_scan(self):
        dog, dig, _ = dc._parse_vision_content(
            'Sorry! NO_DOG here, digging: NO', "p")
        assert dog == "NO_DOG"
        assert dig is False

    def test_unknown_verdict_coerced_to_uncertain(self):
        dog, _, _ = dc._parse_vision_content('{"dog":"MAYBE"}', "p")
        assert dog == "UNCERTAIN"


class TestVisionFallbackGuard:
    def test_identical_endpoint_and_key_skips_pointless_fallback(self, tmp_path, capsys):
        """The default config resolves both providers to the same OpenRouter
        key, so an account-level 429 rejects the 'fallback' identically."""
        cfg = make_cfg(tmp_path)
        cfg.vision_url = cfg.fallback_url = "https://openrouter.ai/api/v1/chat/completions"
        cfg.vision_key = cfg.fallback_key = "same-key"
        calls = []

        def fake_verify(*a, **k):
            calls.append(a[4])  # provider_label
            return None

        dc.vision_verify_with, orig = fake_verify, dc.vision_verify_with
        try:
            verify = dc.make_vision_verifier(cfg)
            assert verify("/tmp/x.jpg") is None
        finally:
            dc.vision_verify_with = orig
        assert calls == ["primary"]
        assert "same endpoint+key" in capsys.readouterr().err

    def test_third_tier_used_when_first_two_fail(self, tmp_path):
        """primary and fallback fail -> fallback2 (DeepSeek) is tried."""
        cfg = make_cfg(tmp_path)
        cfg.vision_url = "https://openrouter.ai/api/v1/chat/completions"
        cfg.vision_key = "or-key"
        cfg.fallback_url = "https://generativelanguage.googleapis.com/v1beta/x"
        cfg.fallback_key = "g-key"
        cfg.fallback2_url = "https://api.deepseek.com/chat/completions"
        cfg.fallback2_key = "ds-key"
        calls = []

        def fake_verify(image_path, api_url, model, api_key, label, **k):
            calls.append(label)
            return None if label != "fallback2" else {"dog": "DOG", "digging": False}

        dc.vision_verify_with, orig = fake_verify, dc.vision_verify_with
        try:
            verify = dc.make_vision_verifier(cfg)
            result = verify("/tmp/x.jpg")
        finally:
            dc.vision_verify_with = orig
        assert calls == ["primary", "fallback", "fallback2"]
        assert result["dog"] == "DOG"

    def test_fallback2_skipped_when_duplicate_of_primary(self, tmp_path, capsys):
        """fallback2 pointing at the same endpoint+key as primary is skipped."""
        cfg = make_cfg(tmp_path)
        cfg.vision_url = cfg.fallback2_url = "https://openrouter.ai/api/v1/chat/completions"
        cfg.vision_key = cfg.fallback2_key = "same-key"
        calls = []

        def fake_verify(*a, **k):
            calls.append(a[4])
            return None

        dc.vision_verify_with, orig = fake_verify, dc.vision_verify_with
        try:
            verify = dc.make_vision_verifier(cfg)
            assert verify("/tmp/x.jpg") is None
        finally:
            dc.vision_verify_with = orig
        assert calls == ["primary"]
        assert "same endpoint+key" in capsys.readouterr().err

    def test_different_provider_does_fall_back(self, tmp_path):
        cfg = make_cfg(tmp_path)
        cfg.vision_url = "https://openrouter.ai/api/v1/chat/completions"
        cfg.vision_key = "or-key"
        cfg.fallback_url = "https://generativelanguage.googleapis.com/v1beta/x"
        cfg.fallback_key = "g-key"
        calls = []

        def fake_verify(image_path, api_url, model, api_key, label, **k):
            calls.append(label)
            return None if label == "primary" else {"dog": "DOG", "digging": False}

        dc.vision_verify_with, orig = fake_verify, dc.vision_verify_with
        try:
            verify = dc.make_vision_verifier(cfg)
            result = verify("/tmp/x.jpg")
        finally:
            dc.vision_verify_with = orig
        assert calls == ["primary", "fallback"]
        assert result["dog"] == "DOG"
