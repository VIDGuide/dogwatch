"""static_suppressor.py — suppress recurring false positives from fixed scene objects.

The detection model consistently scores certain static structural elements
(beams, railings, pipes) at 0.5-0.7 "dog" confidence on every frame they appear
in. The motion gate blocks most of these (nothing moved → no inference), but any
real pixel change elsewhere in the scene (shadow shift, compression jitter,
wind) opens the gate, and the model immediately re-scores the same static spot.

This module detects that pattern: if the same bbox region fires repeatedly at a
similar position without ever having "arrived" (no significant spatial movement
between detections), it's a static object — suppress it.

## Why `digging` is never suppressed

The original design suppressed *any* event type once a region had fired
`max_hits` times at the same position. That is wrong for this application,
because of a stark asymmetry in the cost of being wrong:

  * Wrongly suppressing a `dog_at_fence` false positive → you don't get a
    Telegram ping you didn't want. That's the whole point.
  * Wrongly suppressing a real `digging` event → the siren never sounds, nobody
    is told, and the dogs keep digging under the fence. That is the exact
    outcome this entire project exists to prevent.

And suppression was reachable for a genuinely stationary dog: events are rate
limited to one per `event_cooldown_seconds` (30s) per track, so a dog that walks
to the fence and digs *in one spot* accumulates `max_hits: 3` in about 90
seconds and is then classified as a structural element. `dig_stationary_px`
explicitly permits a digging dog to drift up to 50px, so `record_movement`'s
20px-between-frames trigger was not a reliable rescue.

So event types listed in `static_suppression_protected_events` (default:
`digging`) are never suppressed. Better still, a protected event is treated as
positive evidence that the region is *not* structural, and clears the region's
accumulated hits — a beam does not dig.

## Why movement grants a grace period

`record_movement` used to delete the matching region outright. That reset
`hit_count` to zero, so a dog that arrived and then held still simply
re-accumulated three hits and got suppressed again ~90s later. Movement now
stamps the region with `last_movement_ts` and suppression is withheld while that
stamp is fresh (`static_suppression_movement_grace_seconds`), so "something
demonstrably moved here recently" keeps protecting the spot.

## Why suppression is logged

It previously returned True silently, so a suppressed real event was invisible:
no log line, no counter, nothing to correlate against "why didn't I get an
alert?". Every first-time suppression of a region now logs, repeats are rate
limited, and `stats()` exposes running totals.

Config keys (per-camera, all optional):
    "static_suppression_enabled": true,       # default true
    "static_suppression_iou_threshold": 0.7,  # bbox overlap to consider "same spot"
    "static_suppression_max_hits": 3,         # consecutive same-spot hits before suppressing
    "static_suppression_decay_seconds": 300,  # forget a suppressed region after this long
                                              # without a detection (handles lighting changes
                                              # that shift the false-positive spot)
    "static_suppression_protected_events": ["digging"],   # never suppressed
    "static_suppression_movement_grace_seconds": null,    # defaults to decay_seconds
"""
import time

#: Event types that must never be suppressed. See the module docstring — the
#: cost of dropping a real digging event is categorically worse than the cost of
#: letting a false one through to vision verification.
DEFAULT_PROTECTED_EVENTS = ("digging",)


def _iou(box_a, box_b):
    """Compute Intersection over Union between two boxes [x0, y0, x1, y1]."""
    x0 = max(box_a[0], box_b[0])
    y0 = max(box_a[1], box_b[1])
    x1 = min(box_a[2], box_b[2])
    y1 = min(box_a[3], box_b[3])

    inter = max(0, x1 - x0) * max(0, y1 - y0)
    if inter == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class _SuppressedRegion:
    """Tracks a single region that has triggered repeatedly without movement."""

    __slots__ = ("bbox", "hit_count", "last_hit_ts", "suppressed",
                 "last_movement_ts", "suppress_log_ts", "suppressed_count")

    def __init__(self, bbox, ts):
        self.bbox = list(bbox)
        self.hit_count = 1
        self.last_hit_ts = ts
        self.suppressed = False
        self.last_movement_ts = None
        self.suppress_log_ts = 0.0
        self.suppressed_count = 0

    def matches(self, bbox, iou_threshold):
        return _iou(self.bbox, bbox) >= iou_threshold

    def record_hit(self, bbox, ts):
        # Update bbox to a running average (smooths jitter)
        for i in range(4):
            self.bbox[i] = 0.7 * self.bbox[i] + 0.3 * bbox[i]
        self.hit_count += 1
        self.last_hit_ts = ts

    def reset_hits(self, ts):
        """Positive evidence of a real animal here — start counting again."""
        self.hit_count = 0
        self.suppressed = False
        self.last_movement_ts = ts

    def is_expired(self, now, decay_seconds):
        return (now - self.last_hit_ts) > decay_seconds

    def recently_moved(self, now, grace_seconds):
        if self.last_movement_ts is None:
            return False
        return (now - self.last_movement_ts) <= grace_seconds


class StaticSuppressor:
    """Suppress detections from static scene regions that fire repeatedly."""

    def __init__(self, cfg, camera_name="camera", log=print):
        self.enabled = cfg.get("static_suppression_enabled", True)
        self.iou_threshold = cfg.get("static_suppression_iou_threshold", 0.7)
        self.max_hits = cfg.get("static_suppression_max_hits", 3)
        self.decay_seconds = cfg.get("static_suppression_decay_seconds", 300.0)
        protected = cfg.get("static_suppression_protected_events",
                            DEFAULT_PROTECTED_EVENTS)
        self.protected_events = frozenset(protected or ())
        grace = cfg.get("static_suppression_movement_grace_seconds")
        self.movement_grace_seconds = (
            self.decay_seconds if grace is None else float(grace))
        self.camera_name = camera_name
        self._log = log
        self._regions = []  # list of _SuppressedRegion

        # Running totals, surfaced via stats() — suppression used to be
        # completely unobservable.
        self._total_suppressed = 0
        self._total_protected = 0
        self._total_movement_grace = 0

    def should_suppress(self, bbox, ts, event_type=None):
        """Return True if this bbox should be suppressed as a static false positive.

        Call this for each detection that would otherwise fire an event, passing
        the event type so protected types (``digging``) are never dropped.
        """
        if not self.enabled:
            return False

        # Expire old regions
        self._regions = [
            r for r in self._regions if not r.is_expired(ts, self.decay_seconds)
        ]

        region = None
        for candidate in self._regions:
            if candidate.matches(bbox, self.iou_threshold):
                region = candidate
                break

        # Protected event types are never suppressed, and are treated as
        # evidence that this region is not a structural false positive.
        if event_type is not None and event_type in self.protected_events:
            self._total_protected += 1
            if region is not None:
                was_suppressed = region.suppressed
                region.record_hit(bbox, ts)
                region.reset_hits(ts)
                if was_suppressed:
                    self._log(
                        f"[{self.camera_name}] static_suppressor: '{event_type}' at "
                        f"{[int(v) for v in bbox]} CLEARS a previously suppressed "
                        f"region — a structural element does not dig")
            else:
                new_region = _SuppressedRegion(bbox, ts)
                new_region.last_movement_ts = ts
                self._regions.append(new_region)
            return False

        if region is None:
            # New region — start tracking
            self._regions.append(_SuppressedRegion(bbox, ts))
            return False

        region.record_hit(bbox, ts)

        # Something demonstrably moved here recently — withhold suppression.
        if region.recently_moved(ts, self.movement_grace_seconds):
            self._total_movement_grace += 1
            return False

        if region.hit_count >= self.max_hits:
            region.suppressed = True
            region.suppressed_count += 1
            self._total_suppressed += 1
            self._maybe_log_suppression(region, bbox, ts, event_type)
            return True
        return False

    def _maybe_log_suppression(self, region, bbox, ts, event_type):
        """Log the first suppression per region, then at most every 5 minutes.

        Suppression was previously silent, which made a suppressed real event
        indistinguishable from "nothing happened".
        """
        if region.suppress_log_ts and (ts - region.suppress_log_ts) < 300:
            return
        region.suppress_log_ts = ts
        label = event_type or "event"
        self._log(
            f"[{self.camera_name}] static_suppressor: suppressing '{label}' at "
            f"{[int(v) for v in bbox]} — {region.hit_count} hits at the same spot "
            f"with no movement (region total {region.suppressed_count}). "
            f"If this is a real dog, lower static_suppression_max_hits' "
            f"sensitivity or add the event type to "
            f"static_suppression_protected_events.")

    def record_movement(self, bbox, ts):
        """Call when a tracked object moves significantly.

        Stamps every overlapping region as "recently moved" rather than deleting
        it. Deleting merely reset the hit count, so a dog that arrived and then
        held still re-accumulated max_hits and was suppressed again a couple of
        minutes later; the stamp keeps protecting the spot for
        movement_grace_seconds.
        """
        if not self.enabled:
            return
        # A looser IoU than the same-spot threshold: the moving box only needs
        # to overlap the region it is arriving at.
        loose = self.iou_threshold * 0.5
        for region in self._regions:
            if region.matches(bbox, loose):
                region.reset_hits(ts)

    @property
    def suppressed_count(self):
        """Number of currently suppressed regions."""
        return sum(1 for r in self._regions if r.suppressed)

    def stats(self):
        """Diagnostics: cumulative counters plus current region state."""
        return {
            "regions_tracked": len(self._regions),
            "regions_suppressed": self.suppressed_count,
            "events_suppressed": self._total_suppressed,
            "events_protected": self._total_protected,
            "events_movement_grace": self._total_movement_grace,
        }
