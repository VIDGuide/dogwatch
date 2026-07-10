# Home Assistant example

This is a reference for the Home Assistant side of DogWatch: the entities the
pipeline creates automatically, the small amount of optional manual YAML, and
the Lovelace dashboard cards (boolean status tiles + camera snapshots) taken
from a working "Single AMS" dashboard.

Two cameras are assumed, matching the shipped configs:

| Camera | `mqtt_base_topic` | Snapshot topic |
|--------|-------------------|----------------|
| Fence (`camera`) | `dogwatch` | `dogwatch/snapshot` |
| Rear-East (`rear-east`) | `dogwatch/rear-east` | `dogwatch/rear-east/snapshot` |

---

## 1. Auto-discovered entities (no YAML needed)

`mqtt_publisher.py` publishes MQTT discovery configs on startup/reconnect, so
these appear automatically under two devices (`Dogwatch camera`,
`Dogwatch rear-east`) as long as MQTT discovery is enabled in HA
(it is by default):

| Entity | Type | From |
|--------|------|------|
| `binary_sensor.dogwatch_camera_camera_dog_at_fence` | `binary_sensor` (device_class `motion`, `off_delay: 180`) | discovery |
| `binary_sensor.dogwatch_camera_camera_dog_digging` | `binary_sensor` | discovery |
| `binary_sensor.dogwatch_rear_east_rear_east_dog_at_fence` | `binary_sensor` | discovery |
| `binary_sensor.dogwatch_rear_east_rear_east_dog_digging` | `binary_sensor` | discovery |
| `camera.fence_dogwatch` | `camera` (topic `dogwatch/snapshot`) | discovery |
| `camera.rear_east_dogwatch` | `camera` (topic `dogwatch/rear-east/snapshot`) | discovery |

> Entity IDs may get a numeric suffix (e.g. `camera.fence_dogwatch_2`) if a
> previous entity claimed the friendly name first — harmless, just reference
> whatever IDs your instance actually shows.

The `off_delay` on the binary sensors is what makes a triggered tile fall back
to *clear* automatically after 180 s even if the OFF message is lost — no
automation required.

---

## 2. Optional manual YAML — snapshot freshness sensors

The pipeline publishes a companion "last snapshot time" on
`<topic>/snapshot/ts` (epoch seconds). These aren't auto-discovered; add them
to `configuration.yaml` if you want to show *when* each snapshot last updated
(handy for spotting a frozen feed):

```yaml
mqtt:
  - sensor:
      - name: "Fence Snapshot TS"
        unique_id: dogwatch_fence_snapshot_ts
        state_topic: "dogwatch/snapshot/ts"
        value_template: "{{ value | float(0) | timestamp_local('%H:%M:%S') }}"
      - name: "Rear East Snapshot TS"
        unique_id: dogwatch_rear_east_snapshot_ts
        state_topic: "dogwatch/rear-east/snapshot/ts"
        value_template: "{{ value | float(0) | timestamp_local('%H:%M:%S') }}"
```

---

## 3. Lovelace dashboard cards

A grid layout with, per camera: an "at fence" tile, a "digging" tile, and the
live annotated snapshot. The status tiles use
[`card-mod`](https://github.com/thomasloven/lovelace-card-mod) (install via
HACS) to pulse red while triggered.

### Status tile (boolean) — pulses red when ON

```yaml
type: entity
entity: binary_sensor.dogwatch_camera_camera_dog_at_fence
name: "Fence: At Fence"
icon: mdi:shield-alert
card_mod:
  style: |
    @keyframes pulse-red {
      0%   { background-color: #c62828; box-shadow: 0 0 8px #c62828; }
      50%  { background-color: #ff1744; box-shadow: 0 0 20px #ff1744; }
      100% { background-color: #c62828; box-shadow: 0 0 8px #c62828; }
    }
    .state { font-size: 1em !important; font-weight: 600 !important; }
    ha-state-icon { --mdc-icon-size: 22px !important; }
    ha-card {
      min-height: 158px !important;
      display: flex !important;
      flex-direction: column !important;
      align-items: center !important;
      justify-content: center !important;
      {% if is_state(config.entity, 'on') %}
        animation: pulse-red 1s ease-in-out infinite !important;
        border: 2px solid #ff5252 !important;
      {% endif %}
    }
    ha-card .state, ha-card .card-header, ha-card .name, ha-card * {
      {% if is_state(config.entity, 'on') %}
        color: white !important;
      {% endif %}
    }
    ha-card ha-state-icon {
      {% if is_state(config.entity, 'on') %}
        --state-icon-color: #ff5252 !important;
        color: #ff5252 !important;
      {% endif %}
    }
```

> `config.entity` lets you reuse the exact same `card_mod` block for all four
> tiles — just change the top-level `entity:` and `name:`. (The original
> dashboard hard-coded the entity id inside the `{% if %}`; `config.entity`
> is the DRY equivalent.)

Repeat with:
- `binary_sensor.dogwatch_camera_camera_dog_digging` — name `Fence: Digging`
- `binary_sensor.dogwatch_rear_east_rear_east_dog_at_fence` — name `Rear-East: At Fence`
- `binary_sensor.dogwatch_rear_east_rear_east_dog_digging` — name `Rear-East: Digging`

### Camera snapshot tile

```yaml
type: picture-entity
entity: camera.fence_dogwatch      # or camera.rear_east_dogwatch
show_state: false
show_name: false
camera_view: auto
card_mod:
  style: |
    ha-card { max-height: 160px !important; min-height: 160px !important; }
    img {
      max-height: 140px !important;
      object-fit: contain !important;
      background: #1a1a1a !important;
    }
```

### Full section (both cameras)

```yaml
type: sections
sections:
  - type: grid
    cards:
      - type: heading
        heading: DogWatch
      - type: grid
        columns: 3
        cards:
          # --- Fence ---
          - type: entity
            entity: binary_sensor.dogwatch_camera_camera_dog_at_fence
            name: "Fence: At Fence"
            icon: mdi:shield-alert
            # (card_mod style as above)
          - type: entity
            entity: binary_sensor.dogwatch_camera_camera_dog_digging
            name: "Fence: Digging"
            icon: mdi:shield-alert
            # (card_mod style as above)
          - type: picture-entity
            entity: camera.fence_dogwatch
            show_state: false
            show_name: false
            camera_view: auto
          # --- Rear-East ---
          - type: entity
            entity: binary_sensor.dogwatch_rear_east_rear_east_dog_at_fence
            name: "Rear-East: At Fence"
            icon: mdi:shield-alert
            # (card_mod style as above)
          - type: entity
            entity: binary_sensor.dogwatch_rear_east_rear_east_dog_digging
            name: "Rear-East: Digging"
            icon: mdi:shield-alert
            # (card_mod style as above)
          - type: picture-entity
            entity: camera.rear_east_dogwatch
            show_state: false
            show_name: false
            camera_view: auto
```

---

## Notes

- **card-mod** is the only extra dependency (HACS → Frontend → card-mod).
  Drop the `card_mod:` blocks and the tiles still work, just without the red
  pulse.
- Snapshots are **retained** MQTT images, so a card shows the last frame
  immediately on load; it refreshes on each event and on the 60 s live-still
  tick.
- If a `camera.*` entity is missing, check that MQTT discovery is enabled and
  that the detector logged `MQTT connected` for that camera.
