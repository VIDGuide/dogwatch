# Dog Watch — Alexa custom skill (setup guide)

The one Alexa path that knows **which Echo** invoked it. Custom skills receive
the invoking device's `deviceId` (routines and the HA Cloud Smart Home skill
don't), so we can announce the find-dogs result on exactly that Echo.

## Intents

- **FindDogsIntent** — "where are the dogs" → full scan (preferred camera
  order, early exit when found).
- **CheckCameraIntent** — "check for the dogs at the back gate" → scans ONLY
  that camera (slot `camera`, type `CAMERA`). The skill resolves the name to
  an NVR channel and sends `{"deviceId": ..., "channel": N}` to the webhook;
  HA publishes `{"ack": false, "channel": N}` on the trigger topic and the
  notifier scans just that channel. Unknown camera → full-scan fallback.

## Architecture

```
"Alexa, ask dog watch where are the dogs"
   -> Echo hears it -> Alexa runs the "Dog Watch" skill
   -> skill backend (this folder) reads event.context.System.device.deviceId
   -> POSTs {"deviceId": "..."} to HA Cloud webhook (fire-and-forget)
   -> speaks an immediate ack ("On it, checking the yard cameras...")
   -> HA automation find_the_dogs_skill_trigger maps deviceId -> device key
   -> publishes dogwatch/find-dogs/trigger/<device> (payload "silent",
      so the notifier does NOT re-announce the ack)
   -> notifier runs the scan, publishes dogwatch/find-dogs/result/<device>
   -> HA automation find_the_dogs_announce_result_invoking_echo announces
      the result on notify.<device>_speak (that Echo only)
```

## Steps (Alexa side, ~15 min)

1. **Create the developer account** — https://developer.amazon.com
   (sign in with the Amazon account that owns the Echos; it's free).
2. **Alexa Developer Console** — https://developer.amazon.com/alexa/console/ask
   → **Create Skill**.
3. Skill name: `Dog Watch`. **Type: Custom**. Hosting: **Alexa-hosted
   (Node.js)**. Template: **Hello World** (it gives you a working index.js to
   replace). → Create skill.
4. **Build tab → Interaction Model → JSON Editor**: paste the contents of
   `interaction-model.json` (sets the invocation name to "dog watch" + the
   find-dogs phrases). → **Save Model** → **Build Model**.
5. **Code tab**: replace `index.js` with the contents of `index.js` from this
   folder. Set the `HA_WEBHOOK_URL` constant (below). → **Deploy**.
6. **Test** — on any Echo: *"Alexa, ask dog watch where are the dogs"*.
   The invoking Echo should say the ack, then ~30 s later the result, and no
   OTHER Echo should speak.

## The webhook URL

`https://YOUR-INSTANCE.ui.nabu.casa/api/webhook/dogwatch_find_dogs_skill`

- `YOUR-INSTANCE.ui.nabu.casa` = the HA Cloud remote URL (HA → Settings →
  Home Assistant Cloud → Remote URL; it's the same slug used for remote access).
- `dogwatch_find_dogs_skill` = the webhook id registered by the HA automation
  `find_dogs_skill_trigger` (automations.yaml). The automation maps each
  deviceId to its Echo via the `alexa_devices` device registry ids
  (entity_registry `notify.*_speak` device_id fields).

## Hardening the webhook (recommended)

An HA webhook has no authentication of its own — the URL *is* the credential.
And it is a credential that leaks like a URL: it sits in `index.js`, in
Alexa's build output, and in your automation. Anyone who gets it can trigger a
full camera scan at will, which spends vision-API quota and makes your Echos
talk.

Two optional constants in `index.js` close that off. Both default to empty,
which keeps the original behaviour, so this is opt-in:

1. **`WEBHOOK_SHARED_SECRET`** — sent as an `X-Dogwatch-Token` header. Generate
   one (`openssl rand -hex 32`), set it in `index.js`, and add a matching
   condition to `find_dogs_skill_trigger` so unmatched requests are dropped:

   ```yaml
   automation:
     - id: find_dogs_skill_trigger
       trigger:
         - platform: webhook
           webhook_id: dogwatch_find_dogs_skill
           local_only: false
           allowed_methods: [POST]
       condition:
         - condition: template
           value_template: >-
             {{ trigger.headers.get('x-dogwatch-token') == 'PASTE_THE_SAME_SECRET' }}
       action: ...
   ```

   Put the secret in `secrets.yaml` and reference it with `!secret` rather than
   inlining it. Note HA lowercases header names in `trigger.headers`.

2. **`EXPECTED_APPLICATION_ID`** — your skill id from the developer console
   ("Your Skill ID", `amzn1.ask.skill.…`). The handler rejects any invocation
   carrying a different one.

   Alexa-hosted skills do **not** need the request-signature verification a
   self-hosted HTTPS endpoint requires: the Alexa service invokes the Lambda
   directly, so there is no HTTP request to verify. The application id check is
   the equivalent control at this layer.

## HA side (already deployed)

- `automation.find_dogs_skill_trigger` — webhook trigger, `local_only: false`
  (required! HA silently drops remote webhook requests when local_only is
  true — returns HTTP 200 without processing).
- Device map: all 12 Echos' deviceIds → device keys, unknown → base topic
  (default group announce).
- MQTT listener (`find-dogs-mqtt.py`) honours payload `"silent"` to skip the
  ack (the skill speaks it), and routes result to `result/<device>`.

## Gotchas

- Skill responses must return within ~8 s — the backend POSTs fire-and-forget
  and speaks the ack immediately; HA does the scan async.
- If an Echo's deviceId isn't in the automation map, the trigger falls back to
  the base topic → the default 3-Echo group announce.
- The skill speaks a canned (rotating) ack; the DeepSeek-varied ack in the
  notifier is suppressed for skill triggers to avoid double-announcing.
