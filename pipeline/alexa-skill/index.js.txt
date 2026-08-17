// Dog Finder — Alexa custom skill backend (Alexa-hosted, Node.js)
//
// Purpose: the ONE Alexa path that knows WHICH Echo invoked it. The skill
// receives the invoking device's deviceId (only custom skills get this —
// Smart Home skills and routines don't), POSTs it to the Home Assistant
// webhook, and speaks an immediate ack. HA then runs the find-dogs scan
// and announces the result ONLY on that Echo (via the existing
// device-suffixed MQTT flow + invoking-echo automations).
//
// Intents:
//   FindDogsIntent    — "where are the dogs" -> full scan of all cameras
//   CheckCameraIntent — "check for the dogs at the back gate" -> scan ONLY
//                       that camera (slot CAMERA, resolved to a channel id)
//
// Setup: create a Custom / Alexa-hosted (Node.js) skill named e.g.
// "Dog Finder", set the invocation name (interaction model in
// interaction-model.json), then paste this file into the Code tab as
// index.js. Replace HA_WEBHOOK_URL with your instance's URL (see SETUP.md).
//
// Flow:
//   "Alexa, ask dog finder where are the dogs"
//     -> this handler -> POST {"deviceId": "<invoking echo id>"} to HA
//     -> speak ack ("On it, checking the yard cameras for the dogs.")
//     -> HA: webhook automation maps deviceId -> device key, publishes
//        dogwatch/find-dogs/trigger/<device> (payload "silent" so the
//        notifier does NOT re-announce the ack)
//     -> notifier scans, publishes result/<device>
//     -> HA automation announces the result on notify.<device>_speak
//        (the invoking Echo only)
//
//   "Alexa, ask dog finder to check for the dogs at the back gate"
//     -> same, but POST {"deviceId": ..., "channel": 14} -> HA publishes
//        {"ack": false, "channel": 14} -> notifier scans ONLY ch14

'use strict';

const https = require('https');

// Replace the host with your Nabu Casa remote URL (Settings -> Home
// Assistant Cloud shows it; it's the same slug as your remote access URL).
const HA_WEBHOOK_URL =
  'https://YOUR-INSTANCE.ui.nabu.casa/api/webhook/dogwatch_find_dogs_skill';

// Camera slot id (from the CAMERA slot type) -> NVR channel id.
const SLOT_ID_TO_CHANNEL = {
  back_gate: 14,
  back_door: 8,
  rear_fence: 10,
  clothes_line: 12,
  side_run: 1,
  carport: 2,
  front_driveway: 3,
  rear_shed: 4,
  pool: 11,
  front_door: 13,
  toy_area: 15,
  william_st: 16,
};

// Fallback: free-text camera name -> channel id (matches config
// channel_names / voice_names + common aliases).
const NAME_TO_CHANNEL = {
  'back gate': 14, 'rear gate': 14, gate: 14,
  'back door': 8, 'rear west': 8, 'rear west wide': 8,
  'rear fence': 10, 'garage rear': 10, fence: 10,
  'clothes line': 12, clothesline: 12, 'rear east': 12,
  'side run': 1, 'side east': 1, 'side yard': 1, side: 1,
  carport: 2,
  'front driveway': 3, driveway: 3,
  'rear shed': 4, shed: 4,
  pool: 11, 'pool area': 11,
  'front door': 13,
  'toy area': 15,
  'william street': 16, 'william st': 16,
};

const ACK_LINES = [
  'On it, checking the yard cameras for the dogs.',
  'On it, scanning the yard for the dogs now.',
  'Sure, checking the cameras for the dogs.',
  'On it, looking for the dogs in the yard.',
];

const CAMERA_ACK_LINES = [
  'On it, checking the {camera} for the dogs.',
  'On it, scanning the {camera} now.',
  'Sure, checking the {camera} for the dogs.',
  'Looking at the {camera} for the dogs now.',
];

const HELP_LINE =
  'You can ask me where the dogs are, and I will scan the yard cameras ' +
  'and tell you which room or area they are in. You can also ask me to ' +
  'check a specific spot, like the back gate or the back door.';

function postJson(body) {
  return new Promise((resolve) => {
    const payload = JSON.stringify(body);
    let settled = false;
    const done = () => {
      if (!settled) { settled = true; resolve(); }
    };
    try {
      const req = https.request(HA_WEBHOOK_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(payload),
        },
        timeout: 6000,
      }, (res) => {
        res.resume(); // drain
        done();
      });
      req.on('timeout', () => { req.destroy(); done(); });
      req.on('error', done);
      req.write(payload);
      req.end();
    } catch (e) {
      done();
    }
  });
}

function speak(ssml) {
  return {
    version: '1.0',
    response: {
      outputSpeech: { type: 'SSML', ssml: `<speak>${ssml}</speak>` },
      shouldEndSession: true,
    },
  };
}

function pick(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

function getDeviceId(event) {
  return event.context && event.context.System && event.context.System.device
    ? event.context.System.device.deviceId
    : '';
}

function resolveChannel(intent) {
  const slot = intent && intent.slots && intent.slots.camera;
  if (!slot) return null;
  // Prefer the slot-type id when Alexa matched a known value.
  const resolutions =
    slot.resolutions && slot.resolutions.resolutionsPerAuthority;
  if (resolutions && resolutions[0] && resolutions[0].values &&
      resolutions[0].values[0] && resolutions[0].values[0].value &&
      resolutions[0].values[0].value.id) {
    const id = resolutions[0].values[0].value.id;
    if (SLOT_ID_TO_CHANNEL[id]) return SLOT_ID_TO_CHANNEL[id];
  }
  // Fallback: match the spoken text against the name map.
  const spoken = String(slot.value || '').toLowerCase().replace(/^the /, '')
    .trim();
  return NAME_TO_CHANNEL[spoken] || null;
}

exports.handler = async (event) => {
  const reqType = event.request && event.request.type;
  const intentName =
    reqType === 'IntentRequest' && event.request.intent
      ? event.request.intent.name
      : '';

  // Stop / Cancel / Help — no scan.
  if (intentName === 'AMAZON.StopIntent' || intentName === 'AMAZON.CancelIntent') {
    return speak('Okay.');
  }
  if (intentName === 'AMAZON.HelpIntent') {
    return speak(HELP_LINE);
  }

  const deviceId = getDeviceId(event);

  if (intentName === 'CheckCameraIntent') {
    const channel = resolveChannel(event.request.intent);
    const body = { deviceId };
    if (channel) {
      body.channel = channel;
    }
    if (deviceId) {
      await postJson(body); // fire-and-forget; respond to Alexa immediately
    }
    if (channel) {
      const spoken = String(
        event.request.intent.slots.camera.value || '').trim();
      const line = pick(CAMERA_ACK_LINES).replace('{camera}', spoken);
      return speak(line);
    }
    return speak(
      'I could not tell which spot that is. Checking the whole yard ' +
      'for the dogs.');
  }

  // LaunchRequest ("open dog finder") or FindDogsIntent — full scan.
  if (deviceId) {
    await postJson({ deviceId });
  }
  return speak(pick(ACK_LINES));
};
