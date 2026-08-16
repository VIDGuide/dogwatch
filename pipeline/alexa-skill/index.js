// Dog Finder — Alexa custom skill backend (Alexa-hosted, Node.js)
//
// Purpose: the ONE Alexa path that knows WHICH Echo invoked it. The skill
// receives the invoking device's deviceId (only custom skills get this —
// Smart Home skills and routines don't), POSTs it to the Home Assistant
// webhook, and speaks an immediate ack. HA then runs the find-dogs scan
// and announces the result ONLY on that Echo (via the existing
// device-suffixed MQTT flow + invoking-echo automations).
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

'use strict';

const https = require('https');

// Replace the host with your Nabu Casa remote URL (Settings -> Home
// Assistant Cloud shows it; it's the same slug as your remote access URL).
const HA_WEBHOOK_URL =
  'https://YOUR-INSTANCE.ui.nabu.casa/api/webhook/dogwatch_find_dogs_skill';

const ACK_LINES = [
  'On it, checking the yard cameras for the dogs.',
  'On it, scanning the yard for the dogs now.',
  'Sure, checking the cameras for the dogs.',
  'On it, looking for the dogs in the yard.',
];

const HELP_LINE =
  'You can ask me where the dogs are, and I will scan the yard cameras ' +
  'and tell you which room or area they are in.';

function postDeviceId(deviceId) {
  return new Promise((resolve) => {
    const body = JSON.stringify({ deviceId });
    let settled = false;
    const done = () => {
      if (!settled) { settled = true; resolve(); }
    };
    try {
      const req = https.request(HA_WEBHOOK_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(body),
        },
        timeout: 6000,
      }, (res) => {
        res.resume(); // drain
        done();
      });
      req.on('timeout', () => { req.destroy(); done(); });
      req.on('error', done);
      req.write(body);
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

  // LaunchRequest ("open dog finder") or the FindDogs intent — trigger scan.
  const deviceId =
    event.context && event.context.System && event.context.System.device
      ? event.context.System.device.deviceId
      : '';
  if (deviceId) {
    // Fire-and-forget: respond to Alexa immediately, HA does the work.
    await postDeviceId(deviceId);
  }

  const ack = ACK_LINES[Math.floor(Math.random() * ACK_LINES.length)];
  return speak(ack);
};
