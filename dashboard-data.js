// functions/dashboard-data.js
// Returns all mechanic data from Twilio Sync as JSON for the dashboard

const TIERS = [
  { name: 'Bronze',  min: 0,    max: 499      },
  { name: 'Silver',  min: 500,  max: 1499     },
  { name: 'Gold',    min: 1500, max: 2999     },
  { name: 'Diamond', min: 3000, max: Infinity },
];
function getTier(pts) {
  return (TIERS.find(t => pts >= t.min && pts <= t.max) || TIERS[0]).name;
}

exports.handler = async function (context, event, callback) {
  const response = new Twilio.Response();
  response.appendHeader('Access-Control-Allow-Origin', '*');
  response.appendHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  response.appendHeader('Access-Control-Allow-Headers', 'Content-Type');
  response.appendHeader('Content-Type', 'application/json');

  // Handle preflight
  if (event.request && event.request.method === 'OPTIONS') {
    response.setStatusCode(204);
    response.setBody('');
    return callback(null, response);
  }

  try {
    const client = context.getTwilioClient();
    const docs   = await client.sync.v1.services('default').documents.list({ limit: 1000 });

    const mechanics = docs
      .filter(d => d.data && d.data.name)
      .map(d => ({
        phone:            d.uniqueName,
        name:             d.data.name             || 'Unknown',
        points:           d.data.points           || 0,
        tier:             getTier(d.data.points   || 0),
        images_submitted: d.data.images_submitted || 0,
        last_city:        d.data.last_city        || '',
        last_state:       d.data.last_state       || '',
        last_device:      d.data.last_device      || '',
        joined_at:        d.data.joined_at        || '',
      }));

    response.setBody(JSON.stringify({ ok: true, mechanics }));
  } catch (e) {
    response.setStatusCode(500);
    response.setBody(JSON.stringify({ ok: false, error: e.message }));
  }

  callback(null, response);
};
