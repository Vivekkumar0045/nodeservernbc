// server.js — Local dashboard server for NBC Bearing
// Usage: node server.js
// Then open: http://localhost:3000

const http   = require('http');
const fs     = require('fs');
const path   = require('path');
const { execFile } = require('child_process');
const { GoogleGenerativeAI } = require('@google/generative-ai');

// ── Load .env ──────────────────────────────────────────────────────────────
try {
  const envFile = path.join(__dirname, '.env');
  if (fs.existsSync(envFile)) {
    fs.readFileSync(envFile, 'utf-8')
      .split(/\r?\n/)
      .forEach(line => {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith('#')) return;
        const eqIdx = trimmed.indexOf('=');
        if (eqIdx < 1) return;
        const key = trimmed.slice(0, eqIdx).trim();
        const val = trimmed.slice(eqIdx + 1).trim();
        process.env[key] = val;
      });
  }
} catch (_) {}

const PORT     = process.env.PORT || 3000;
const ROOT     = __dirname;
const CSV_PATH = path.join(ROOT, 'data', 'mechanics.csv');

const HEADERS = ['phone','name','points','tier','images_submitted',
                 'last_city','last_state','last_device','joined_at'];

// Parse CSV → array of objects
function parseCSV(text) {
  const lines = text.trim().split('\n');
  if (lines.length < 2) return [];
  const headers = lines[0].split(',').map(h => h.trim());
  return lines.slice(1).filter(l => l.trim()).map(line => {
    const cols = [];
    let cur = '', inQ = false;
    for (const ch of line) {
      if (ch === '"') { inQ = !inQ; }
      else if (ch === ',' && !inQ) { cols.push(cur.trim()); cur = ''; }
      else { cur += ch; }
    }
    cols.push(cur.trim());
    const obj = {};
    headers.forEach((h, i) => obj[h] = cols[i] ?? '');
    return {
      phone:            obj.phone || '',
      name:             obj.name  || '',
      points:           parseInt(obj.points) || 0,
      tier:             (obj.tier || 'Bronze').replace(/^[^\w]+/, '').trim(),
      images_submitted: parseInt(obj.images_submitted) || 0,
      last_city:        obj.last_city   || '',
      last_state:       obj.last_state  || '',
      last_device:      obj.last_device || '',
      joined_at:        obj.joined_at   || '',
    };
  });
}

// Serve static file
function serveFile(res, filePath, contentType) {
  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404);
      res.end('Not found');
      return;
    }
    res.writeHead(200, { 'Content-Type': contentType });
    res.end(data);
  });
}

// JSON response helper
function json(res, status, obj) {
  res.writeHead(status, {
    'Content-Type': 'application/json'
  });
  res.end(JSON.stringify(obj));
}

// ── HuggingFace Inference fallback ─────────────────────────────────────────
const https = require('https');
const HF_MODELS = [
  'mistralai/Mistral-7B-Instruct-v0.3',
  'mistralai/Mixtral-8x7B-Instruct-v0.1',
  'microsoft/Phi-3-mini-4k-instruct',
];

function hfRequest(model, prompt) {
  return new Promise((resolve, reject) => {
    const token = process.env.HF_TOKEN;
    if (!token) return reject(new Error('HF_TOKEN not set'));
    const body = JSON.stringify({
      inputs: prompt,
      parameters: { max_new_tokens: 4096, temperature: 0.7, return_full_text: false },
    });
    const options = {
      hostname: 'api-inference.huggingface.co',
      path: `/models/${model}`,
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(body),
      },
    };
    const req = https.request(options, r => {
      let d = '';
      r.on('data', c => d += c);
      r.on('end', () => {
        try {
          const parsed = JSON.parse(d);
          if (parsed.error) return reject(new Error(parsed.error));
          // HF returns [{generated_text: '...'}] or {generated_text: '...'}
          const text = Array.isArray(parsed)
            ? (parsed[0]?.generated_text || '')
            : (parsed.generated_text || JSON.stringify(parsed));
          resolve(text.trim());
        } catch (e) { reject(new Error('HF parse error: ' + d.slice(0, 100))); }
      });
    });
    req.on('error', reject);
    req.write(body);
    req.end();
  });
}

async function callHuggingFace(prompt, label) {
  for (const model of HF_MODELS) {
    try {
      console.log(`[${label}] Trying HuggingFace: ${model}`);
      const text = await hfRequest(model, prompt);
      console.log(`[${label}] ✅ HuggingFace used: ${model}`);
      return text;
    } catch (e) {
      console.log(`[${label}] HF ${model} failed: ${e.message.slice(0, 80)}`);
    }
  }
  throw new Error('All Gemini and HuggingFace models failed. Try again later.');
}

const server = http.createServer((req, res) => {
  const url = req.url.split('?')[0];

  // ── Proxy endpoints to hide external APIs
  if (req.method === 'POST' && url === '/api/predict/rul') {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', () => {
      const options = {
        hostname: 'vivek45537-nbc.hf.space',
        path: '/predict/rul',
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) }
      };
      const proxyReq = https.request(options, proxyRes => {
        res.writeHead(proxyRes.statusCode, proxyRes.headers);
        proxyRes.pipe(res);
      });
      proxyReq.on('error', e => json(res, 500, { error: e.message }));
      proxyReq.write(body);
      proxyReq.end();
    });
    return;
  }

  if (req.method === 'POST' && url === '/api/predict/dealer-segment') {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', () => {
      const options = {
        hostname: 'vivek45537-nbc.hf.space',
        path: '/predict/dealer-segment',
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) }
      };
      const proxyReq = https.request(options, proxyRes => {
        res.writeHead(proxyRes.statusCode, proxyRes.headers);  
        proxyRes.pipe(res);
      });
      proxyReq.on('error', e => json(res, 500, { error: e.message }));
      proxyReq.write(body);
      proxyReq.end();
    });
    return;
  }

  // ── GET / → dashboard.html
  if (req.method === 'GET' && (url === '/' || url === '/dashboard.html')) {
    return serveFile(res, path.join(ROOT, 'dashboard.html'), 'text/html');
  }

  // ── GET /chart.min.js
  if (req.method === 'GET' && url === '/chart.min.js') {
    return serveFile(res, path.join(ROOT, 'chart.min.js'), 'application/javascript');
  }

  // ── GET /data → parse CSV → return JSON
  if (req.method === 'GET' && url === '/data') {
    if (!fs.existsSync(CSV_PATH)) {
      return json(res, 200, { ok: true, mechanics: [], message: 'No CSV yet. Click Sync.' });
    }
    try {
      const text  = fs.readFileSync(CSV_PATH, 'utf-8');
      const mechanics = parseCSV(text).sort((a, b) => b.points - a.points);
      return json(res, 200, { ok: true, mechanics });
    } catch (e) {
      return json(res, 500, { ok: false, error: e.message });
    }
  }

  // ── POST /sync → run export.js → update CSV
  if (req.method === 'POST' && url === '/sync') {
    console.log('[sync] Running export.js…');
    const exportPath = path.join(ROOT, 'export.js');
    execFile(process.execPath, [exportPath], { cwd: ROOT }, (err, stdout, stderr) => {
      if (err) {
        console.error('[sync] Failed:', stderr || err.message);
        return json(res, 500, { ok: false, error: stderr || err.message });
      }
      console.log('[sync] Done:\n' + stdout);
      // Return updated CSV data immediately
      try {
        const text = fs.readFileSync(CSV_PATH, 'utf-8');
        const mechanics = parseCSV(text).sort((a, b) => b.points - a.points);
        return json(res, 200, { ok: true, mechanics, log: stdout.trim() });
      } catch (e) {
        return json(res, 200, { ok: true, mechanics: [], log: stdout.trim() });
      }
    });
    return;
  }

  // ── GET /nexus-data → bearings + dealer CSVs
  if (req.method === 'GET' && url === '/nexus-data') {
    try {
      const BEAR_PATH   = path.join(ROOT, 'data', 'bearings_data.csv');
      const DEALER_PATH = path.join(ROOT, 'data', 'dealer_network.csv');
      const parseSimpleCSV = (text) => {
        const lines = text.trim().split('\n');
        const headers = lines[0].split(',').map(h => h.trim());
        return lines.slice(1).filter(l => l.trim()).map(line => {
          const cols = line.split(',').map(c => c.trim());
          const obj = {};
          headers.forEach((h, i) => obj[h] = cols[i] ?? '');
          return obj;
        });
      };
      const bearings = fs.existsSync(BEAR_PATH)   ? parseSimpleCSV(fs.readFileSync(BEAR_PATH,   'utf-8')) : [];
      const dealers  = fs.existsSync(DEALER_PATH) ? parseSimpleCSV(fs.readFileSync(DEALER_PATH, 'utf-8')) : [];
      return json(res, 200, { ok: true, bearings, dealers });
    } catch (e) {
      return json(res, 500, { ok: false, error: e.message });
    }
  }

  // ── POST /nexus-analyze → Gemini AI fleet + dealer intelligence
  if (req.method === 'POST' && url === '/nexus-analyze') {
    (async () => {
      try {
        const GEMINI_KEY = process.env.GEMINI_API_KEY;
        if (!GEMINI_KEY) throw new Error('GEMINI_API_KEY not set');
        const BP = path.join(ROOT, 'data', 'bearings_data.csv');
        const DP = path.join(ROOT, 'data', 'dealer_network.csv');
        if (!fs.existsSync(BP)) throw new Error('bearings_data.csv not found in data/');
        const pcsv = t => {
          const lines = t.trim().split('\n'); const hdrs = lines[0].split(',').map(h => h.trim());
          return lines.slice(1).filter(l => l.trim()).map(line => {
            const cols = line.split(',').map(c => c.trim()); const o = {};
            hdrs.forEach((h, i) => o[h] = cols[i] ?? ''); return o;
          });
        };
        const bearings  = pcsv(fs.readFileSync(BP, 'utf-8'));
        const dealers   = fs.existsSync(DP) ? pcsv(fs.readFileSync(DP, 'utf-8')) : [];
        const total     = bearings.length;
        const healthy   = bearings.filter(b => b.Status === 'Healthy').length;
        const degrading = bearings.filter(b => b.Status === 'Degrading').length;
        const critical  = bearings.filter(b => b.Status === 'Critical').length;
        const avgRUL    = Math.round(bearings.reduce((s, b) => s + Number(b.RUL_Days || 0), 0) / total);
        const locMap    = {};
        bearings.forEach(b => {
          if (!locMap[b.Location]) locMap[b.Location] = { total: 0, critical: 0, ruls: [] };
          locMap[b.Location].total++;
          if (b.Status === 'Critical') locMap[b.Location].critical++;
          locMap[b.Location].ruls.push(Number(b.RUL_Days || 0));
        });
        const locSummary = Object.entries(locMap).map(([loc, s]) => {
          const avg = Math.round(s.ruls.reduce((a, r) => a + r, 0) / s.ruls.length);
          return `${loc}: ${s.total} units, ${s.critical} critical, avg RUL ${avg}d`;
        }).join('; ');
        const critUnits = bearings.filter(b => b.Status === 'Critical').slice(0, 10)
          .map(b => `${b.Unit_ID}@${b.Location} RUL=${b.RUL_Days}d T=${b.Temperature_C}C Vib=${b.Vibration_mm_s}`).join(', ');
        const dealerSummary = dealers.map(d =>
          `${d.Dealer_ID} ${d.Name}@${d.Location}: Inv=${d.Inventory_Level} Resp=${d.Service_Responsiveness_Score} TAT=${d.Turnaround_Time_Hrs}h CSI=${d.Customer_Satisfaction_Index}`
        ).join('\n');
        const prompt = [
          'You are an industrial IoT and maintenance intelligence expert for NBC Bearing Company.',
          'Analyze this real-time fleet + dealer data and return a comprehensive JSON report.',
          '',
          `BEARING FLEET (${total} units): Healthy=${healthy} Degrading=${degrading} Critical=${critical} AvgRUL=${avgRUL}d`,
          `LOCATIONS: ${locSummary}`,
          `CRITICAL UNITS (top 10): ${critUnits}`,
          '',
          `DEALER NETWORK (${dealers.length} dealers):`,
          dealerSummary,
          '',
          'STEP 1: Identify critical patterns in bearing failures (failure modes, temp, vibration, lubrication).',
          'STEP 2: Rank all locations by risk severity.',
          'STEP 3: Evaluate dealer performance gaps from given data.',
          'STEP 4: Build a prioritized maintenance schedule for the 5-8 most urgent units.',
          'STEP 5: Calculate an overall fleet health score 0-100.',
          'STEP 6: Write a 2-3 sentence executive summary for management.',
          '',
          'Respond ONLY with this valid JSON structure:',
          '{',
          '  "thinking": "brief chain-of-thought across all 6 steps",',
          '  "fleet_health_score": 75,',
          '  "bearing_insights": [{"icon":"🔴","title":"...","priority":"High","detail":"...","action":"..."}],',
          '  "maintenance_schedule": [{"unit_id":"...","location":"...","rul_days":0,"recommendation":"...","urgency":"Immediate"}],',
          '  "dealer_insights": [{"icon":"📊","title":"...","priority":"High","detail":"...","action":"..."}],',
          '  "location_risks": [{"location":"...","risk_level":"High","units":50,"critical_count":5,"recommendation":"..."}],',
          '  "exec_summary": "management summary here"',
          '}',
          'Rules: bearing_insights=4-6 items, maintenance_schedule=5-8 items, dealer_insights=3-5 items, location_risks=one per location, urgency=Immediate/Soon/Planned, priority=High/Medium/Low.',
        ].join('\n');

        const genAI  = new GoogleGenerativeAI(GEMINI_KEY);
        const MODELS = ['gemini-3-flash-preview','gemini-2.5-flash','gemini-2.0-flash'];
        let result;
        for (const modelName of MODELS) {
          try {
            console.log(`[nexus-analyze] Trying ${modelName}…`);
            const model = genAI.getGenerativeModel({
              model: modelName,
              generationConfig: { temperature: 1.0, topP: 0.95, maxOutputTokens: 32768, responseMimeType: 'application/json' },
            });
            result = await model.generateContent(prompt);
            console.log(`[nexus-analyze] ✅ ${modelName}`);
            break;
          } catch (e) { console.log(`[nexus-analyze] ${modelName} failed: ${e.message.slice(0, 60)}`); }
        }
        let analysis;
        if (!result) {
          const hfText = await callHuggingFace(prompt + '\n\nRespond with valid JSON only.', 'nexus-analyze');
          let c = hfText; const fb = c.indexOf('{'), lb = c.lastIndexOf('}');
          if (fb !== -1 && lb > fb) c = c.slice(fb, lb + 1);
          analysis = JSON.parse(c);
        } else {
          let c = result.response.text().trim().replace(/^```(?:json)?\s*/im, '').replace(/```\s*$/im, '').trim();
          const fb = c.indexOf('{'), lb = c.lastIndexOf('}');
          if (fb !== -1 && lb > fb) c = c.slice(fb, lb + 1);
          analysis = JSON.parse(c);
        }
        return json(res, 200, { ok: true, analysis });
      } catch (e) {
        console.error('[nexus-analyze]', e.message);
        return json(res, 500, { ok: false, error: e.message });
      }
    })();
    return;
  }

  // ── POST /nexus-chat → Conversational AI about bearings + dealers
  if (req.method === 'POST' && url === '/nexus-chat') {
    let nbody = '';
    req.on('data', chunk => { nbody += chunk; });
    req.on('end', () => {
      (async () => {
        try {
          const GEMINI_KEY = process.env.GEMINI_API_KEY;
          if (!GEMINI_KEY) throw new Error('GEMINI_API_KEY not set');
          const { message, history } = JSON.parse(nbody);
          if (!message) throw new Error('No message provided');
          const BP = path.join(ROOT, 'data', 'bearings_data.csv');
          const DP = path.join(ROOT, 'data', 'dealer_network.csv');
          const pcsv = t => {
            const lines = t.trim().split('\n'); const hdrs = lines[0].split(',').map(h => h.trim());
            return lines.slice(1).filter(l => l.trim()).map(line => {
              const cols = line.split(',').map(c => c.trim()); const o = {};
              hdrs.forEach((h, i) => o[h] = cols[i] ?? ''); return o;
            });
          };
          let ctx = 'No data loaded.';
          if (fs.existsSync(BP)) {
            const bearings  = pcsv(fs.readFileSync(BP, 'utf-8'));
            const dealers   = fs.existsSync(DP) ? pcsv(fs.readFileSync(DP, 'utf-8')) : [];
            const total     = bearings.length;
            const critical  = bearings.filter(b => b.Status === 'Critical').length;
            const degrading = bearings.filter(b => b.Status === 'Degrading').length;
            const avgRUL    = Math.round(bearings.reduce((s, b) => s + Number(b.RUL_Days || 0), 0) / total);
            ctx = `FLEET: ${total} bearings — ${bearings.filter(b => b.Status === 'Healthy').length} Healthy, ${degrading} Degrading, ${critical} Critical. Avg RUL: ${avgRUL}d.` +
              `\nDEALERS: ${dealers.length} in network.` +
              `\nCritical units: ${bearings.filter(b => b.Status === 'Critical').slice(0, 6).map(b => `${b.Unit_ID}@${b.Location}(RUL=${b.RUL_Days}d)`).join(', ')}`;
          }
          const systemInstruction = [
            'You are NBC Nexus AI, an expert industrial IoT and maintenance intelligence assistant for NBC Bearing Company.',
            ctx,
            'Guidelines:',
            '- Answer questions about bearing status, RUL, failure modes, maintenance schedules',
            '- Reference specific unit IDs and data when relevant',
            '- Give actionable maintenance and operational recommendations',
            '- For dealer questions, analyze inventory, responsiveness, and CSI scores',
            '- Keep responses concise but comprehensive. Use technical bearing terminology.',
          ].join('\n');
          const genAI  = new GoogleGenerativeAI(GEMINI_KEY);
          const MODELS = ['gemini-3-flash-preview','gemini-2.5-flash','gemini-2.0-flash'];
          let result;
          for (const modelName of MODELS) {
            try {
              const model = genAI.getGenerativeModel({
                model: modelName, systemInstruction,
                generationConfig: { temperature: 0.8, topP: 0.9, maxOutputTokens: 2048 },
              });
              const chat = model.startChat({
                history: (history || []).slice(-16).map(h => ({ role: h.role, parts: [{ text: h.text }] })),
              });
              result = await chat.sendMessage(message); break;
            } catch (e) { /* try next model */ }
          }
          if (!result) {
            const reply2 = await callHuggingFace(`${systemInstruction}\n\nUser: ${message}\nAssistant:`, 'nexus-chat');
            return json(res, 200, { ok: true, reply: reply2 });
          }
          return json(res, 200, { ok: true, reply: result.response.text().trim() });
        } catch (e) { return json(res, 500, { ok: false, error: e.message }); }
      })();
    });
    return;
  }

  // ── POST /analyze → Gemini AI analysis
  if (req.method === 'POST' && url === '/analyze') {
    (async () => {
      try {
        const GEMINI_KEY = process.env.GEMINI_API_KEY;
        if (!GEMINI_KEY) throw new Error('GEMINI_API_KEY not set in .env');

        if (!fs.existsSync(CSV_PATH)) throw new Error('No CSV data found. Run a sync first.');
        const mechanics = parseCSV(fs.readFileSync(CSV_PATH, 'utf-8'));

        // ── Compute aggregated stats for prompt ────────────────────────────
        const totalMechanics = mechanics.length;
        const totalPoints    = mechanics.reduce((s, m) => s + m.points, 0);
        const totalPhotos    = mechanics.reduce((s, m) => s + m.images_submitted, 0);

        const tierCounts = { Bronze: 0, Silver: 0, Gold: 0, Diamond: 0 };
        mechanics.forEach(m => { tierCounts[m.tier] = (tierCounts[m.tier] || 0) + 1; });

        // City aggregation
        const cityMap = {};
        mechanics.forEach(m => {
          if (!m.last_city) return;
          if (!cityMap[m.last_city]) cityMap[m.last_city] = { city: m.last_city, state: m.last_state, count: 0, points: 0, photos: 0 };
          cityMap[m.last_city].count++;
          cityMap[m.last_city].points += m.points;
          cityMap[m.last_city].photos += m.images_submitted;
        });
        const cityStats = Object.values(cityMap).sort((a, b) => b.count - a.count);

        // Mechanics close to tier upgrade
        const nearUpgrade = mechanics.filter(m => {
          const thresholds = [500, 1500, 3000];
          return thresholds.some(t => m.points >= t - 100 && m.points < t);
        });

        const csvSummary = mechanics.map(m =>
          `${m.name}|${m.last_city||'Unknown'}|${m.last_state||'Unknown'}|${m.tier}|${m.points}pts|${m.images_submitted}photos`
        ).join('\n');

        // ── Chain-of-thought Gemini prompt ─────────────────────────────────
        const prompt = `
You are a world-class sales analytics AI for NBC Bearing, an Indian industrial bearing manufacturer.
NBC Bearing runs a WhatsApp loyalty bot where mechanics across India:
- Register and submit camera photos of NBC bearing installations
- Earn 10 points per verified photo
- Progress through tiers: Bronze(0-499) → Silver(500-1499) → Gold(1500-2999) → Diamond(3000+)
- Rewards include discounts, merchandise, and dealer priority access

═══════════════════════════════════════
CURRENT MECHANIC DATA SNAPSHOT:
═══════════════════════════════════════
Total Mechanics Registered: ${totalMechanics}
Total Loyalty Points Awarded: ${totalPoints}
Total Camera Photos Verified: ${totalPhotos}
Avg Photos per Mechanic: ${(totalPhotos / totalMechanics).toFixed(1)}
Avg Points per Mechanic: ${(totalPoints / totalMechanics).toFixed(0)}

TIER BREAKDOWN:
- 🥉 Bronze: ${tierCounts.Bronze} mechanics
- 🥈 Silver: ${tierCounts.Silver} mechanics
- 🥇 Gold:   ${tierCounts.Gold} mechanics
- 💎 Diamond: ${tierCounts.Diamond} mechanics

TOP CITIES BY MECHANIC DENSITY:
${cityStats.slice(0, 12).map(c => `- ${c.city}, ${c.state}: ${c.count} mechanics, ${c.photos} photos, ${c.points} total pts`).join('\n')}

MECHANICS CLOSE TO TIER UPGRADE (within 100 pts):
${nearUpgrade.map(m => `- ${m.name} (${m.tier}, ${m.points}pts, ${m.last_city})`).join('\n') || 'None currently'}

INDIVIDUAL MECHANIC DATA (Name|City|State|Tier|Points|Photos):
${csvSummary}

═══════════════════════════════════════
CHAIN-OF-THOUGHT INSTRUCTIONS:
═══════════════════════════════════════
Think through these questions step by step BEFORE generating output:

STEP 1 — ENGAGEMENT ANALYSIS:
 - What % of mechanics are highly active (Gold + Diamond)? Is this good or bad?
 - Which cities show strongest engagement (photos per mechanic ratio)?
 - Any city with many mechanics but low photos = disengagement risk?

STEP 2 — GEOGRAPHIC OPPORTUNITY:
 - Which major Indian industrial cities are MISSING from our data (proxy for untapped market)?
 - Which states have high mechanic density suggesting strong NBC brand presence?

STEP 3 — SALES FORECASTING (30-day):
 - If current avg submission rate continues per city, project photo submissions for next 30 days
 - Factor in tier progression: Silver/Gold mechanics submit 2x more than Bronze
 - Give realistic growth % estimates

STEP 4 — PPC PLANNING:
 - For Google Ads: target mechanics searching "bearing supplier near me", "NBC bearing dealer"
 - For Meta Ads: target 25-45 male mechanics, auto repair interest, tier-2/tier-3 cities
 - Allocate INR budget proportional to market size and current mechanic density gaps
 - Prioritize cities where we have SOME mechanics (proof of concept) but room to grow

STEP 5 — STOCK SUPPLY PLANNING:
 - High photos = high bearing replacement rate = need more stock
 - Diamond/Gold mechanics in a city = dealers should stock premium SKUs
 - Bronze-heavy cities = entry-level bearing SKUs (6200, 6300 series)

STEP 6 — RETENTION STRATEGIES:
 - How to push Bronze→Silver? (most critical mass)
 - Gold→Diamond upsell strategies?
 - What rewards/nudges work best at each tier?

═══════════════════════════════════════
OUTPUT REQUIREMENTS:
═══════════════════════════════════════
Return ONLY a valid JSON object with NO markdown, NO code fences, NO explanation outside the JSON.
Use this exact structure:

{
  "thinking_summary": "2-3 sentences summarizing your chain-of-thought reasoning",
  "insights": [
    {
      "title": "Short insight title",
      "detail": "2-3 sentence data-driven explanation",
      "action": "Specific actionable next step",
      "impact": "High|Medium|Low",
      "icon": "emoji"
    }
  ],
  "forecast": [
    {
      "city": "City name",
      "state": "State name",
      "current_mechanics": 0,
      "current_photos_total": 0,
      "projected_photos_30d": 0,
      "projected_new_mechanics_30d": 0,
      "growth_pct": 0,
      "confidence": "High|Medium|Low"
    }
  ],
  "ppc": [
    {
      "city": "City name",
      "state": "State",
      "platform": "Google Ads|Meta Ads|Both",
      "budget_inr": 0,
      "keywords": ["keyword1", "keyword2"],
      "target_audience": "Description",
      "best_time": "e.g. Weekdays 8am-12pm",
      "expected_reach": 0,
      "expected_leads": 0,
      "priority": "High|Medium|Low"
    }
  ],
  "stock": [
    {
      "city": "City name",
      "state": "State",
      "priority": "High|Medium|Low",
      "mechanics_count": 0,
      "photos_submitted": 0,
      "recommended_stock_units": 0,
      "top_skus": ["SKU1", "SKU2", "SKU3"],
      "rationale": "Why this city needs this stock"
    }
  ],
  "retention": [
    {
      "from_tier": "Bronze|Silver|Gold",
      "to_tier": "Silver|Gold|Diamond",
      "mechanics_count": 0,
      "avg_pts_needed": 0,
      "strategy": "Specific engagement strategy",
      "incentive": "Specific reward/incentive to offer",
      "timeline": "e.g. 3-4 weeks with 3 photos/week"
    }
  ],
  "untapped_cities": [
    {
      "city": "City",
      "state": "State",
      "why": "Reason this city is high potential",
      "suggested_action": "Specific first step"
    }
  ],
  "summary": "Executive summary paragraph in plain English, 4-5 sentences covering overall health, top opportunity, and top risk."
}

Provide 5 insights, top 8 cities for forecast, top 6 cities for PPC, top 8 cities for stock, all 3 tier transitions for retention, and 4 untapped cities.
`;

        console.log('[analyze] Sending to Gemini…');
        const genAI = new GoogleGenerativeAI(GEMINI_KEY);

        // Try models in order until one works
        const MODELS = [
          'gemini-3-flash-preview',
          'gemini-2.5-flash',
          'gemini-2.0-flash',
        ];
        let result;
        for (const modelName of MODELS) {
          try {
            const model = genAI.getGenerativeModel({
              model: modelName,
              generationConfig: {
                temperature: 1.0,
                topP: 0.95,
                maxOutputTokens: 65536,
                responseMimeType: 'application/json',
              },
            });
            result = await model.generateContent(prompt);
            console.log(`[analyze] ✅ Used model: ${modelName}`);
            break;
          } catch (tryErr) {
            console.log(`[analyze] Model ${modelName} failed: ${tryErr.message.split('\n')[0]}`);
          }
        }
        if (!result) {
          // ── HuggingFace fallback ────────────────────────────────────────
          console.log('[analyze] All Gemini models failed — falling back to HuggingFace…');
          const hfPrompt = `${prompt}\n\nIMPORTANT: Respond with valid JSON only. No markdown, no explanation.`;
          const hfText = await callHuggingFace(hfPrompt, 'analyze');
          let cleaned2 = hfText;
          const fb = cleaned2.indexOf('{'); const lb = cleaned2.lastIndexOf('}');
          if (fb !== -1 && lb > fb) cleaned2 = cleaned2.slice(fb, lb + 1);
          const analysis2 = JSON.parse(cleaned2);
          return json(res, 200, { ok: true, analysis: analysis2 });
        }
        const rawText  = result.response.text().trim();

        // Robustly extract the JSON object — handles markdown fences, leading/trailing text, extra commentary
        let cleaned = rawText;
        // Strip markdown code fences (```json ... ``` or ``` ... ```)
        cleaned = cleaned.replace(/^```(?:json)?\s*/im, '').replace(/```\s*$/im, '').trim();
        // If Gemini still prefixed text, find the first '{' and last '}' and extract that range
        const firstBrace = cleaned.indexOf('{');
        const lastBrace  = cleaned.lastIndexOf('}');
        if (firstBrace !== -1 && lastBrace !== -1 && lastBrace > firstBrace) {
          cleaned = cleaned.slice(firstBrace, lastBrace + 1);
        }

        let analysis;
        try {
          analysis = JSON.parse(cleaned);
        } catch (parseErr) {
          console.error('[analyze] JSON parse failed. Raw text snippet:', rawText.slice(0, 300));
          throw new Error(`Gemini returned malformed JSON: ${parseErr.message}`);
        }

        console.log('[analyze] ✅ Gemini responded successfully');
        return json(res, 200, { ok: true, analysis });
      } catch (e) {
        console.error('[analyze] Error:', e.message);
        return json(res, 500, { ok: false, error: e.message });
      }
    })();
    return;
  }

  // ── POST /chat → Gemini conversational AI with mechanic data context
  if (req.method === 'POST' && url === '/chat') {
    let body = '';
    req.on('data', chunk => { body += chunk; });
    req.on('end', () => {
      (async () => {
        try {
          const GEMINI_KEY = process.env.GEMINI_API_KEY;
          if (!GEMINI_KEY) throw new Error('GEMINI_API_KEY not set in .env');

          const { message, history } = JSON.parse(body);
          if (!message) throw new Error('No message provided');

          // Load current mechanic data as context
          let mechContext = 'No mechanic CSV data available yet.';
          if (fs.existsSync(CSV_PATH)) {
            const mechanics = parseCSV(fs.readFileSync(CSV_PATH, 'utf-8'));
            if (mechanics.length > 0) {
              const tierCounts = { Bronze:0, Silver:0, Gold:0, Diamond:0 };
              const cityMap = {};
              mechanics.forEach(m => {
                tierCounts[m.tier] = (tierCounts[m.tier]||0)+1;
                if (m.last_city) cityMap[m.last_city] = (cityMap[m.last_city]||0)+1;
              });
              const topCities = Object.entries(cityMap).sort((a,b)=>b[1]-a[1]).slice(0,8);
              const totalPts   = mechanics.reduce((s,m)=>s+m.points,0);
              const totalPhotos = mechanics.reduce((s,m)=>s+m.images_submitted,0);
              const top5 = [...mechanics].sort((a,b)=>b.points-a.points).slice(0,5);

              mechContext = `
CURRENT MECHANIC DATA (${mechanics.length} registered):
- Tier split: Bronze=${tierCounts.Bronze}, Silver=${tierCounts.Silver}, Gold=${tierCounts.Gold}, Diamond=${tierCounts.Diamond}
- Total points awarded: ${totalPts}
- Total photos submitted: ${totalPhotos}
- Avg photos/mechanic: ${(totalPhotos/mechanics.length).toFixed(1)}
- Top cities: ${topCities.map(c=>c[0]+' ('+c[1]+')').join(', ')}
- Top 5 mechanics: ${top5.map(m=>m.name+' '+m.tier+' '+m.points+'pts').join(', ')}
- All mechanics (Name|Tier|Points|City): ${mechanics.map(m=>`${m.name}|${m.tier}|${m.points}|${m.last_city||'Unknown'}`).join(', ')}
`;
            }
          }

          const systemInstruction = `You are Sales AI for NBC Bearing — an Indian industrial bearing manufacturer.
You have access to the company's WhatsApp mechanic loyalty program data.
You help the NBC sales team with insights, forecasts, strategy questions, and data queries.

ABOUT THE LOYALTY PROGRAM:
- Mechanics register via WhatsApp and submit camera photos of NBC bearing installations
- Each verified photo earns 10 points
- Tiers: Bronze(0-499) → Silver(500-1499) → Gold(1500-2999) → Diamond(3000+)

${mechContext}

Guidelines:
- Be concise but insightful — use bullet points and numbers where helpful
- Reference specific mechanic names/cities from the data when relevant
- For forecasts, explain your reasoning
- Always suggest actionable next steps
- Keep responses under 250 words unless asked for detail
- Use Indian context (INR, Indian cities, Indian industrial market)
- You can help with: competitor analysis, pricing strategy, inventory planning, loyalty program ideas, campaign ideas`;

          const genAI = new GoogleGenerativeAI(GEMINI_KEY);
          const MODELS = ['gemini-3-flash-preview','gemini-2.5-flash','gemini-2.0-flash'];

          let result;
          for (const modelName of MODELS) {
            try {
              const model = genAI.getGenerativeModel({
                model: modelName,
                systemInstruction,
                generationConfig: { temperature: 0.8, topP: 0.9, maxOutputTokens: 4096 },
              });

              // Build chat history for multi-turn context
              const chat = model.startChat({
                history: (history || []).map(h => ({
                  role: h.role,
                  parts: [{ text: h.text }]
                }))
              });

              result = await chat.sendMessage(message);
              console.log(`[chat] ✅ ${modelName}`);
              break;
            } catch (e) {
              console.log(`[chat] ${modelName} failed: ${e.message.split('\n')[0]}`);
            }
          }
          if (!result) {
            // ── HuggingFace fallback ──────────────────────────────────────
            console.log('[chat] All Gemini models failed — falling back to HuggingFace…');
            const hfPrompt = `${systemInstruction}\n\nUser: ${message}\nAssistant:`;
            const reply2 = await callHuggingFace(hfPrompt, 'chat');
            return json(res, 200, { ok: true, reply: reply2 });
          }

          const reply = result.response.text().trim();
          return json(res, 200, { ok: true, reply });
        } catch (e) {
          console.error('[chat] Error:', e.message);
          return json(res, 500, { ok: false, error: e.message });
        }
      })();
    });
    return;
  }

  res.writeHead(404);
  res.end('Not found');
});

server.on('error', (err) => {
  if (err.code === 'EADDRINUSE') {
    console.error(`\n❌ Port ${PORT} is already in use.`);
    console.error(`   This usually means server.js is already running.`);
    console.error(`   To fix: Stop the other process, or run:\n`);
    console.error(`   Get-Process node | Stop-Process -Force   (PowerShell)`);
    console.error(`   taskkill /F /IM node.exe                 (CMD)\n`);
  } else {
    console.error('Server error:', err);
  }
  process.exit(1);
});

server.listen(PORT, () => {
  console.log(`\n✅ NBC Bearing Dashboard running at http://localhost:${PORT}`);
  console.log(`   Press Ctrl+C to stop.\n`);
});
