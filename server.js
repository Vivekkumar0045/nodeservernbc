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

  // ── POST /analyze → Gemini AI full-channel bearing sales intelligence
  if (req.method === 'POST' && url === '/analyze') {
    (async () => {
      try {
        const GEMINI_KEY = process.env.GEMINI_API_KEY;
        if (!GEMINI_KEY) throw new Error('GEMINI_API_KEY not set in .env');

        // ── Load all 4 channel CSVs ────────────────────────────────────────
        const readCsv = (name) => {
          const p = path.join(ROOT, 'data', name);
          return fs.existsSync(p) ? parseCSV(fs.readFileSync(p, 'utf-8')) : [];
        };

        const mechanics    = fs.existsSync(CSV_PATH) ? parseCSV(fs.readFileSync(CSV_PATH, 'utf-8')) : [];
        const stockists    = readCsv('stockists.csv');
        const distributors = readCsv('distributors.csv');
        const retailers    = readCsv('retailers.csv');
        const hierarchy    = readCsv('network_hierarchy.csv');

        // ── Mechanic aggregates ───────────────────────────────────────────
        const totalMechanics = mechanics.length;
        const totalPhotos    = mechanics.reduce((s, m) => s + (+m.images_submitted || 0), 0);
        const totalPoints    = mechanics.reduce((s, m) => s + (+m.points || 0), 0);
        const tierCounts     = { Bronze:0, Silver:0, Gold:0, Diamond:0 };
        mechanics.forEach(m => { tierCounts[m.tier] = (tierCounts[m.tier]||0) + 1; });
        const highTierPct = Math.round(((tierCounts.Gold + tierCounts.Diamond) / Math.max(totalMechanics, 1)) * 100);

        // City-level mechanic activity (demand signal)
        const cityMap = {};
        mechanics.forEach(m => {
          if (!m.last_city) return;
          if (!cityMap[m.last_city]) cityMap[m.last_city] = { city:m.last_city, state:m.last_state, count:0, photos:0, points:0 };
          cityMap[m.last_city].count++;
          cityMap[m.last_city].photos  += (+m.images_submitted || 0);
          cityMap[m.last_city].points  += (+m.points || 0);
        });
        const topCities = Object.values(cityMap).sort((a,b)=>b.photos-a.photos).slice(0,10);

        // ── Stockist aggregates ────────────────────────────────────────────
        const stkTotalTarget = stockists.reduce((s,x)=>s+(+x.monthly_target_units||0),0);
        const zoneTargets    = {};
        stockists.forEach(s => { zoneTargets[s.zone] = (zoneTargets[s.zone]||0) + (+s.monthly_target_units||0); });
        const stkSummary = stockists.map(s =>
          `${s.name}|${s.city}|${s.zone}Zone|Target:${s.monthly_target_units}units|Credit:${s.credit_limit_lakh}L`
        ).join('\n');

        // ── Distributor aggregates ─────────────────────────────────────────
        const dstTotalTarget = distributors.reduce((s,x)=>s+(+x.monthly_target_units||0),0);
        const dstSummary = distributors.map(d =>
          `${d.name}|${d.city},${d.state}|Stockist:${d.stockist_id}|Target:${d.monthly_target_units}units|Retailers:${d.total_retailers}`
        ).join('\n');

        // ── Retailer aggregates ────────────────────────────────────────────
        const retTotalPurchase = retailers.reduce((s,x)=>s+(+x.monthly_purchase_units||0),0);
        const retOutstanding   = retailers.reduce((s,x)=>s+(+x.outstanding_amount||0),0);
        const retTierCounts    = {};
        retailers.forEach(r => { retTierCounts[r.tier] = (retTierCounts[r.tier]||0)+1; });
        const retSummary = retailers.map(r =>
          `${r.shop_name}|${r.city},${r.state}|Dist:${r.distributor_id}|Purchase:${r.monthly_purchase_units}units/mo|Pts:${r.loyalty_points}|Tier:${r.tier}|Outstanding:₹${r.outstanding_amount}`
        ).join('\n');

        // ── Network hierarchy (mechanic-to-stockist mapping) ───────────────
        const mechToChain = {};
        hierarchy.forEach(row => {
          mechToChain[row.mechanic_phone] = { stk: row.stockist_id, dst: row.distributor_id, ret: row.retailer_id };
        });
        const mappedMechanics = mechanics.filter(m => mechToChain[m.phone]).length;

        // ── Zone-level mechanic demand rollup ──────────────────────────────
        const zoneDemand = {};
        hierarchy.forEach(row => {
          const mech = mechanics.find(m => m.phone == row.mechanic_phone);
          const stk  = stockists.find(s => s.stockist_id === row.stockist_id);
          if (!stk) return;
          const zone = stk.zone || 'Unknown';
          if (!zoneDemand[zone]) zoneDemand[zone] = { photos:0, points:0, mechanics:0 };
          if (mech) {
            zoneDemand[zone].photos   += (+mech.images_submitted || 0);
            zoneDemand[zone].points   += (+mech.points || 0);
            zoneDemand[zone].mechanics++;
          }
        });

        // ── Build prompt ───────────────────────────────────────────────────
        const prompt = `
You are a world-class B2B sales analytics AI for NBC Bearing, an Indian industrial bearing manufacturer.
Your goal is to increase NBC's AFTERMARKET BEARING SALES by analyzing all 4 channel layers simultaneously.

═══════════════════════════════════════════════════════════
NETWORK STRUCTURE (5 stockists → 15 distributors → 45 retailers → 49 mechanics)
═══════════════════════════════════════════════════════════

── STOCKISTS (regional hubs, sell to distributors) ──
${stkSummary}
Total Monthly Target: ${stkTotalTarget.toLocaleString()} units/mo
Zone Targets: ${Object.entries(zoneTargets).map(([z,t])=>`${z}:${t}`).join(' | ')}

── DISTRIBUTORS (buy from stockists, sell to retailers) ──
${dstSummary}
Total Distributor Target: ${dstTotalTarget.toLocaleString()} units/mo

── RETAILERS (frontline shops, sell to end users & mechanics) ──
${retSummary}
Total Retailer Monthly Purchase: ${retTotalPurchase.toLocaleString()} units/mo
Outstanding Credit: ₹${retOutstanding.toLocaleString()}
Retailer Tier Mix: ${Object.entries(retTierCounts).map(([t,n])=>`${t}:${n}`).join(', ')}

── MECHANICS (aftermarket demand signal — earn points by photographing NBC installations) ──
Total Mechanics: ${totalMechanics} | Mapped to network: ${mappedMechanics}
Total Photos (bearing installations verified): ${totalPhotos}
Tier distribution: Bronze:${tierCounts.Bronze} | Silver:${tierCounts.Silver} | Gold:${tierCounts.Gold} | Diamond:${tierCounts.Diamond}
High-tier mechanics (Gold+Diamond): ${highTierPct}%
Avg photos per mechanic: ${(totalPhotos/Math.max(totalMechanics,1)).toFixed(1)}

Zone-level mechanic demand (photos = bearing replacements):
${Object.entries(zoneDemand).map(([z,d])=>`  ${z}: ${d.mechanics} mechanics, ${d.photos} photos, ${d.points} pts`).join('\n')||'  No mapped data yet'}

Top cities by mechanic activity:
${topCities.map(c=>`  ${c.city},${c.state}: ${c.count} mechs, ${c.photos} photos`).join('\n')}

NOTE: Every photo = 1 bearing installation verified = real aftermarket demand. Points are a proxy for mechanic engagement, not the end goal.

═══════════════════════════════════════════════════════════
CHAIN-OF-THOUGHT INSTRUCTIONS:
═══════════════════════════════════════════════════════════

STEP 1 — PIPELINE HEALTH:
 - Compare stockist targets vs estimated retailer actuals. Estimate achievement %.
 - Which zone has strongest demand (photos + retailer purchase combined)?
 - Identify the weakest zone or gap in the network.
 - Mechanic demand signal: High(>200 total photos), Medium(50-200), Low(<50)

STEP 2 — KEY INSIGHTS (6 insights):
 - What does the mechanic photo data tell us about aftermarket demand by zone?
 - Which retailers are performing above/below target?
 - Are there distributors with too few retailers?
 - What's the outstanding credit risk?
 - Which zones need urgent PPC/territory action?

STEP 3 — 30-DAY SALES FORECAST BY ZONE:
 - For each of 5 zones (North/South/East/West/Central):
   - current_monthly_units = distributor targets for that zone (realistic actuals ~ 70-85% of target)
   - projected_30d_units = factoring mechanic demand signal, retailer activity, and zone seasonality
   - growth_pct = realistic YoY growth estimate
   - demand_signal = High/Medium/Low based on zone mechanic activity
 - Higher photo activity in a zone = higher bearing replacement demand = higher sales potential

STEP 4 — CHANNEL PERFORMANCE SCORECARD (all 4 channels):
 - Stockists: score 0-100 based on zone coverage, target feasibility, credit limits
 - Distributors: score based on retailer count, target vs retailer actual
 - Retailers: score based on purchase volume, outstanding, tier mix
 - Mechanics: score based on photo rate, tier progression, coverage

STEP 5 — PPC CAMPAIGNS (6 campaigns targeting aftermarket sales):
 - DO NOT target mechanic recruitment in PPC — target SELLING BEARINGS to:
   a) Retailers: "NBC Bearing dealer near me", "buy NBC bearings wholesale"
   b) Mechanics: "NBC bearing installation", "bearing replacement supplier"
   c) Fleet buyers: "industrial bearing supplier India", "automotive bearing distributor"
   d) B2B: target purchasing managers in auto workshops, manufacturing plants
 - Platforms: Google Ads for B2B intent, Meta Ads for mechanics (25-45 male, workshop interest)
 - Allocate budget to highest-demand zones first
 - Include specific keyword lists relevant to NBC Bearing aftermarket

STEP 6 — TERRITORY OPPORTUNITIES (6 cities/zones):
 - Where do we have active mechanics but NO mapped retailer? → immediate retailer recruitment
 - Where do we have Gold/Diamond mechanics but low retailer purchase? → push stock through
 - Which states have zero mechanic coverage? → distributor expansion
 - Cities with high mechanic photography but no nearby stockist? → supply gap

STEP 7 — STOCK RECOMMENDATIONS (6 locations):
 - High mechanic photos in a city = high bearing wear = stock demand at nearest retailer/distributor
 - Gold/Diamond mechanics use premium series: deep groove bearings 6200-6310, taper TAPER series
 - Bronze-heavy areas: entry-level 6000, 6200 series
 - Outstanding credit = cash flow issue → reduce credit, push pre-paid stock programs
 - Flag urgency: High if critical zone stock gap, Medium if growing demand, Low if stable

STEP 8 — TOP PERFORMERS (8 entities across all 4 channels):
 - 2 top retailers by monthly purchase volume
 - 2 top distributors by retailer coverage and target
 - 1 top stockist by zone performance
 - 2 top mechanics (Diamond/Gold) driving demand
 - 1 combined metric winner

═══════════════════════════════════════════════════════════
OUTPUT — Return ONLY valid JSON, no markdown fences:
═══════════════════════════════════════════════════════════
{
  "thinking_summary": "2-3 sentence summary of chain-of-thought reasoning focused on bearing sales",
  "pipeline_health": {
    "total_target_units_monthly": 0,
    "estimated_actual_units_monthly": 0,
    "achievement_pct": 0,
    "top_zone": "North|South|East|West|Central",
    "weakest_zone": "Zone name",
    "mechanic_demand_signal": "High|Medium|Low",
    "mechanic_engagement_pct": 0
  },
  "insights": [
    { "title": "Short title", "detail": "2-3 sentence data-driven explanation", "action": "Specific next step", "impact": "High|Medium|Low", "icon": "emoji" }
  ],
  "sales_forecast": [
    { "zone": "North", "current_monthly_units": 0, "projected_30d_units": 0, "growth_pct": 0, "demand_signal": "High|Medium|Low", "key_driver": "one sentence", "confidence": "High|Medium|Low" }
  ],
  "channel_performance": [
    { "channel": "Stockists", "icon": "🏭", "score": 0, "target_achievement_pct": 0, "top_entity": "entity name", "strength": "one-line strength", "gap": "one-line gap", "action": "specific action" }
  ],
  "ppc_campaigns": [
    { "campaign_name": "Campaign Name", "objective": "Drive bearing sales to retailers", "target_channel": "Retailers|Mechanics|Distributors|B2B", "region": "zone or city", "platform": "Google Ads|Meta Ads|Both", "budget_inr": 0, "keywords": ["kw1","kw2","kw3"], "target_audience": "description", "expected_impressions": 0, "expected_conversions": 0, "priority": "High|Medium|Low" }
  ],
  "territory_opportunities": [
    { "zone": "North", "city": "City", "state": "State", "opportunity_type": "Expand network|Increase volume|Fill gap|Launch", "potential_units": 0, "action": "specific step", "priority": "High|Medium|Low" }
  ],
  "stock_recommendations": [
    { "zone": "North", "city": "City", "retailer_count": 0, "mechanic_activity": "X mechanics, Y photos", "recommended_units": 0, "sku_focus": ["NBC 6205","NBC 6305"], "rationale": "why", "urgency": "High|Medium|Low" }
  ],
  "top_performers": [
    { "rank": 1, "entity_type": "Retailer|Distributor|Stockist|Mechanic", "name": "name", "city": "city", "metric_label": "Monthly Purchase", "metric_value": "180 units", "recognition": "short acknowledgement" }
  ],
  "summary": "Executive summary paragraph (4-5 sentences): overall sales pipeline health, top opportunity, key risk, recommended first action for NBC management to drive aftermarket volume growth."
}

Provide: 6 insights, 5 zones for forecast, 4 channel scorecards, 6 PPC campaigns, 6 territory opportunities, 6 stock recommendations, 8 top performers.
`;

        console.log('[analyze] Sending full-channel prompt to Gemini…');
        const genAI = new GoogleGenerativeAI(GEMINI_KEY);

        const MODELS = ['gemini-3-flash-preview', 'gemini-2.5-flash', 'gemini-2.0-flash'];
        let result;
        for (const modelName of MODELS) {
          try {
            const model = genAI.getGenerativeModel({
              model: modelName,
              generationConfig: { temperature: 1.0, topP: 0.95, maxOutputTokens: 65536, responseMimeType: 'application/json' }
            });
            result = await model.generateContent(prompt);
            console.log(`[analyze] ✅ Used model: ${modelName}`);
            break;
          } catch (tryErr) {
            console.log(`[analyze] Model ${modelName} failed: ${tryErr.message.split('\n')[0]}`);
          }
        }

        if (!result) {
          console.log('[analyze] All Gemini models failed — falling back to HuggingFace…');
          const hfText = await callHuggingFace(prompt + '\n\nRespond with valid JSON only.', 'analyze');
          let c2 = hfText;
          const fb2 = c2.indexOf('{'); const lb2 = c2.lastIndexOf('}');
          if (fb2 !== -1 && lb2 > fb2) c2 = c2.slice(fb2, lb2 + 1);
          return json(res, 200, { ok: true, analysis: JSON.parse(c2) });
        }

        let cleaned = result.response.text().trim();
        cleaned = cleaned.replace(/^```(?:json)?\s*/im, '').replace(/```\s*$/im, '').trim();
        const fb = cleaned.indexOf('{'); const lb = cleaned.lastIndexOf('}');
        if (fb !== -1 && lb !== -1 && lb > fb) cleaned = cleaned.slice(fb, lb + 1);

        let analysis;
        try {
          analysis = JSON.parse(cleaned);
        } catch (parseErr) {
          console.error('[analyze] JSON parse failed:', cleaned.slice(0, 300));
          throw new Error(`Gemini returned malformed JSON: ${parseErr.message}`);
        }

        console.log('[analyze] ✅ Full-channel sales intelligence ready');
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

  // ── GET /network-data → stockists, distributors, retailers, hierarchy CSVs
  if (req.method === 'GET' && url === '/network-data') {
    try {
      const pcsv = (text) => {
        const lines = text.trim().split('\n');
        const headers = lines[0].split(',').map(h => h.trim());
        return lines.slice(1).filter(l => l.trim()).map(line => {
          const cols = line.split(',').map(c => c.trim());
          const obj = {};
          headers.forEach((h, i) => obj[h] = cols[i] ?? '');
          return obj;
        });
      };
      const read = (name) => {
        const p = path.join(ROOT, 'data', name);
        return fs.existsSync(p) ? pcsv(fs.readFileSync(p, 'utf-8')) : [];
      };
      return json(res, 200, {
        ok: true,
        stockists:    read('stockists.csv'),
        distributors: read('distributors.csv'),
        retailers:    read('retailers.csv'),
        hierarchy:    read('network_hierarchy.csv'),
      });
    } catch (e) {
      return json(res, 500, { ok: false, error: e.message });
    }
  }

  // ── POST /network-analyze → Gemini AI analysis for stockist/distributor/retailer
  if (req.method === 'POST' && url === '/network-analyze') {
    let nbody = '';
    req.on('data', chunk => { nbody += chunk; });
    req.on('end', () => {
      (async () => {
        try {
          const GEMINI_KEY = process.env.GEMINI_API_KEY;
          if (!GEMINI_KEY) throw new Error('GEMINI_API_KEY not set in .env');
          const { entity } = JSON.parse(nbody);
          if (!['stockist','distributor','retailer'].includes(entity))
            throw new Error('entity must be stockist, distributor, or retailer');

          const pcsv = (text) => {
            const lines = text.trim().split('\n');
            const headers = lines[0].split(',').map(h => h.trim());
            return lines.slice(1).filter(l => l.trim()).map(line => {
              const cols = line.split(',').map(c => c.trim());
              const obj = {}; headers.forEach((h, i) => obj[h] = cols[i] ?? ''); return obj;
            });
          };
          const read = (name) => {
            const p = path.join(ROOT, 'data', name);
            return fs.existsSync(p) ? pcsv(fs.readFileSync(p, 'utf-8')) : [];
          };
          const stockists    = read('stockists.csv');
          const distributors = read('distributors.csv');
          const retailers    = read('retailers.csv');
          const hierarchy    = read('network_hierarchy.csv');

          let dataSummary = '';
          if (entity === 'stockist') {
            dataSummary = stockists.map(s =>
              `Zone=${s.zone} City=${s.city} State=${s.state} Target=${s.monthly_target_units}units CreditLimit=₹${s.credit_limit_lakh}L Distributors=${distributors.filter(d=>d.stockist_id===s.stockist_id).length} Status=${s.status}`
            ).join('\n');
          } else if (entity === 'distributor') {
            dataSummary = distributors.map(d => {
              const rets = retailers.filter(r => r.distributor_id === d.distributor_id).length;
              const stk = stockists.find(s => s.stockist_id === d.stockist_id);
              return `ID=${d.distributor_id} City=${d.city} State=${d.state} Stockist=${stk?.zone||d.stockist_id} Target=${d.monthly_target_units}units CreditLimit=₹${d.credit_limit_lakh}L Retailers=${rets} Status=${d.status}`;
            }).join('\n');
          } else {
            dataSummary = retailers.map(r => {
              const dist = distributors.find(d => d.distributor_id === r.distributor_id);
              const mechCount = hierarchy.filter(h => h.retailer_id === r.retailer_id).length;
              return `ID=${r.retailer_id} City=${r.city} State=${r.state} Distributor=${dist?.city||r.distributor_id} Tier=${r.tier} Purchases=${r.monthly_purchase_units}units Points=${r.loyalty_points} Outstanding=₹${r.outstanding_amount} Mechanics=${mechCount}`;
            }).join('\n');
          }

          const entityLabel = entity === 'stockist' ? 'Stockist (Zone Hub)' : entity === 'distributor' ? 'Distributor' : 'Retailer';
          const prompt = [
            `You are a supply chain and sales analytics AI for NBC Bearing, an Indian industrial bearing manufacturer.`,
            `Analyze the following ${entityLabel} network data and return a JSON intelligence report.`,
            ``,
            `DATA (${entity === 'stockist' ? stockists.length : entity === 'distributor' ? distributors.length : retailers.length} records):`,
            dataSummary,
            ``,
            `CONTEXT: NBC Bearing operates a 4-tier channel: Stockists → Distributors → Retailers → Mechanics.`,
            `There are ${stockists.length} stockists, ${distributors.length} distributors, ${retailers.length} retailers, and ${hierarchy.length} mechanic-retailer links.`,
            ``,
            `ANALYSIS STEPS:`,
            `1. Identify the top 3 performing ${entity}s and what makes them successful.`,
            `2. Identify the bottom 2-3 ${entity}s that need attention and why.`,
            `3. Spot geographic concentration risks or gaps.`,
            `4. Identify credit limit vs throughput mismatches.`,
            `5. Recommend 3-4 concrete strategic actions for the next 30 days.`,
            `6. Identify 2-3 key risks in the ${entity} layer of the channel.`,
            `7. Spot 2-3 growth opportunities.`,
            ``,
            `Return ONLY valid JSON with this exact structure:`,
            `{`,
            `  "thinking": "brief chain-of-thought across all 7 steps (2-3 sentences)",`,
            `  "exec_summary": "2-3 sentence executive summary for management",`,
            `  "insights": [{"icon":"emoji","title":"...","priority":"High|Medium|Low","detail":"...","action":"..."}],`,
            `  "opportunities": [{"icon":"emoji","title":"...","detail":"...","action":"..."}],`,
            `  "risks": [{"icon":"emoji","title":"...","priority":"High|Medium|Low","detail":"...","action":"..."}],`,
            `  "recommendations": [{"title":"...","detail":"...","priority":"High|Medium|Low"}]`,
            `}`,
            `Rules: insights=4-5 items, opportunities=2-3 items, risks=2-3 items, recommendations=3-4 items.`,
          ].join('\n');

          const genAI = new GoogleGenerativeAI(GEMINI_KEY);
          const MODELS = ['gemini-3-flash-preview','gemini-2.5-flash','gemini-2.0-flash'];
          let result;
          for (const modelName of MODELS) {
            try {
              console.log(`[network-analyze/${entity}] Trying ${modelName}…`);
              const model = genAI.getGenerativeModel({
                model: modelName,
                generationConfig: { temperature: 1.0, topP: 0.95, maxOutputTokens: 16384, responseMimeType: 'application/json' },
              });
              result = await model.generateContent(prompt);
              console.log(`[network-analyze/${entity}] ✅ ${modelName}`);
              break;
            } catch (e) { console.log(`[network-analyze/${entity}] ${modelName} failed: ${e.message.slice(0,60)}`); }
          }
          let analysis;
          if (!result) {
            const hfText = await callHuggingFace(prompt + '\n\nRespond with valid JSON only.', `network-analyze/${entity}`);
            let c = hfText; const fb = c.indexOf('{'), lb = c.lastIndexOf('}');
            if (fb !== -1 && lb > fb) c = c.slice(fb, lb + 1);
            analysis = JSON.parse(c);
          } else {
            let c = result.response.text().trim().replace(/^```(?:json)?\s*/im,'').replace(/```\s*$/im,'').trim();
            const fb = c.indexOf('{'), lb = c.lastIndexOf('}');
            if (fb !== -1 && lb > fb) c = c.slice(fb, lb + 1);
            analysis = JSON.parse(c);
          }
          return json(res, 200, { ok: true, analysis });
        } catch (e) {
          console.error('[network-analyze]', e.message);
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
