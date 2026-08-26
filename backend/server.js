require('dotenv').config();
const express = require('express');
const cors = require('cors');
const path = require('path');
const jwt = require('jsonwebtoken');
const { createClient } = require('@supabase/supabase-js');
const scheduler = require('./scheduler');
const { LOCALITIES, CENTER } = require('./localities');

const app = express();
const PORT = process.env.PORT || 5500;

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

//  Supabase Client ──
const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_KEY = process.env.SUPABASE_KEY;

if (!SUPABASE_URL || !SUPABASE_KEY) {
  console.error('SUPABASE_URL or SUPABASE_KEY missing in .env');
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

//  Admin auth config ──
const ADMIN_USERNAME = process.env.ADMIN_USERNAME || 'admin';
const ADMIN_PASSWORD = process.env.ADMIN_PASSWORD || 'CityDemo2026';
const JWT_SECRET      = process.env.JWT_SECRET || 'change-this-secret-before-real-use';
const JWT_EXPIRES_IN  = '12h';

// Escalation thresholds (kept in sync with nlp_pipeline/NLProcessing.py)
const ESCALATION_NET_UPVOTES = 20;
const ESCALATION_SEVERITY    = 8;

// "Suggested forward" thresholds for admin review — deliberately lower than
// the auto-escalation thresholds above, so admins can act early on a
// complaint that's trending before it hits auto-forward.
const SUGGEST_NET_UPVOTES = 10;
const SUGGEST_SEVERITY    = 6;

// Placeholder department inboxes for the manual-forward proof of concept.
// Swap for real addresses (or pull from a DB table) when wiring up real mail.
const CATEGORY_AUTHORITY_EMAIL = {
  roads:       'roads.dept@hamirpurmc.gov.in',
  water:       'water.dept@hamirpurmc.gov.in',
  electricity: 'electricity.dept@hamirpurmc.gov.in',
  waste:       'sanitation.dept@hamirpurmc.gov.in',
  other:       'grievance@hamirpurmc.gov.in',
};

// Voting radius
const VOTE_RADIUS_KM = 0.5; // 500 metres

//  config health
//  to track whether credentials are working; doesn't crash the server when fail.
const health = {
  supabase: { ok: false, lastCheck: null, error: null },
  twitter: { ok: false, lastCheck: null, error: null }
};

async function checkSupabaseHealth() {
  try {
    const { error } = await supabase.from('analyzed_tweets').select('id').limit(1);
    health.supabase.ok = !error;
    health.supabase.error = error ? error.message : null;
    if (error) console.warn(`Supabase issue: ${error.message}`);
  } catch (e) {
    health.supabase.ok = false;
    health.supabase.error = e.message;
    console.warn(`Supabase unreachable: ${e.message}`);
  }
  health.supabase.lastCheck = new Date().toISOString();
}

// check start + every 10 minutes
checkSupabaseHealth();
setInterval(checkSupabaseHealth, 10 * 60 * 1000);

//  Helpers 
function withinRadius(uLat, uLon, iLat, iLon, km = VOTE_RADIUS_KM) {
  const toRad = d => d * Math.PI / 180;
  const R = 6371;
  const dLat = toRad(iLat - uLat), dLon = toRad(iLon - uLon);
  const a = Math.sin(dLat/2)**2 + Math.cos(toRad(uLat)) * Math.cos(toRad(iLat)) * Math.sin(dLon/2)**2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a)) <= km;
}

// Real tweets/simulated complaints only get a locality-centroid match (see
// findNearestLocality); manual Test Media posts carry a precise pin-dropped
// or browser-geolocated point. That distinction drives the map's exact vs.
// approximate circle styling.
function locationPrecision(t) {
  if (!t.latitude || !t.longitude) return null;
  return t.source === 'manual' ? 'exact' : 'approximate';
}

// Suggests a manual forward for complaints trending toward escalation but
// not yet past the auto-forward thresholds — surfaced as an admin-only hint.
function suggestForward(t) {
  if (t.forwarded) return { suggested_forward: false, suggested_forward_reason: null };
  const net = (t.upvotes || 0) - (t.downvotes || 0);
  const severity = t.severity_score || 0;
  if (severity >= SUGGEST_SEVERITY) {
    return { suggested_forward: true, suggested_forward_reason: `High severity (${severity})` };
  }
  if (net >= SUGGEST_NET_UPVOTES) {
    return { suggested_forward: true, suggested_forward_reason: `Trending — ${net} net upvotes` };
  }
  return { suggested_forward: false, suggested_forward_reason: null };
}

function fmtTweet(t) {
  const base = t.severity_score || 0;
  return {
    id:                 String(t.tweet_id || t.id),
    text:               t.text,
    severity_score:     base,
    effective_severity: base + (t.upvotes || 0) - (t.downvotes || 0),
    sentiment_score:    t.sentiment_score,
    issue_category:     t.issue_category,
    bert_label:         t.bert_label,
    troll_flag:         t.troll_flag,
    upvotes:            t.upvotes || 0,
    downvotes:          t.downvotes || 0,
    like_count:         t.like_count || 0,
    retweet_count:      t.retweet_count || 0,
    timestamp:          t.timestamp,
    latitude:           t.latitude,
    longitude:          t.longitude,
    locality_name:      t.locality_name || null,
    location_precision: locationPrecision(t),
    source:             t.source || 'twitter',
    has_location:       !!(t.latitude && t.longitude),
    forwarded:          !!t.forwarded,
    forwarded_at:       t.forwarded_at || null,
    escalation_reason:  t.escalation_reason || null,
  };
}

// Admin view includes personally-identifying fields and forward-review hints
// not shown on the public dashboard.
function fmtAdminTweet(t) {
  return {
    ...fmtTweet(t),
    user_id:  t.user_id,
    username: t.username,
    ...suggestForward(t),
  };
}

// ── Lightweight classifier for the Test Media tool ─────────────────────────
// Mirrors nlp_pipeline/NLProcessing.py's category keywords and severity
// formula, so a manually-posted test tweet gets an instant, consistent
// result without depending on the Python/DistilBERT pipeline being up.
const CATEGORY_KEYWORDS = {
  roads:       ['road', 'pothole', 'street', 'highway'],
  water:       ['water', 'pipeline', 'drain', 'sewage'],
  electricity: ['electricity', 'power', 'light', 'transformer'],
  waste:       ['garbage', 'waste', 'trash', 'dump'],
};

function classifyCategory(text) {
  const t = (text || '').toLowerCase();
  for (const [cat, keywords] of Object.entries(CATEGORY_KEYWORDS)) {
    if (keywords.some(w => t.includes(w))) return cat;
  }
  return 'other';
}

const TROLL_PATTERNS = [
  { re: /\b(?:idiot|stupid|dumb|fool|hate|nonsense)\b/i, weight: 3 },
  { re: /[!?]{3,}/,                                        weight: 1 },
  { re: /\b[A-Z]{4,}\b/,                                    weight: 2 },
  { re: /https?:\/\/[^\s]+/,                                weight: 2 },
  { re: /\b(\w+)\s+\1\s+\1\b/i,                             weight: 2 },
  { re: /\b(?:kill|die|worst|useless|trash)\b/i,            weight: 3 },
];

function detectTroll(text) {
  let score = 0;
  for (const { re, weight } of TROLL_PATTERNS) if (re.test(text || '')) score += weight;
  return { isTroll: score >= 3, score };
}

function findNearestLocality(lat, lon) {
  const toRad = d => d * Math.PI / 180;
  const R = 6371;
  let best = null, bestDist = Infinity;
  for (const loc of LOCALITIES) {
    const dLat = toRad(loc.lat - lat), dLon = toRad(loc.lon - lon);
    const a = Math.sin(dLat/2)**2 + Math.cos(toRad(lat)) * Math.cos(toRad(loc.lat)) * Math.sin(dLon/2)**2;
    const dist = R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    if (dist < bestDist) { bestDist = dist; best = loc; }
  }
  return best ? best.name : null;
}

const TEST_FIRST_NAMES = ['rohan', 'priya', 'aman', 'neha', 'vikram', 'simran', 'karan', 'ayesha', 'manav', 'divya'];
function randomTestUsername() {
  const name = TEST_FIRST_NAMES[Math.floor(Math.random() * TEST_FIRST_NAMES.length)];
  return `${name}_${Math.floor(100 + Math.random() * 900)}`;
}

async function maybeEscalate(tweet) {
  if (tweet.forwarded) return {}; // already forwarded, nothing to do

  const netUpvotes = (tweet.upvotes || 0) - (tweet.downvotes || 0);
  const severity = tweet.severity_score || 0;

  let reason = null;
  if (netUpvotes >= ESCALATION_NET_UPVOTES) reason = 'vote_threshold';
  else if (severity >= ESCALATION_SEVERITY) reason = 'high_severity';

  if (!reason) return {};

  return {
    forwarded: true,
    forwarded_at: new Date().toISOString(),
    escalation_reason: reason,
  };
}

//  Auth middleware ──
function requireAdmin(req, res, next) {
  const header = req.headers.authorization || '';
  const token = header.startsWith('Bearer ') ? header.slice(7) : null;
  if (!token) return res.status(401).json({ error: 'Missing token' });

  try {
    req.admin = jwt.verify(token, JWT_SECRET);
    next();
  } catch (e) {
    return res.status(401).json({ error: 'Invalid or expired token' });
  }
}

//  API: Health ──
app.get("/api/health", (req, res) => res.json(health));

//  API: Pipeline status ─
app.get("/api/pipeline/status", (req, res) => res.json(scheduler.state));

//  API: Localities gazetteer (for the map tab) ──
app.get('/api/localities', (req, res) => {
  res.json({ center: CENTER, radius_km: 5, localities: LOCALITIES });
});

//  API: Get all tweets 
app.get('/api/tweets', async (req, res) => {
  const { data, error } = await supabase
    .from('analyzed_tweets')
    .select('*')
    .order('severity_score', { ascending: false })
    .limit(100);

  if (error) return res.status(500).json({ error: error.message });
  res.json(data.map(fmtTweet));
});

//  API: Single tweet ──
app.get('/api/tweets/:id', async (req, res) => {
  const { data, error } = await supabase
    .from('analyzed_tweets')
    .select('*')
    .eq('tweet_id', req.params.id)
    .single();

  if (error || !data) return res.status(404).json({ error: 'Tweet not found' });
  res.json(fmtTweet(data));
});

//  API: Vote ─
app.post('/api/tweets/:id/vote', async (req, res) => {
  const { type, userLat, userLon } = req.body;
  if (!['up', 'down'].includes(type)) return res.status(400).json({ error: 'Invalid vote type' });

  const { data: tweet, error: fetchErr } = await supabase
    .from('analyzed_tweets')
    .select('*')
    .eq('tweet_id', req.params.id)
    .single();

  if (fetchErr || !tweet) return res.status(404).json({ error: 'Tweet not found' });

  // A complaint without a resolved location can't be voted on — there's
  // nothing to check proximity against.
  if (!tweet.latitude || !tweet.longitude) {
    return res.status(422).json({
      error: 'This complaint has no location on record yet, so it cannot be voted on.',
    });
  }

  // User location is mandatory for voting — that's the entire mechanism.
  if (userLat == null || userLon == null) {
    return res.status(400).json({
      error: 'Location access is required to vote on this complaint.',
      requiresLocation: true,
    });
  }

  const uLat = parseFloat(userLat);
  const uLon = parseFloat(userLon);

  if (!withinRadius(uLat, uLon, tweet.latitude, tweet.longitude, VOTE_RADIUS_KM)) {
    return res.status(403).json({
      error: 'Your location and the complaint location do not match. You must be within 500 m of the complaint to vote.',
      requiresLocation: true,
      locationMismatch: true,
    });
  }

  const update = {
    upvotes:   (tweet.upvotes   || 0) + (type === 'up'   ? 1 : 0),
    downvotes: (tweet.downvotes || 0) + (type === 'down' ? 1 : 0),
  };
  update.effective_severity = (tweet.severity_score || 0) + update.upvotes - update.downvotes;

  // Check escalation against the *post-vote* tallies
  Object.assign(update, await maybeEscalate({ ...tweet, ...update }));

  const { data: updated, error: updateErr } = await supabase
    .from('analyzed_tweets')
    .update(update)
    .eq('tweet_id', req.params.id)
    .select()
    .single();

  if (updateErr) return res.status(500).json({ error: updateErr.message });
  res.json(fmtTweet(updated));
});

//  API: Issues 
app.get('/api/issues', async (req, res) => {
  const { data, error } = await supabase
    .from('analyzed_tweets')
    .select('*')
    .order('effective_severity', { ascending: false })
    .limit(200);

  if (error) return res.status(500).json({ error: error.message });
  res.json(data.map(fmtTweet));
});

// POST /api/pipeline/trigger — manually force ACTIVE mode + NLP
app.post('/api/pipeline/trigger', (req, res) => {
  scheduler.forceActive();
  res.json({ ok: true, state: scheduler.state });
});

//  Test Media API ────────────────────────────────────────────────────────
// Lets a tester post a mock "tweet" with an explicit location from the
// Test Media page and see it processed and voteable immediately, without
// waiting on the Python/NLP pipeline or a real Twitter post.
app.post('/api/test/post', async (req, res) => {
  const { text, username, latitude, longitude, category, sentiment, engagement, isTroll } = req.body || {};

  if (!text || !text.trim()) return res.status(400).json({ error: 'Text is required' });
  if (latitude == null || longitude == null) {
    return res.status(400).json({ error: 'A location is required — pick one on the map or use current location.' });
  }

  const lat = parseFloat(latitude);
  const lon = parseFloat(longitude);
  const sent = Math.max(-1, Math.min(1, parseFloat(sentiment ?? 0)));
  const eng  = Math.max(0, Math.min(500, parseFloat(engagement ?? 0)));

  const cat = category && category !== 'auto' ? category : classifyCategory(text);
  const troll = typeof isTroll === 'boolean' ? { isTroll, score: isTroll ? 3 : 0 } : detectTroll(text);
  const severity = Math.round(((1 - sent) * 5 + Math.min(eng / 100, 5)) * 100) / 100;
  const locality = findNearestLocality(lat, lon);

  const doc = {
    tweet_id:           `test_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
    user_id:            `test_${Math.random().toString(36).slice(2, 8)}`,
    username:           (username && username.trim()) || randomTestUsername(),
    text:                text.trim(),
    cleaned_text:        text.trim().toLowerCase(),
    sentiment_score:     sent,
    severity_score:      severity,
    effective_severity:  severity,
    issue_category:      cat,
    bert_label:          sent >= 0 ? 'POSITIVE' : 'NEGATIVE',
    bert_score:          Math.min(0.5 + Math.abs(sent), 0.99),
    troll_flag:          troll.isTroll,
    troll_score:         troll.score,
    troll_reasons:       [],
    upvotes: 0, downvotes: 0, like_count: 0, retweet_count: 0,
    latitude: lat, longitude: lon, locality_name: locality,
    source:              'manual',
    forwarded:           severity >= ESCALATION_SEVERITY,
    forwarded_at:        severity >= ESCALATION_SEVERITY ? new Date().toISOString() : null,
    escalation_reason:   severity >= ESCALATION_SEVERITY ? 'high_severity' : null,
    timestamp:           new Date().toISOString(),
  };

  const { data: inserted, error } = await supabase
    .from('analyzed_tweets')
    .insert(doc)
    .select()
    .single();

  if (error) return res.status(500).json({ error: error.message });
  res.json(fmtTweet(inserted));
});

//  Admin API ──
app.post('/api/admin/login', (req, res) => {
  const { username, password } = req.body || {};
  if (username !== ADMIN_USERNAME || password !== ADMIN_PASSWORD) {
    return res.status(401).json({ error: 'Invalid credentials' });
  }
  const token = jwt.sign({ username }, JWT_SECRET, { expiresIn: JWT_EXPIRES_IN });
  res.json({ token, expiresIn: JWT_EXPIRES_IN });
});

app.get('/api/admin/complaints', requireAdmin, async (req, res) => {
  const { data, error } = await supabase
    .from('analyzed_tweets')
    .select('*')
    .order('created_at', { ascending: false })
    .limit(500);

  if (error) return res.status(500).json({ error: error.message });
  res.json(data.map(fmtAdminTweet));
});

app.get('/api/admin/stats', requireAdmin, async (req, res) => {
  const { data, error } = await supabase.from('analyzed_tweets').select('*');
  if (error) return res.status(500).json({ error: error.message });

  const total = data.length;
  const forwarded = data.filter(t => t.forwarded).length;
  const twitterCount = data.filter(t => (t.source || 'twitter') === 'twitter').length;
  const simulatedCount = data.filter(t => t.source === 'simulated').length;
  const manualCount = data.filter(t => t.source === 'manual').length;
  const trollCount = data.filter(t => t.troll_flag).length;
  const suggestedCount = data.filter(t => suggestForward(t).suggested_forward).length;
  const byCategory = {};
  for (const t of data) {
    const c = t.issue_category || 'other';
    byCategory[c] = (byCategory[c] || 0) + 1;
  }

  res.json({ total, forwarded, twitterCount, simulatedCount, manualCount, trollCount, suggestedCount, byCategory });
});

// Lets an admin forward any complaint regardless of its vote/severity
// standing. Simulated — same as auto-escalation, this only logs the
// forward on the record; no real email is sent from the server.
app.post('/api/admin/complaints/:id/forward', requireAdmin, async (req, res) => {
  const { data: tweet, error: fetchErr } = await supabase
    .from('analyzed_tweets')
    .select('*')
    .eq('tweet_id', req.params.id)
    .single();

  if (fetchErr || !tweet) return res.status(404).json({ error: 'Complaint not found' });

  const update = {
    forwarded: true,
    forwarded_at: new Date().toISOString(),
    escalation_reason: 'manual_admin',
  };

  const { data: updated, error } = await supabase
    .from('analyzed_tweets')
    .update(update)
    .eq('tweet_id', req.params.id)
    .select()
    .single();

  if (error) return res.status(500).json({ error: error.message });
  res.json(fmtAdminTweet(updated));
});

// Department inbox lookup for the admin "Forward" mail compose modal.
app.get('/api/admin/authority-email', requireAdmin, (req, res) => {
  res.json(CATEGORY_AUTHORITY_EMAIL);
});

//  Start ──
app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/api/health`);
  console.log(`Admin login: ${ADMIN_USERNAME} / (see .env for password)`);
});
// Start pipeline scheduler
scheduler.start();