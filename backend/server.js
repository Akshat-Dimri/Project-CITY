require('dotenv').config();
const express = require('express');
const cors = require('cors');
const path = require('path');
const { createClient } = require('@supabase/supabase-js');
const scheduler = require('./scheduler');

const app = express();
const PORT = process.env.PORT || 5500;

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// ─── Supabase Client ───────────────────────────────────────────────────────────
const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_KEY = process.env.SUPABASE_KEY;

if (!SUPABASE_URL || !SUPABASE_KEY) {
  console.error('❌ SUPABASE_URL or SUPABASE_KEY missing in .env');
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

// ─── Config Health State ──────────────────────────────────────────────────────
// Tracks whether credentials are working; doesn't crash the server if they fail.
const health = {
  supabase: { ok: false, lastCheck: null, error: null },
  twitter: { ok: false, lastCheck: null, error: null }
};

async function checkSupabaseHealth() {
  try {
    const { error } = await supabase.from('analyzed_tweets').select('id').limit(1);
    health.supabase.ok = !error;
    health.supabase.error = error ? error.message : null;
    if (error) console.warn(`⚠️  Supabase issue: ${error.message}`);
  } catch (e) {
    health.supabase.ok = false;
    health.supabase.error = e.message;
    console.warn(`⚠️  Supabase unreachable: ${e.message}`);
  }
  health.supabase.lastCheck = new Date().toISOString();
}

// Check on start + every 10 minutes
checkSupabaseHealth();
setInterval(checkSupabaseHealth, 10 * 60 * 1000);

// ─── Helpers ──────────────────────────────────────────────────────────────────
function withinRadius(uLat, uLon, iLat, iLon, km = 5) {
  const toRad = d => d * Math.PI / 180;
  const R = 6371;
  const dLat = toRad(iLat - uLat), dLon = toRad(iLon - uLon);
  const a = Math.sin(dLat/2)**2 + Math.cos(toRad(uLat)) * Math.cos(toRad(iLat)) * Math.sin(dLon/2)**2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a)) <= km;
}

function fmtTweet(t) {
  const base = t.severity_score || 0;
  return {
    id:               String(t.tweet_id || t.id),
    text:             t.text,
    severity_score:   base,
    effective_severity: base + (t.upvotes || 0) - (t.downvotes || 0),
    sentiment_score:  t.sentiment_score,
    issue_category:   t.issue_category,
    bert_label:       t.bert_label,
    troll_flag:       t.troll_flag,
    upvotes:          t.upvotes || 0,
    downvotes:        t.downvotes || 0,
    like_count:       t.like_count || 0,
    retweet_count:    t.retweet_count || 0,
    timestamp:        t.timestamp,
    latitude:         t.latitude,
    longitude:        t.longitude,
    has_location:     !!(t.latitude && t.longitude),
  };
}

// ─── API: Health ──────────────────────────────────────────────────────────────
app.get("/api/health", (req, res) => res.json(health));

// ─── API: Pipeline status ────────────────────────────────────────────────────
app.get("/api/pipeline/status", (req, res) => res.json(scheduler.state));

// ─── API: Get all tweets ──────────────────────────────────────────────────────
app.get('/api/tweets', async (req, res) => {
  const { data, error } = await supabase
    .from('analyzed_tweets')
    .select('*')
    .order('severity_score', { ascending: false })
    .limit(100);

  if (error) return res.status(500).json({ error: error.message });
  res.json(data.map(fmtTweet));
});

// ─── API: Single tweet ────────────────────────────────────────────────────────
app.get('/api/tweets/:id', async (req, res) => {
  const { data, error } = await supabase
    .from('analyzed_tweets')
    .select('*')
    .eq('tweet_id', req.params.id)
    .single();

  if (error || !data) return res.status(404).json({ error: 'Tweet not found' });
  res.json(fmtTweet(data));
});

// ─── API: Vote ────────────────────────────────────────────────────────────────
app.post('/api/tweets/:id/vote', async (req, res) => {
  const { type, userLat, userLon } = req.body;
  if (!['up', 'down'].includes(type)) return res.status(400).json({ error: 'Invalid vote type' });

  const { data: tweet, error: fetchErr } = await supabase
    .from('analyzed_tweets')
    .select('*')
    .eq('tweet_id', req.params.id)
    .single();

  if (fetchErr || !tweet) return res.status(404).json({ error: 'Tweet not found' });

  // Geo-check if both sides have coordinates
  if (userLat && userLon && tweet.latitude && tweet.longitude) {
    if (!withinRadius(parseFloat(userLat), parseFloat(userLon), tweet.latitude, tweet.longitude)) {
      return res.status(403).json({ error: 'Must be within 5km to vote', requiresLocation: true });
    }
  }

  const update = {
    upvotes:   (tweet.upvotes   || 0) + (type === 'up'   ? 1 : 0),
    downvotes: (tweet.downvotes || 0) + (type === 'down' ? 1 : 0),
  };

  // First voter sets the location
  if (userLat && userLon && !tweet.latitude) {
    update.latitude  = parseFloat(userLat);
    update.longitude = parseFloat(userLon);
  }

  update.effective_severity = (tweet.severity_score || 0) + update.upvotes - update.downvotes;

  const { data: updated, error: updateErr } = await supabase
    .from('analyzed_tweets')
    .update(update)
    .eq('tweet_id', req.params.id)
    .select()
    .single();

  if (updateErr) return res.status(500).json({ error: updateErr.message });
  res.json(fmtTweet(updated));
});

// ─── API: Issues (alias used by frontend map) ─────────────────────────────────
app.get('/api/issues', async (req, res) => {
  const { data, error } = await supabase
    .from('analyzed_tweets')
    .select('*')
    .order('effective_severity', { ascending: false })
    .limit(200);

  if (error) return res.status(500).json({ error: error.message });
  res.json(data.map(fmtTweet));
});

// ─── Start ────────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`🚀 Server running at http://localhost:${PORT}`);
  console.log(`📊 Health check: http://localhost:${PORT}/api/health`);
});
// Start pipeline scheduler
scheduler.start();
