/**
 * backfill-locations.js
 *
 * One-off fix for complaints that were analyzed before they had a resolved
 * location (analyzed_tweets.latitude/longitude are null) — this happens
 * whenever a real tweet's text doesn't contain one of the exact keyword
 * phrases in localities.js's strict gazetteer match.
 *
 * This script re-attempts a *looser*, best-effort keyword match per row,
 * and if that still finds nothing, falls back to the NIT Hamirpur campus
 * center so every complaint ends up plottable on the map. It only touches
 * rows that currently have no location — nothing else is changed, and the
 * live NLP pipeline's stricter matching (localities.py) is untouched, so
 * new complaints are matched the same way as before.
 *
 * Run once from backend/:
 *   node scripts/backfill-locations.js
 */
require('dotenv').config({ path: require('path').join(__dirname, '../.env') });
const { createClient } = require('@supabase/supabase-js');
const { LOCALITIES, CENTER } = require('../localities');

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_KEY = process.env.SUPABASE_KEY;
if (!SUPABASE_URL || !SUPABASE_KEY) {
  console.error('SUPABASE_URL or SUPABASE_KEY missing in backend/.env');
  process.exit(1);
}
const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

// Broader, best-effort keywords per locality — used ONLY by this backfill,
// separate from the stricter list the live pipeline uses for new tweets.
const GUESS_KEYWORDS = {
  'NIT Hamirpur Campus':            ['nit', 'campus', 'hostel', 'college'],
  'Degree College Chowk':           ['chowk', 'bus stand'],
  'Hamirpur Town Center':           ['town', 'bazaar', 'market', 'ward'],
  'Green Park Colony':              ['green park'],
  'Patel Nagar':                    ['patel nagar', 'nagar'],
  'Vegetable Market (Sabzi Mandi)': ['mandi', 'vegetable', 'sabzi'],
  'District Library Area':          ['library'],
  'Sarahkar':                       ['sarahkar'],
  'Majhog Sultani':                 ['majhog'],
  'Daruhi':                         ['daruhi'],
  'NH-88 Bypass Road':              ['highway', 'bypass', 'nh88', 'nh 88', 'nh-88'],
};

function guessLocality(text) {
  const t = (text || '').toLowerCase();
  for (const loc of LOCALITIES) {
    const kws = GUESS_KEYWORDS[loc.name] || [];
    if (kws.some(kw => t.includes(kw))) return loc;
  }
  return null;
}

const round6 = n => Math.round(n * 1e6) / 1e6;
const jitter = () => (Math.random() - 0.5) * 0.004; // ~±200m, so pins don't stack exactly

async function run() {
  const { data: rows, error } = await supabase
    .from('analyzed_tweets')
    .select('id, tweet_id, text')
    .or('latitude.is.null,longitude.is.null');

  if (error) {
    console.error('Fetch failed:', error.message);
    process.exit(1);
  }

  console.log(`Found ${rows.length} complaint(s) with no location.\n`);
  let updated = 0, defaulted = 0;

  for (const row of rows) {
    const guess = guessLocality(row.text);
    const loc = guess || CENTER; // fall back to campus center if nothing matches at all
    if (!guess) defaulted++;

    const { error: updErr } = await supabase
      .from('analyzed_tweets')
      .update({
        latitude: round6(loc.lat + jitter()),
        longitude: round6(loc.lon + jitter()),
        locality_name: loc.name,
      })
      .eq('id', row.id);

    if (updErr) {
      console.warn(`  ✗ ${row.tweet_id}: ${updErr.message}`);
    } else {
      updated++;
      console.log(`  ✓ ${row.tweet_id} -> ${loc.name}${guess ? '' : ' (default fallback — no keyword match)'}`);
    }
  }

  console.log(`\nDone. ${updated}/${rows.length} updated (${defaulted} used the default fallback location).`);
  console.log('Reload the dashboard — the Map tab should now show pins and circles for these.');
}

run();