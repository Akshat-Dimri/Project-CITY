/**
 * scheduler.js
 * Manages the tweet fetcher, simulated-data fallback, and NLP pipeline as
 * child processes.
 *
 * States:
 *   DORMANT → checks for new tweets every 6 hours
 *   ACTIVE  → checks every 10 minutes, NLP runs alongside
 *
 * DORMANT → ACTIVE : new tweet found
 * ACTIVE  → DORMANT: 24 hours pass with no new tweets
 *
 * Hybrid data source:
 *   Every cycle, the real Twitter fetcher (initial_fetch.py) is tried
 *   first. It writes fetch_status.json after each attempt. If it failed
 *   (auth error, exhausted free-tier quota, rate limit — anything that
 *   isn't a clean "ok"), the scheduler immediately falls back to
 *   mock_generator.py so the demo/app keeps producing fresh complaints
 *   without any manual intervention. Every complaint is tagged with its
 *   real source ('twitter' | 'simulated') so the portal can show which
 *   is which.
 */

const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const { createClient } = require('@supabase/supabase-js');

const ROOT     = path.join(__dirname, '..');
const FETCHER  = path.join(ROOT, 'nlp_pipeline', 'initial_fetch.py');
const MOCKGEN  = path.join(ROOT, 'nlp_pipeline', 'mock_generator.py');
const NLP      = path.join(ROOT, 'nlp_pipeline', 'NLProcessing.py');
const STATUS_FILE = path.join(ROOT, 'nlp_pipeline', 'fetch_status.json');

const DORMANT_INTERVAL  = 6 * 60 * 60 * 1000;  // 6 hours
const ACTIVE_INTERVAL   = 10 * 60 * 1000;       // 10 minutes
const ACTIVE_TIMEOUT    = 24 * 60 * 60 * 1000;  // 24 hours no new tweets → dormant

const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY);

//  Exported state (server.js reads this for /api/pipeline/status) ─
const state = {
  mode:            'DORMANT',   // 'DORMANT' | 'ACTIVE'
  fetcherPid:      null,
  nlpPid:          null,
  lastNewTweet:    null,        // ISO timestamp of last new tweet found
  lastCheck:       null,        // ISO timestamp of last fetch attempt
  lastSource:      null,        // 'twitter' | 'simulated' — which produced the last successful pass
  lastFetchReason: null,        // reason string from fetch_status.json (e.g. 'ok', 'quota_or_access_level')
  cyclesSinceNew:  0,           // active cycles with no new tweets
};

let fetcherProc = null;
let nlpProc     = null;
let timer       = null;

//  Helpers 
function log(msg) {
  console.log(`[scheduler] ${new Date().toISOString()} — ${msg}`);
}

async function countRawTweets() {
  const { count, error } = await supabase
    .from('raw_tweets')
    .select('*', { count: 'exact', head: true });
  if (error) { log(`  count error: ${error.message}`); return null; }
  return count;
}

function readFetchStatus() {
  try {
    const raw = fs.readFileSync(STATUS_FILE, 'utf8');
    return JSON.parse(raw);
  } catch (e) {
    return null; // no status file yet, or unreadable — treat as unknown
  }
}

//  Process management ─
function runPythonOnce(script, label) {
  return new Promise(resolve => {
    const proc = spawn('python3', [script], {
      env: { ...process.env },
      stdio: ['ignore', 'pipe', 'pipe']
    });

    if (label === 'fetcher') {
      fetcherProc = proc;
      state.fetcherPid = proc.pid;
    }

    proc.on('error', err => { log(`[${label}] SPAWN ERROR: ${err.message}`); resolve(); });
    proc.stdout.on('data', d => log(`[${label}] ${d.toString().trim()}`));
    proc.stderr.on('data', d => log(`[${label}:err] ${d.toString().trim()}`));
    proc.on('exit', code => {
      log(`[${label}] exited (code ${code})`);
      if (label === 'fetcher') { fetcherProc = null; state.fetcherPid = null; }
      resolve();
    });

    log(`Started ${label} (pid ${proc.pid})`);
  });
}

function spawnPython(script, label) {
  const proc = spawn('python3', [script], {
    env: { ...process.env },
    stdio: ['ignore', 'pipe', 'pipe']
  });

  proc.on('error', err => log(`[${label}] SPAWN ERROR: ${err.message}`));
  proc.stdout.on('data', d => log(`[${label}] ${d.toString().trim()}`));
  proc.stderr.on('data', d => log(`[${label}:err] ${d.toString().trim()}`));
  proc.on('exit', code => log(`[${label}] exited (code ${code})`));

  log(`Started ${label} (pid ${proc.pid})`);
  return proc;
}

function killProc(proc, label) {
  if (!proc || proc.killed) return;
  proc.kill('SIGTERM');
  log(`Stopped ${label}`);
}

//  State transitions ──
function goActive() {
  if (state.mode === 'ACTIVE') return;
  log('→ ACTIVE');
  state.mode = 'ACTIVE';
  state.cyclesSinceNew = 0;

  // Start NLP if not running
  if (!nlpProc || nlpProc.killed) {
    nlpProc = spawnPython(NLP, 'NLP');
    state.nlpPid = nlpProc.pid;
  }

  clearInterval(timer);
  timer = setInterval(activeCycle, ACTIVE_INTERVAL);
}

function goDormant() {
  if (state.mode === 'DORMANT') return;
  log('→ DORMANT');
  state.mode = 'DORMANT';

  killProc(nlpProc, 'NLP');
  nlpProc = null;
  state.nlpPid = null;

  clearInterval(timer);
  timer = setInterval(dormantCycle, DORMANT_INTERVAL);
}

//  Cycles ─

// Run a single fetch pass (real Twitter, falling back to the simulated
// generator on failure) and return whether any new tweets were found.
async function runFetchPass() {
  const before = await countRawTweets();
  if (before === null) {
    log('ERROR: Could not count tweets before fetch — skipping pass');
    return false;
  }

  state.lastCheck = new Date().toISOString();

  // 1. Try the real Twitter fetcher
  await runPythonOnce(FETCHER, 'fetcher');
  let status = readFetchStatus();

  // 2. If it failed for any reason, fall back to the simulated generator
  if (!status || status.ok === false) {
    const reason = status ? status.reason : 'no_status_file';
    log(`WARNING: Twitter fetch unavailable (${reason}) — falling back to simulated data`);
    await runPythonOnce(MOCKGEN, 'mock-generator');
    status = readFetchStatus();
  }

  state.lastFetchReason = status ? status.reason : 'unknown';
  state.lastSource = status ? status.source : null;

  const after = await countRawTweets();
  if (after === null) {
    log('ERROR: Could not count tweets after fetch');
    return false;
  }

  const found = after > before;
  if (found) {
    state.lastNewTweet = new Date().toISOString();
    log(`${after - before} new tweet(s) found (source: ${state.lastSource})`);
  } else {
    log('No new tweets this pass');
  }
  return found;
}

async function dormantCycle() {
  log('Dormant check...');
  const found = await runFetchPass();
  if (found) goActive();
}

async function activeCycle() {
  log('Active check...');
  const found = await runFetchPass();

  if (found) {
    state.cyclesSinceNew = 0;
  } else {
    state.cyclesSinceNew++;
    const hoursIdle = (state.cyclesSinceNew * ACTIVE_INTERVAL) / 3600000;
    log(`No new tweets for ${hoursIdle.toFixed(1)}h`);
    if (state.cyclesSinceNew * ACTIVE_INTERVAL >= ACTIVE_TIMEOUT) {
      goDormant();
    }
  }
}

//  Boot 
function start() {
  log('Scheduler starting in DORMANT mode');
  // Run an immediate dormant check on boot, then settle into interval
  dormantCycle();
  timer = setInterval(dormantCycle, DORMANT_INTERVAL);
}

function forceActive() {
  log('Manual trigger — forcing ACTIVE mode');
  state.lastNewTweet = new Date().toISOString();
  goActive();
}

module.exports = { start, state, forceActive };
