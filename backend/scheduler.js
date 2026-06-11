/**
 * scheduler.js
 * Manages the tweet fetcher and NLP pipeline as child processes.
 *
 * States:
 *   DORMANT → checks for new tweets every 6 hours
 *   ACTIVE  → checks every 10 minutes, NLP runs alongside
 *
 * DORMANT → ACTIVE : new tweet found
 * ACTIVE  → DORMANT: 24 hours pass with no new tweets
 */

const { spawn } = require('child_process');
const path = require('path');
const { createClient } = require('@supabase/supabase-js');

const ROOT = path.join(__dirname, '..');
const FETCHER = path.join(ROOT, 'nlp_pipeline', 'initial_fetch.py');
const NLP     = path.join(ROOT, 'nlp_pipeline', 'NLProcessing.py');

const DORMANT_INTERVAL  = 6 * 60 * 60 * 1000;  // 6 hours
const ACTIVE_INTERVAL   = 10 * 60 * 1000;       // 10 minutes
const ACTIVE_TIMEOUT    = 24 * 60 * 60 * 1000;  // 24 hours no new tweets → dormant

const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY);

//  Exported state (server.js reads this for /api/pipeline/status) ─
const state = {
  mode:          'DORMANT',   // 'DORMANT' | 'ACTIVE'
  fetcherPid:    null,
  nlpPid:        null,
  lastNewTweet:  null,        // ISO timestamp of last new tweet found
  lastCheck:     null,        // ISO timestamp of last fetch attempt
  cyclesSinceNew: 0,          // active cycles with no new tweets
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

//  Process management ─
function spawnPython(script, label) {
  const proc = spawn('python', [script], {
    env: { ...process.env },
    stdio: ['ignore', 'pipe', 'pipe']
  });

  proc.stdout.on('data', d => log(`[${label}] ${d.toString().trim()}`));
  proc.stderr.on('data', d => log(`[${label}:err] ${d.toString().trim()}`));
  proc.on('exit', code => log(`[${label}] exited (code ${code})`));

  log(` Started ${label} (pid ${proc.pid})`);
  return proc;
}

function killProc(proc, label) {
  if (!proc || proc.killed) return;
  proc.kill('SIGTERM');
  log(`⏹ Stopped ${label}`);
}

//  State transitions ──
function goActive() {
  if (state.mode === 'ACTIVE') return;
  log('🟢 → ACTIVE');
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
  log('⚪ → DORMANT');
  state.mode = 'DORMANT';

  killProc(nlpProc, 'NLP');
  nlpProc = null;
  state.nlpPid = null;

  clearInterval(timer);
  timer = setInterval(dormantCycle, DORMANT_INTERVAL);
}

//  Cycles ─

// Run a single fetch pass and return whether any new tweets were found
async function runFetchPass() {
  const before = await countRawTweets();
  state.lastCheck = new Date().toISOString();

  // Spawn fetcher, let it run one pass (it exits after one 15-min rate-limit cycle)
  await new Promise(resolve => {
    const proc = spawnPython(FETCHER, 'fetcher');
    fetcherProc = proc;
    state.fetcherPid = proc.pid;
    proc.on('exit', resolve);
  });

  fetcherProc = null;
  state.fetcherPid = null;

  const after = await countRawTweets();
  if (before === null || after === null) return false;

  const found = after > before;
  if (found) {
    state.lastNewTweet = new Date().toISOString();
    log(` ${after - before} new tweet(s) found`);
  } else {
    log('  No new tweets this pass');
  }
  return found;
}

async function dormantCycle() {
  log(' Dormant check...');
  const found = await runFetchPass();
  if (found) goActive();
}

async function activeCycle() {
  log(' Active check...');
  const found = await runFetchPass();

  if (found) {
    state.cyclesSinceNew = 0;
  } else {
    state.cyclesSinceNew++;
    const hoursIdle = (state.cyclesSinceNew * ACTIVE_INTERVAL) / 3600000;
    log(` No new tweets for ${hoursIdle.toFixed(1)}h`);
    if (state.cyclesSinceNew * ACTIVE_INTERVAL >= ACTIVE_TIMEOUT) {
      goDormant();
    }
  }
}

//  Boot 
function start() {
  log(' Scheduler starting in DORMANT mode');
  // Run an immediate dormant check on boot, then settle into interval
  dormantCycle();
  timer = setInterval(dormantCycle, DORMANT_INTERVAL);
}

module.exports = { start, state };
