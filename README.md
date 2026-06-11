# Project City — Civic Issues Tracker

A full-stack portal that pulls civic complaint tweets, runs NLP analysis, and displays them on a dashboard with voting and severity ranking.

## Stack

| Layer | Tech |
|---|---|
| Tweet fetch | Python + Tweepy |
| NLP pipeline | Python — TextBlob, DistilBERT (HuggingFace), troll detection |
| Database | Supabase (PostgreSQL) |
| Backend API | Node.js + Express |
| Frontend | Vanilla HTML/CSS/JS |
| Orchestrator | Python + Textual TUI |

## Structure

```
project_city/
├── backend/
│   ├── server.js          # Express API (Supabase-backed)
│   ├── package.json
│   └── public/index.html  # Dashboard frontend
├── nlp_pipeline/
│   ├── initial_fetch.py   # Pulls tweets from Twitter → Supabase
│   ├── NLProcessing.py    # Sentiment + severity + troll detection
│   └── requirements.txt
├── orchestrator.py         # TUI to run all components together
├── supabase_schema.sql     # Run this once in Supabase SQL editor
├── .env.example
└── .gitignore
```

## Setup

### 1. Supabase
1. Create a project at [supabase.com](https://supabase.com)
2. Run `supabase_schema.sql` in the SQL editor
3. Copy your project URL and anon key

### 2. Twitter / X
1. Get a Bearer Token from [developer.x.com](https://developer.x.com)
2. Basic (free) tier works for search

### 3. Environment
```bash
cp .env.example .env
# Fill in SUPABASE_URL, SUPABASE_KEY, TWITTER_BEARER_TOKEN
```

### 4. Install
```bash
# Node backend
cd backend && npm install

# Python pipeline
pip install -r nlp_pipeline/requirements.txt
```

### 5. Run

**All at once (TUI):**
```bash
pip install textual
python orchestrator.py
```

**Individually:**
```bash
# Terminal 1 — tweet fetcher
python nlp_pipeline/initial_fetch.py

# Terminal 2 — NLP processor
python nlp_pipeline/NLProcessing.py

# Terminal 3 — backend + dashboard
cd backend && node server.js
# Dashboard at http://localhost:5500
```

## Health Check

`GET /api/health` returns the live status of Supabase and Twitter connections without taking the portal down.

## My Contribution

- Wrote the NLP pipeline (`NLProcessing.py`) — sentiment analysis, severity scoring, issue classification, troll detection
- Wrote the tweet fetcher (`initial_fetch.py`) and orchestrator
- Frontend built with AI tools; product decisions (category taxonomy, severity formula, geo-voting logic) made by me
