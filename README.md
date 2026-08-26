# Project City

Project City is a civic issue monitoring platform that collects public complaints from social media, analyzes them using NLP techniques, and presents them through a dashboard with severity ranking, issue categorization, location-gated community voting, and automatic escalation.

Originally developed as part of the B.Tech Engineering Physics program at NIT Hamirpur.

## Features

* Hybrid data collection — real Twitter API with automatic fallback to a simulated complaint generator if Twitter is unavailable (auth error, exhausted free-tier quota, rate limit), so the pipeline keeps running unattended
* NLP-based sentiment and issue analysis
* Severity scoring and prioritization
* Issue categorization
* **Location-gated community voting** — each complaint is tagged with an approximate location; a vote only counts if the voter is within 500 m of it
* **Interactive map tab** (OpenStreetMap/Leaflet) — complaints plotted with a 500 m radius circle, color-coded by severity; filter by category, severity, location, or upvotes; hover 2 seconds over a circle for a detail tooltip
* **Test Media interface** — a mock social-feed composer (opens in its own tab) for posting dummy tweets with an explicit location, so voting/escalation can be tested end-to-end without waiting on the Twitter/NLP pipeline
* **Automatic escalation** — a complaint auto-forwards to the "concerned authority" (simulated) once it hits 20+ net upvotes or severity ≥ 8
* **Admin portal** (built into the same page) — authenticated view of every complaint including the reporting user, vote counts, source, and forwarding status
* **Demo mode** — a scripted, ~15-second click-to-play walkthrough of the entire complaint → analysis → voting → escalation flow, for presentations
* Automated pipeline orchestration

## Technology Stack

| Layer            | Technology                          |
| ---------------- | ------------------------------------ |
| Tweet Collection | Python, Tweepy                       |
| Fallback Data    | Python (mock_generator.py)           |
| NLP Processing   | TextBlob, DistilBERT, NLTK           |
| Database         | Supabase (PostgreSQL)                |
| Backend API      | Node.js, Express, JSON Web Tokens    |
| Frontend         | HTML, CSS, JavaScript, Leaflet/OSM   |
| Orchestration    | Python, Textual                      |

## Architecture

```text
Twitter API ──(fails?)──> Simulated Generator
    |                              |
    └──────────> raw_tweets (Supabase) <──────────┘
                        |
                        v
              NLP Processing Pipeline
             (+ locality resolution)
                        |
                        v
             analyzed_tweets (Supabase)
                        |
                        v
                  Express API
              (voting, escalation,
               admin auth)
                        |
                        v
         Dashboard (Issues / Map / Admin)
```

## Repository Structure

```text
project_city/
├── backend/
│   ├── server.js
│   ├── scheduler.js
│   ├── localities.js
│   ├── package.json
│   └── public/
│       └── index.html
├── nlp_pipeline/
│   ├── initial_fetch.py
│   ├── mock_generator.py
│   ├── NLProcessing.py
│   ├── localities.py
│   └── requirements.txt
├── orchestrator.py
├── supabase_schema.sql        (original — kept for reference)
├── supabase_schema_v2.sql     (migration: run this on an existing DB)
├── supabase_schema_new.sql    (fresh install: run this on a brand-new DB)
├── render.yaml
├── .env.example
└── .gitignore
```

## Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd project_city
```

### 2. Configure Supabase

* **Existing DB (already ran `supabase_schema.sql`)**: run `supabase_schema_v2.sql` in the SQL Editor — it only adds the new columns needed for location voting, source tagging, and escalation.
* **Brand-new DB**: run `supabase_schema_new.sql` instead.
* Disable RLS for the required tables if running locally.
* Copy the project URL and anon key.

### 3. Configure Environment Variables

Create a `.env` file from `.env.example`.

```env
SUPABASE_URL=
SUPABASE_KEY=
TWITTER_BEARER_TOKEN=
TWITTER_QUERY=
PORT=5500

ADMIN_USERNAME=admin
ADMIN_PASSWORD=CityDemo2026
JWT_SECRET=change-this-to-a-long-random-string
```

The admin credentials are intentionally shown on the login screen for hackathon demos — change them before any real deployment.

### 4. Install Dependencies

Backend:

```bash
cd backend
npm install
```

Python:

```bash
pip install -r nlp_pipeline/requirements.txt
pip install textual
```

## Running the Project

Start all components through the orchestrator:

```bash
python orchestrator.py
```

Services launched:

* Tweet Fetcher (with automatic fallback to the simulated generator)
* NLP Processor
* Backend API
* Frontend Dashboard

Dashboard:

```text
http://localhost:5500
```

Tabs: **Issues** (table + filters) · **Map** (OSM with 500m complaint circles) · **Admin** (login required).

## Hybrid Data Source

Every fetch cycle, `initial_fetch.py` tries the real Twitter API first and writes `nlp_pipeline/fetch_status.json` with the outcome. If it fails for any reason (invalid/expired token, exhausted free-tier quota, rate limit), `scheduler.js` immediately falls back to `mock_generator.py`, which inserts 1–3 realistic simulated complaints, geographically constrained to real localities within 5 km of NIT Hamirpur (see `nlp_pipeline/localities.py`). Every complaint carries a `source` field (`twitter` or `simulated`) shown as a badge throughout the UI, so nothing is presented as real that isn't.

This means the demo keeps producing fresh data for months without any manual intervention, even if the Twitter free tier stops working again.

## Location-Based Voting

* Real tweets are matched against a gazetteer of local areas by keyword; simulated complaints carry their location directly from the generator.
* A vote is only accepted if the voter's browser-reported location is within **500 m** of the complaint's location.
* Votes outside that radius are rejected with a clear message; voting is blocked entirely if the browser denies location access, or if a complaint has no resolved location yet.

## Auto-Escalation

A complaint is automatically marked `forwarded` (simulated — logged and visible in the admin portal, no real email/SMS is sent) when either:

* **Net upvotes** (upvotes − downvotes) reach **20**, or
* **Severity score** reaches **8** (out of 10) on arrival.

Thresholds are defined in both `nlp_pipeline/NLProcessing.py` and `backend/server.js` — keep them in sync if you change them.

## NLP Pipeline

The processing pipeline performs:

* Text cleaning
* Sentiment analysis using TextBlob
* Issue classification
* Severity estimation
* DistilBERT-based labeling
* Troll and spam detection
* **Locality resolution** (keyword match against the gazetteer, or passthrough for simulated data)

### Severity Score

```text
severity = (1 - sentiment) * 5 + min((likes + retweets) / 100, 5)
```

Score range:

```text
0 - 10
```

Higher values indicate higher urgency.

### Issue Categories

* Roads
* Water
* Electricity
* Waste
* Other

## Test Media Interface

Open `/test-media.html` (linked from the "Post Test Tweet" button on the main dashboard, opens in a new tab) to post a mock tweet directly:

* Type the complaint text, optionally set a username
* Pick a location by choosing from the locality list, clicking on a small map, or using your browser's current location
* Optionally override the category, adjust the sentiment/engagement sliders (these drive the severity score using the same formula as the real pipeline), or force a troll flag for testing
* Submitting processes the post instantly (category, severity, location, troll check) via a lightweight classifier built into the Node backend — no dependency on the Python/DistilBERT pipeline being up — and it appears immediately in the Issues tab, Map tab, and Admin portal, tagged with a "Test / Manual" source badge

This is the fastest way to test the 500 m voting radius and the auto-escalation threshold without waiting for real tweets or the simulated generator.

## API Endpoints

| Method | Endpoint                  | Description                                    | Auth  |
| ------ | ------------------------- | ----------------------------------------------- | ----- |
| GET    | `/api/health`             | Service health                                  | —     |
| GET    | `/api/pipeline/status`    | Pipeline status                                 | —     |
| GET    | `/api/localities`         | Gazetteer used by the map tab                   | —     |
| GET    | `/api/tweets`             | All analyzed tweets                             | —     |
| GET    | `/api/tweets/:id`         | Single tweet                                    | —     |
| POST   | `/api/tweets/:id/vote`    | Vote on issue (500 m location check enforced)   | —     |
| GET    | `/api/issues`             | Issues sorted by severity                       | —     |
| POST   | `/api/test/post`          | Post a test tweet via the Test Media interface  | —     |
| POST   | `/api/pipeline/trigger`   | Manually force ACTIVE mode                      | —     |
| POST   | `/api/admin/login`        | Admin login → returns a JWT                     | —     |
| GET    | `/api/admin/complaints`   | All complaints incl. reporting user             | Admin |
| GET    | `/api/admin/stats`        | Aggregate counts for the admin dashboard        | Admin |

## Deployment

The project is configured for deployment on Render.

Required environment variables:

```env
SUPABASE_URL
SUPABASE_KEY
TWITTER_BEARER_TOKEN
PORT
ADMIN_USERNAME
ADMIN_PASSWORD
JWT_SECRET
```

## Troubleshooting

**`ModuleNotFoundError: No module named 'websockets.asyncio'`** (Python fetcher/NLP)
`supabase-py`'s realtime dependency needs a newer `websockets` than may already be installed on your machine. Fix:
```bash
pip install --upgrade "websockets>=13" --force-reinstall
```
This is already pinned in `requirements.txt` for fresh installs.

**`MODULE_NOT_FOUND` in the Node server** (backend crashes on boot)
Your `node_modules` is out of sync with `package.json` (usually after pulling an update that added a dependency, e.g. `jsonwebtoken` for admin auth). Fix:
```bash
cd backend
npm install
```

**Supabase SQL editor shows "no rows returned" after running a migration**
That's expected — `supabase_schema_v2.sql` is entirely `ALTER TABLE` / `CREATE INDEX` statements, which don't return rows even on success. Confirm it worked by running:
```sql
select column_name from information_schema.columns where table_name = 'analyzed_tweets';
```
You should see `forwarded`, `source`, `locality_name`, `forwarded_at`, and `escalation_reason` in the result.

## License

MIT License
