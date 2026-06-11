# Project City

Project City is a civic issue monitoring platform that collects public complaints from social media, analyzes them using NLP techniques, and presents them through a dashboard with severity ranking, issue categorization, and community voting.

Originally developed as part of the B.Tech Engineering Physics program at NIT Hamirpur.

## Features

* Automated collection of civic complaint tweets
* NLP-based sentiment and issue analysis
* Severity scoring and prioritization
* Issue categorization
* Community voting system
* Interactive dashboard
* Automated pipeline orchestration

## Technology Stack

| Layer            | Technology                 |
| ---------------- | -------------------------- |
| Tweet Collection | Python, Tweepy             |
| NLP Processing   | TextBlob, DistilBERT, NLTK |
| Database         | Supabase (PostgreSQL)      |
| Backend API      | Node.js, Express           |
| Frontend         | HTML, CSS, JavaScript      |
| Orchestration    | Python, Textual            |

## Architecture

```text
Twitter API
    |
    v
Tweet Fetcher
    |
    v
raw_tweets (Supabase)
    |
    v
NLP Processing Pipeline
    |
    v
analyzed_tweets (Supabase)
    |
    v
Express API
    |
    v
Dashboard
```

## Repository Structure

```text
project_city/
├── backend/
│   ├── server.js
│   ├── scheduler.js
│   ├── package.json
│   └── public/
│       └── index.html
├── nlp_pipeline/
│   ├── initial_fetch.py
│   ├── NLProcessing.py
│   └── requirements.txt
├── orchestrator.py
├── supabase_schema.sql
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

1. Create a Supabase project.
2. Run `supabase_schema.sql` in the SQL Editor.
3. Disable RLS for the required tables if running locally.
4. Copy the project URL and anon key.

### 3. Configure Environment Variables

Create a `.env` file from `.env.example`.

```env
SUPABASE_URL=
SUPABASE_KEY=
TWITTER_BEARER_TOKEN=
TWITTER_QUERY=
PORT=5500
```

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

* Tweet Fetcher
* NLP Processor
* Backend API
* Frontend Dashboard

Dashboard:

```text
http://localhost:5500
```

## NLP Pipeline

The processing pipeline performs:

* Text cleaning
* Sentiment analysis using TextBlob
* Issue classification
* Severity estimation
* DistilBERT-based labeling
* Troll and spam detection

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

## API Endpoints

| Method | Endpoint               | Description               |
| ------ | ---------------------- | ------------------------- |
| GET    | `/api/health`          | Service health            |
| GET    | `/api/pipeline/status` | Pipeline status           |
| GET    | `/api/tweets`          | All analyzed tweets       |
| GET    | `/api/tweets/:id`      | Single tweet              |
| POST   | `/api/tweets/:id/vote` | Vote on issue             |
| GET    | `/api/issues`          | Issues sorted by severity |

## Deployment

The project is configured for deployment on Render.

Required environment variables:

```env
SUPABASE_URL
SUPABASE_KEY
TWITTER_BEARER_TOKEN
PORT
```

## License

MIT License
