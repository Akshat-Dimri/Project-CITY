-- version 3 : location voting, hybrid source tagging, escalation tracking --
-- Use this for a brand-new Supabase project.
-- If you already have v2 tables set up, use supabase_schema_v2.sql instead.

-- Raw tweets (from Twitter fetcher or the simulated fallback generator)
create table if not exists raw_tweets (
  id             bigserial primary key,
  tweet_id       text unique not null,
  user_id        text,
  username       text,
  text           text,
  timestamp      text,
  like_count     int default 0,
  retweet_count  int default 0,
  source         text default 'twitter',   -- 'twitter' | 'simulated'
  latitude       float,                    -- set by the simulator; real tweets are geo-tagged during NLP
  longitude      float,
  locality_name  text,
  created_at     timestamptz default now()
);

-- Analyzed tweets (written by NLP pipeline, read by backend)
create table if not exists analyzed_tweets (
  id                  bigserial primary key,
  tweet_id            text unique not null,
  user_id             text,
  username            text,
  text                text,
  cleaned_text        text,
  sentiment_score     float,
  severity_score      float default 0,
  effective_severity  float default 0,
  issue_category      text,
  bert_label          text,
  bert_score          float,
  troll_flag          boolean default false,
  troll_score         int default 0,
  troll_reasons       text[],
  upvotes             int default 0,
  downvotes           int default 0,
  like_count          int default 0,
  retweet_count       int default 0,
  latitude            float,
  longitude           float,
  locality_name       text,
  source              text default 'twitter',  -- 'twitter' | 'simulated'
  forwarded           boolean default false,
  forwarded_at        timestamptz,
  escalation_reason   text,                     -- 'vote_threshold' | 'high_severity'
  timestamp           text,
  created_at          timestamptz default now()
);

-- Indexes for fast ordering / filtering
create index if not exists idx_severity  on analyzed_tweets (severity_score desc);
create index if not exists idx_effective on analyzed_tweets (effective_severity desc);
create index if not exists idx_forwarded on analyzed_tweets (forwarded);
create index if not exists idx_source    on analyzed_tweets (source);
create index if not exists idx_locality  on analyzed_tweets (locality_name);
