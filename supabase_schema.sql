-- Run this in your Supabase SQL editor to set up the two required tables.

-- Raw tweets (from Twitter fetcher)
create table if not exists raw_tweets (
  id           bigserial primary key,
  tweet_id     text unique not null,
  user_id      text,
  text         text,
  timestamp    text,
  like_count   int default 0,
  retweet_count int default 0,
  created_at   timestamptz default now()
);

-- Analyzed tweets (written by NLP pipeline, read by backend)
create table if not exists analyzed_tweets (
  id                bigserial primary key,
  tweet_id          text unique not null,
  user_id           text,
  text              text,
  cleaned_text      text,
  sentiment_score   float,
  severity_score    float default 0,
  effective_severity float default 0,
  issue_category    text,
  bert_label        text,
  bert_score        float,
  troll_flag        boolean default false,
  troll_score       int default 0,
  troll_reasons     text[],
  upvotes           int default 0,
  downvotes         int default 0,
  like_count        int default 0,
  retweet_count     int default 0,
  latitude          float,
  longitude         float,
  timestamp         text,
  created_at        timestamptz default now()
);

-- Index for fast ordering
create index if not exists idx_severity on analyzed_tweets (severity_score desc);
create index if not exists idx_effective on analyzed_tweets (effective_severity desc);
