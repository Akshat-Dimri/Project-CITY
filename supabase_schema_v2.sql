-- supabase_schema_v2.sql
-- Migration on top of supabase_schema.sql.
-- Run this once in the Supabase SQL editor against your existing project.
-- All statements are idempotent (safe to re-run).

-- raw_tweets: capture source + username, and let simulated complaints
-- carry their location straight from the generator.
alter table raw_tweets add column if not exists username       text;
alter table raw_tweets add column if not exists source         text default 'twitter';
alter table raw_tweets add column if not exists latitude       float;
alter table raw_tweets add column if not exists longitude      float;
alter table raw_tweets add column if not exists locality_name  text;

-- analyzed_tweets: username/source passthrough + escalation tracking.
alter table analyzed_tweets add column if not exists username           text;
alter table analyzed_tweets add column if not exists source             text default 'twitter';
alter table analyzed_tweets add column if not exists locality_name      text;
alter table analyzed_tweets add column if not exists forwarded          boolean default false;
alter table analyzed_tweets add column if not exists forwarded_at       timestamptz;
alter table analyzed_tweets add column if not exists escalation_reason  text;

-- Helpful indexes for the new admin/map views.
create index if not exists idx_forwarded  on analyzed_tweets (forwarded);
create index if not exists idx_source     on analyzed_tweets (source);
create index if not exists idx_locality   on analyzed_tweets (locality_name);
