#  Imports ─
import os
import json
import time
import logging
import tweepy
from datetime import datetime, timezone
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

#  Logging ─
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

#  Config
BEARER_TOKEN  = os.getenv("TWITTER_BEARER_TOKEN")
SUPABASE_URL  = os.getenv("SUPABASE_URL")
SUPABASE_KEY  = os.getenv("SUPABASE_KEY")
SEARCH_QUERY  = os.getenv("TWITTER_QUERY", "#CITYHamirpur -is:retweet")

STATUS_FILE = os.path.join(os.path.dirname(__file__), "fetch_status.json")

if not SUPABASE_URL or not SUPABASE_KEY:
    logging.error("SUPABASE_URL or SUPABASE_KEY not set in .env")
    raise SystemExit(1)

#  Clients
db = create_client(SUPABASE_URL, SUPABASE_KEY)


def write_status(ok: bool, reason: str, saved: int = 0):
    """Written after every run so scheduler.js can decide whether to fall
    back to the simulated generator (e.g. on auth failure or exhausted
    free-tier quota, which surfaces as a 401/402/429)."""
    try:
        with open(STATUS_FILE, "w") as f:
            json.dump({
                "ok": ok,
                "reason": reason,
                "saved": saved,
                "source": "twitter",
                "checked_at": datetime.now(timezone.utc).isoformat(),
            }, f)
    except Exception as e:
        logging.warning(f"WARNING: Could not write status file: {e}")


if not BEARER_TOKEN:
    logging.error("TWITTER_BEARER_TOKEN not set in .env")
    write_status(False, "missing_token")
    raise SystemExit(1)

try:
    twitter = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=False)
    logging.info("Twitter client ready")
except Exception as e:
    logging.error(f"Twitter auth failed: {e}")
    write_status(False, "auth_failed")
    raise SystemExit(1)

#  Fetch logic
last_tweet_id = None


def fetch_tweets():
    global last_tweet_id

    try:
        resp = twitter.search_recent_tweets(
            query=SEARCH_QUERY,
            max_results=10,
            tweet_fields=["created_at", "text", "author_id", "public_metrics"],
            expansions=["author_id"],
            user_fields=["username"],
            since_id=last_tweet_id
        )
    except tweepy.Unauthorized:
        logging.error("Twitter bearer token invalid or expired — check TWITTER_BEARER_TOKEN in .env")
        write_status(False, "unauthorized")
        return
    except tweepy.TooManyRequests:
        logging.warning("WARNING: Rate limit hit.")
        write_status(False, "rate_limited")
        return
    except tweepy.Forbidden as e:
        # Covers Twitter's paid-tier / quota-exhausted responses (e.g. HTTP 402/403
        # "This request is not authorized for the current API access level").
        logging.error(f"Forbidden / quota issue: {e}")
        write_status(False, "quota_or_access_level")
        return
    except Exception as e:
        logging.error(f"Twitter fetch error: {e}")
        write_status(False, "unknown_error")
        return

    if not resp.data:
        logging.info("No new tweets found.")
        write_status(True, "no_new_tweets", 0)
        return

    # Build author_id -> username map from expansions, if present
    usernames = {}
    if resp.includes and "users" in resp.includes:
        for u in resp.includes["users"]:
            usernames[str(u.id)] = u.username

    saved = 0
    for t in resp.data:
        tweet_id = str(t.id)

        # Skip duplicates
        try:
            exists = db.table("raw_tweets").select("id").eq("tweet_id", tweet_id).execute()
            if exists.data:
                continue
        except Exception as e:
            logging.warning(f"WARNING: Duplicate check failed: {e}")
            continue

        author_id = str(t.author_id)
        doc = {
            "tweet_id":      tweet_id,
            "user_id":       author_id,
            "username":      usernames.get(author_id, f"user_{author_id[-4:]}"),
            "text":          t.text,
            "timestamp":     str(t.created_at),
            "like_count":    t.public_metrics["like_count"],
            "retweet_count": t.public_metrics["retweet_count"],
            "source":        "twitter",
        }

        try:
            db.table("raw_tweets").insert(doc).execute()
            saved += 1
        except Exception as e:
            logging.warning(f"WARNING: Insert failed for {tweet_id}: {e}")

    last_tweet_id = resp.data[0].id
    logging.info(f"Saved {saved} new tweet(s).")
    write_status(True, "ok", saved)


#  Run (single pass — scheduler handles the loop externally)
if __name__ == "__main__":
    logging.info(f"Tweet fetcher (single pass) | query: {SEARCH_QUERY}")
    fetch_tweets()
    logging.info("Fetch pass complete. Exiting.")
