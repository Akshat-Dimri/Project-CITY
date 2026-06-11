#  Imports ─
import os
import time
import logging
import tweepy
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

if not BEARER_TOKEN:
    logging.error("❌ TWITTER_BEARER_TOKEN not set in .env")
    raise SystemExit(1)
if not SUPABASE_URL or not SUPABASE_KEY:
    logging.error("❌ SUPABASE_URL or SUPABASE_KEY not set in .env")
    raise SystemExit(1)

#  Clients 
db = create_client(SUPABASE_URL, SUPABASE_KEY)

try:
    twitter = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=True)
    logging.info("✅ Twitter client ready")
except Exception as e:
    logging.error(f"❌ Twitter auth failed: {e}")
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
            since_id=last_tweet_id
        )
    except tweepy.Unauthorized:
        logging.error("❌ Twitter bearer token invalid or expired — check TWITTER_BEARER_TOKEN in .env")
        return
    except tweepy.TooManyRequests:
        logging.warning("⚠️  Rate limit hit. Sleeping 15 min...")
        time.sleep(900)
        return
    except Exception as e:
        logging.error(f"❌ Twitter fetch error: {e}")
        return

    if not resp.data:
        logging.info("ℹ️  No new tweets found.")
        return

    saved = 0
    for t in resp.data:
        tweet_id = str(t.id)

        # Skip duplicates
        try:
            exists = db.table("raw_tweets").select("id").eq("tweet_id", tweet_id).execute()
            if exists.data:
                continue
        except Exception as e:
            logging.warning(f"⚠️  Duplicate check failed: {e}")
            continue

        doc = {
            "tweet_id":      tweet_id,
            "user_id":       str(t.author_id),
            "text":          t.text,
            "timestamp":     str(t.created_at),
            "like_count":    t.public_metrics["like_count"],
            "retweet_count": t.public_metrics["retweet_count"],
        }

        try:
            db.table("raw_tweets").insert(doc).execute()
            saved += 1
        except Exception as e:
            logging.warning(f"⚠️  Insert failed for {tweet_id}: {e}")

    last_tweet_id = resp.data[0].id
    logging.info(f"✅ Saved {saved} new tweet(s).")

#  Run 
#  Run (single pass — scheduler handles the loop externally) 
if __name__ == "__main__":
    logging.info(f"🚀 Tweet fetcher (single pass) | query: {SEARCH_QUERY}")
    fetch_tweets()
    logging.info("✅ Fetch pass complete. Exiting.")
