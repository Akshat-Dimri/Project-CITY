# ── Imports ───────────────────────────────────────────────────────────────────
import re
import os
import string
import logging
import time
import torch
import nltk
from textblob import TextBlob
from supabase import create_client
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from dotenv import load_dotenv

from localities import find_locality_for_text

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ── Config ────────────────────────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    logging.error("SUPABASE_URL or SUPABASE_KEY not set in .env")
    raise SystemExit(1)

ISSUE_CATEGORIES = {
    "roads":       ["road", "pothole", "street", "highway"],
    "water":       ["water", "pipeline", "drain", "sewage"],
    "electricity": ["electricity", "power", "light", "transformer"],
    "waste":       ["garbage", "waste", "trash", "dump"],
    "other":       []
}

SEVERITY_MAX_ENGAGEMENT = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Escalation thresholds — a complaint auto-forwards to the "concerned
# authority" (simulated) once either condition is met.
ESCALATION_NET_UPVOTES = 20
ESCALATION_SEVERITY    = 8

TROLL_PATTERNS = {
    "offensive":   (r"\b(?:idiot|stupid|dumb|fool|hate|nonsense)\b", 3),
    "punctuation": (r"[!?]{3,}", 1),
    "all_caps":    (r"\b[A-Z]{4,}\b", 2),
    "spam_links":  (r"https?://[^\s]+", 2),
    "repeat_word": (r"\b(\w+)\s+\1\s+\1\b", 2),
    "toxic":       (r"\b(?:kill|die|worst|useless|trash)\b", 3),
}

# ── Supabase client ───────────────────────────────────────────────────────────
db = create_client(SUPABASE_URL, SUPABASE_KEY)

def check_supabase():
    """Ping Supabase; warns but doesn't crash if unreachable."""
    try:
        db.table("raw_tweets").select("id").limit(1).execute()
        logging.info("Supabase connection OK")
    except Exception as e:
        logging.warning(f"WARNING: Supabase ping failed: {e}")

check_supabase()

# ── Text helpers ───────────────────────────────────────────────────────────────
nltk.download("punkt", quiet=True)

def clean_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()

def analyze_sentiment(text: str) -> float:
    return TextBlob(text).sentiment.polarity

def calculate_severity(sentiment: float, likes: int, retweets: int) -> float:
    engagement = min((likes + retweets) / SEVERITY_MAX_ENGAGEMENT, 5)
    return round((1 - sentiment) * 5 + engagement, 2)

def classify_issue(text: str) -> str:
    for cat, keywords in ISSUE_CATEGORIES.items():
        if any(w in text for w in keywords):
            return cat
    return "other"

def resolve_location(raw_tweet: dict, cleaned_text: str):
    """Determine (lat, lon, locality_name) for a complaint.

    - Simulated complaints already carry lat/lon/locality_name from the
      mock generator — just pass those through.
    - Real tweets are matched against the locality gazetteer by keyword.
    - If nothing matches, the complaint is left without coordinates (it
      simply won't be plottable on the map or votable until an admin
      assigns a location, which is expected for tweets that don't
      mention a recognizable local area).
    """
    if raw_tweet.get("latitude") and raw_tweet.get("longitude"):
        return raw_tweet["latitude"], raw_tweet["longitude"], raw_tweet.get("locality_name")

    match = find_locality_for_text(cleaned_text)
    if match:
        return match["lat"], match["lon"], match["name"]

    return None, None, None

# ── BERT classifier ────────────────────────────────────────────────────────────
class BertClassifier:
    def __init__(self):
        logging.info("⏳ Loading DistilBERT...")
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(DEVICE)
        self.pipe = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if DEVICE == "cuda" else -1
        )
        logging.info("DistilBERT ready")

    def predict(self, text: str):
        return self.pipe(text)[0]

bert = BertClassifier()

# ── Troll detection ────────────────────────────────────────────────────────────
compiled = {k: (re.compile(v[0], re.IGNORECASE), v[1]) for k, v in TROLL_PATTERNS.items()}

def detect_troll(text: str) -> dict:
    score, reasons = 0, []
    for label, (regex, weight) in compiled.items():
        if regex.search(text):
            score += weight
            reasons.append(label)
    return {"is_troll": score >= 3, "troll_score": score, "reasons": reasons}

# ── Main processing loop ───────────────────────────────────────────────────────
def process_tweets():
    new_count = 0

    # Fetch raw tweets from Supabase
    try:
        res = db.table("raw_tweets").select("*").execute()
        raw_tweets = res.data
    except Exception as e:
        logging.error(f"Failed to fetch raw tweets: {e}")
        return

    logging.info(f"Fetched {len(raw_tweets)} raw tweets from Supabase")

    for tweet in raw_tweets:
        tweet_id = str(tweet.get("tweet_id"))

        # Skip already processed
        try:
            exists = db.table("analyzed_tweets").select("id").eq("tweet_id", tweet_id).execute()
            if exists.data:
                logging.debug(f"⏭️  Already processed: {tweet_id}")
                continue
        except Exception as e:
            logging.warning(f"WARNING: Check-exists failed for {tweet_id}: {e} — processing anyway")

        logging.info(f"Processing {tweet_id}...")
        try:
            text     = clean_text(tweet["text"])
            senti    = analyze_sentiment(text)
            severity = calculate_severity(senti, tweet.get("like_count", 0), tweet.get("retweet_count", 0))
            category = classify_issue(text)
            bert_res = bert.predict(text)
            troll    = detect_troll(tweet["text"])
            lat, lon, locality = resolve_location(tweet, text)

            # A tweet can already qualify for escalation on arrival if its
            # severity is high enough (net upvotes obviously start at 0).
            forwarded = severity >= ESCALATION_SEVERITY

            doc = {
                "tweet_id":       tweet_id,
                "user_id":        str(tweet.get("user_id", "")),
                "username":       tweet.get("username") or f"user_{str(tweet.get('user_id',''))[-4:]}",
                "text":           tweet["text"],
                "cleaned_text":   text,
                "sentiment_score": senti,
                "severity_score": severity,
                "effective_severity": severity,
                "issue_category": category,
                "bert_label":     bert_res["label"],
                "bert_score":     bert_res["score"],
                "troll_flag":     troll["is_troll"],
                "troll_score":    troll["troll_score"],
                "troll_reasons":  troll["reasons"],
                "timestamp":      tweet.get("timestamp"),
                "like_count":     tweet.get("like_count", 0),
                "retweet_count":  tweet.get("retweet_count", 0),
                "upvotes":        0,
                "downvotes":      0,
                "source":         tweet.get("source", "twitter"),
                "latitude":       lat,
                "longitude":      lon,
                "locality_name":  locality,
                "forwarded":      forwarded,
                "forwarded_at":   None,
                "escalation_reason": "high_severity" if forwarded else None,
            }

            db.table("analyzed_tweets").insert(doc).execute()
            logging.info(f"{tweet_id} | cat={category} | sev={severity} | loc={locality} | troll={troll['is_troll']}")
            new_count += 1

        except Exception as e:
            logging.error(f"Error on {tweet_id}: {e}")

    logging.info(f"Cycle done — {new_count} new tweets processed." if new_count else "No new tweets.")

# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.info("NLP pipeline starting...")
    try:
        while True:
            process_tweets()
            logging.info("Sleeping 30s...")
            time.sleep(30)
    except KeyboardInterrupt:
        logging.info("Stopped.")
    except Exception as e:
        logging.error(f"FATAL: Fatal: {e}")
