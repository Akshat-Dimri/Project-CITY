"""
mock_generator.py

Generates realistic, randomized civic-complaint "tweets" and inserts them
into raw_tweets, tagged source='simulated'. This is used as a fallback by
the scheduler whenever the real Twitter fetch fails (auth error, exhausted
free-tier quota, rate limit, etc.) so the demo keeps producing fresh data
without any manual intervention for months at a time.

Every simulated complaint is geographically constrained to a real locality
within ~5km of NIT Hamirpur (see localities.py), and is clearly tagged so
the frontend/admin portal can show a "Simulated" badge alongside real
"Twitter" complaints.

Run directly for a single pass (mirrors initial_fetch.py's CLI contract):
    python3 mock_generator.py
"""

import os
import sys
import json
import random
import logging
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
from supabase import create_client

from localities import random_locality

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    logging.error("SUPABASE_URL or SUPABASE_KEY not set in .env")
    raise SystemExit(1)

db = create_client(SUPABASE_URL, SUPABASE_KEY)

STATUS_FILE = os.path.join(os.path.dirname(__file__), "fetch_status.json")

# ── Complaint templates per category ────────────────────────────────────────
TEMPLATES = {
    "roads": [
        "A large pothole has formed near {loc}, causing traffic disruption and vehicle damage. Urgent repair needed. #CITYHamirpur",
        "The road surface near {loc} has been badly damaged after recent rains. Commuters are struggling daily. #CITYHamirpur",
        "Broken road divider near {loc} is a serious accident risk. Please fix it soon. #CITYHamirpur",
    ],
    "water": [
        "No water supply near {loc} for the past two days. Residents are facing a lot of difficulty. #CITYHamirpur",
        "A water pipeline near {loc} has been leaking for days, wasting water and flooding the street. #CITYHamirpur",
        "Sewage overflow near {loc} is causing a foul smell and health concerns for nearby households. #CITYHamirpur",
    ],
    "electricity": [
        "Streetlights near {loc} have been non-functional for several nights, raising safety concerns. #CITYHamirpur",
        "Frequent power cuts near {loc} over the past week are affecting daily life. #CITYHamirpur",
        "An exposed electric wire near {loc} is extremely dangerous, please send someone urgently. #CITYHamirpur",
    ],
    "waste": [
        "Garbage has been piling up near {loc} for over a week, attracting stray animals. #CITYHamirpur",
        "Overflowing dustbins near {loc} are creating unhygienic conditions for residents. #CITYHamirpur",
        "No waste collection near {loc} this week — the smell is unbearable. #CITYHamirpur",
    ],
    "other": [
        "Stray dogs near {loc} have become aggressive and are a safety concern for children. #CITYHamirpur",
        "Encroachment on the public footpath near {loc} is forcing pedestrians onto the road. #CITYHamirpur",
    ],
}

FIRST_NAMES = ["Rohan", "Priya", "Aman", "Neha", "Vikram", "Simran", "Karan", "Ayesha", "Manav", "Divya"]


def random_username():
    return f"{random.choice(FIRST_NAMES).lower()}_{random.randint(100, 999)}"


def write_status(ok: bool, reason: str, saved: int = 0):
    try:
        with open(STATUS_FILE, "w") as f:
            json.dump({
                "ok": ok,
                "reason": reason,
                "saved": saved,
                "source": "simulated",
                "checked_at": datetime.now(timezone.utc).isoformat(),
            }, f)
    except Exception as e:
        logging.warning(f"Could not write status file: {e}")


def generate_one():
    category = random.choice(list(TEMPLATES.keys()))
    template = random.choice(TEMPLATES[category])
    loc = random_locality()
    text = template.format(loc=loc["name"])

    # Small random jitter (~within a couple hundred metres) so pins don't
    # all stack exactly on the locality's reference point.
    jitter = lambda: random.uniform(-0.002, 0.002)

    return {
        "tweet_id": f"sim_{uuid.uuid4().hex[:12]}",
        "user_id": f"sim_user_{random.randint(1000, 9999)}",
        "username": random_username(),
        "text": text,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "like_count": random.randint(0, 40),
        "retweet_count": random.randint(0, 25),
        "source": "simulated",
        "latitude": round(loc["lat"] + jitter(), 6),
        "longitude": round(loc["lon"] + jitter(), 6),
        "locality_name": loc["name"],
    }


def run(n=None):
    """Insert between 1 and 3 simulated complaints (or `n` if given)."""
    count = n if n is not None else random.randint(1, 3)
    saved = 0
    for _ in range(count):
        doc = generate_one()
        try:
            db.table("raw_tweets").insert(doc).execute()
            saved += 1
            logging.info(f"Inserted simulated complaint {doc['tweet_id']} @ {doc['locality_name']}")
        except Exception as e:
            logging.warning(f"Insert failed for {doc['tweet_id']}: {e}")

    logging.info(f"Simulated pass complete — {saved} complaint(s) inserted.")
    write_status(True, "simulated_fallback", saved)
    return saved


if __name__ == "__main__":
    logging.info("Running simulated complaint generator (fallback mode)...")
    run()
