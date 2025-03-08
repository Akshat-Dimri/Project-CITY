import json
import re
import tweepy
import pandas as pd
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
from flask import Flask, request, jsonify
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Twitter API credentials from environment variables
bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
if not bearer_token:
    raise ValueError("Please set the TWITTER_BEARER_TOKEN environment variable.")

# Authenticate with Twitter client
client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=False)

# Initialize NLP tools
tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Flask app for deployment
app = Flask(__name__)

# Preprocess the data
def preprocess_text(text):
    """Basic text preprocessing."""
    text = text.lower()
    text = " ".join([word for word in text.split() if word.isalnum()])
    return text

# Extract hashtags
def extract_hashtags(text):
    """Extract hashtags from a tweet."""
    return re.findall(r"#(\w+)", text)

# Enrich tweet data
def enrich_tweet(tweet):
    """Enrich tweet data with additional features."""
    text = tweet["text"]
    hashtags = extract_hashtags(text)
    location = tweet.get("geo", {}).get("place_id", None)
    user_influence = tweet.get("user", {}).get("followers_count", 0) or 0
    
    return {
        **tweet,
        "hashtags": hashtags,
        "location": location,
        "user_influence": user_influence
    }

# Categorize tweets based on civic issues
categories = {
    "Infrastructure & Public Services": ["pothole", "traffic", "footpath", "public transport", "streetlight","bad roads", "encroachment"],
    "Water & Sanitation": ["water", "drainage", "toilet", "garbage", "waste management"],
    "Pollution & Environmental Concerns": ["air pollution", "noise pollution", "plastic waste", "deforestation", "recycling"],
    "Public Safety & Law Enforcement": ["crime", "cctv", "emergency", "harassment", "vandalism"],
    "Housing & Urban Development": ["illegal construction", "slum", "affordable housing", "public park"],
    "Healthcare & Hygiene": ["hospital", "hygiene", "epidemic", "medical waste"],
    "Education & Public Welfare": ["school", "teacher", "library", "community center"],
    "Corruption & Bureaucratic Issues": ["bribery", "delay", "red tape", "grievance"],
    "Public Transport & Mobility": ["bus", "train", "metro", "last-mile connectivity"],
    "Energy & Power Supply": ["power cut", "load shedding", "electricity"],
    "Digital & IT Services": ["network", "telecom", "wifi", "digital governance"]
}

category_weights = {category: 1.0 for category in categories.keys()}
category_weights["Public Safety & Law Enforcement"] = 1.3
category_weights["Corruption & Bureaucratic Issues"] = 1.3
category_weights["Water & Sanitation"] = 1.2
category_weights["Healthcare & Hygiene"] = 1.2
category_weights["Pollution & Environmental Concerns"] = 1.1
category_weights["Public Transport & Mobility"] = 1.1
category_weights["Other"] = 0.8

# Severity Score Calculation
def calculate_severity_score(keywords, sentiment_label, sentiment_score, upvotes, location_weight, user_influence, categories_found):
    """ severity calculation with normalization and keyword importance."""
    
    sentiment_score = (2 * sentiment_score) - 1
    user_influence_weight = math.log1p(user_influence)
    
    important_keywords = {"pothole": 1.5, "crime": 2.0, "pollution": 1.2, "water": 1.3}
    keyword_score = sum([important_keywords.get(word, 1) for word in keywords])
    
    severity_score = (
        0.4 * keyword_score +
        0.3 * sentiment_score +
        0.2 * upvotes +
        0.1 * location_weight +
        0.1 * user_influence_weight
    )
    
    category_weight = sum([category_weights.get(cat, 1.0) for cat in categories_found]) / len(categories_found)
    severity_score *= category_weight
    
    return round(severity_score, 2)

# Categorization function
def categorize_tweet(text):
    """Assign multiple categories to tweets."""
    return [category for category, keywords in categories.items() if any(word in text for word in keywords)] or ["Other"]

# Fetch and process tweets
def fetch_and_process_tweets():
    """Fetch tweets, enrich data, and process them."""
    query = "(pothole OR traffic OR water OR pollution OR crime OR hospital OR school OR corruption OR bus OR power OR wifi) lang:en -is:retweet"
    try:
        tweets = client.search_recent_tweets(
            query=query, max_results=10,
            tweet_fields=["created_at", "text", "public_metrics", "geo", "author_id"],
            user_fields=["followers_count"]
        )

        tweet_data = []
        if tweets.data:
            for tweet in tweets.data:
                text = preprocess_text(tweet.text)
                enriched_tweet = enrich_tweet({"tweet_id": tweet.id, "text": text, "retweet_count": tweet.public_metrics.get('retweet_count', 0), "like_count": tweet.public_metrics.get('like_count', 0), "geo": tweet.geo, "user": {"followers_count": tweet.public_metrics.get('followers_count', 0)}})
                categories_found = categorize_tweet(text)
                sentiment_result = sentiment_analyzer(text)[0]
                sentiment_label, sentiment_score = sentiment_result['label'], sentiment_result['score']
                severity_score = calculate_severity_score([], sentiment_label, sentiment_score, enriched_tweet["retweet_count"] + enriched_tweet["like_count"], 1 if enriched_tweet["location"] else 0, enriched_tweet["user_influence"], categories_found)
                tweet_data.append({"tweet_id": enriched_tweet["tweet_id"], "text": text, "sentiment_label": sentiment_label, "sentiment_score": sentiment_score, "severity_score": severity_score, "categories": categories_found})

        return tweet_data
    except Exception as e:
        logger.error(f"Error fetching and processing tweets: {e}")
        return []

# Flask API Endpoints
@app.route('/fetch_and_process_tweets', methods=['GET'])
def fetch_and_process_tweets_api():
    return jsonify({"status": "success", "tweets": fetch_and_process_tweets()})

@app.route('/routes', methods=['GET'])
def list_routes():
    return jsonify([str(rule) for rule in app.url_map.iter_rules()])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
