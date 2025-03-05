import tweepy
import json
import time
from pymongo import MongoClient

# Twitter API v2 credentials
bearer_token = "AAAAAAAAAAAAAAAAAAAAABzTzgEAAAAAZHHPdG4RJxW3TJss4kY5HpRHt7Q%3DoAaIsBYUxpBGPuGqeSaBVPOqSTdfD7vphAl1HtgxBERSJ67HvQ"

# MongoDB connection (Local instance)
client_db = MongoClient("mongodb://localhost:27017/")
db = client_db["civic_complaints"]  # Local database name
collection = db["tweets"]  # Collection name

# Authenticate with Twitter client
twitter_client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=False)

last_tweet_id = None

def fetch_tweets():
    global last_tweet_id
    query = "#CITYHamirpur -is:retweet"

    try:
        tweets = twitter_client.search_recent_tweets(
            query=query, max_results=10, tweet_fields=["created_at", "text", "author_id"], since_id=last_tweet_id
        )

        if tweets.data:
            tweet_data = []
            for tweet in tweets.data:
                tweet_info = {
                    "tweet_id": tweet.id,
                    "user_id": tweet.author_id,
                    "text": tweet.text,
                    "timestamp": str(tweet.created_at)
                }

                # Check if tweet already exists in the database
                if not collection.find_one({"tweet_id": tweet.id}):
                    collection.insert_one(tweet_info)
                    tweet_data.append(tweet_info)

            last_tweet_id = tweets.data[0].id  # Update last tweet ID
            print("Fetched and stored", len(tweet_data), "new tweets.")

    except tweepy.TooManyRequests:
        print("Rate limit reached. Sleeping for 15 minutes...")
        time.sleep(900)  # Sleep for 15 minutes if rate limit is hit

while True:
    fetch_tweets()
    time.sleep(90)  # Adjusted delay to prevent hitting the rate limit
