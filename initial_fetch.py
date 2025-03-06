import tweepy
import time
from pymongo import MongoClient

# Twitter API v2 credentials (Directly embedded)
bearer_token = "AAAAAAAAAAAAAAAAAAAAABzTzgEAAAAAZHHPdG4RJxW3TJss4kY5HpRHt7Q%3DoAaIsBYUxpBGPuGqeSaBVPOqSTdfD7vphAl1HtgxBERSJ67HvQ"

# MongoDB Atlas connection (Direct URI)
mongo_uri = "mongodb+srv://AkshatDimri:X7G0tooefK1Xyo1y@postdata.1mc5t.mongodb.net/"
client_db = MongoClient(mongo_uri)

# Select database and collection
db = client_db["civic_complaints"]
collection = db["tweets"]

# Authenticate with Twitter client
try:
    twitter_client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)
    print("✅ Twitter authentication successful!")
except tweepy.Unauthorized:
    print("❌ ERROR: Invalid Twitter API credentials. Check bearer token.")
    exit(1)

last_tweet_id = None

def fetch_tweets():
    global last_tweet_id
    query = "#CITYHamirpur -is:retweet"
    max_requests = 450  # Max API calls before sleeping
    tweets_fetched = 0

    for _ in range(max_requests):  # Loop until hitting rate limit
        try:
            tweets = twitter_client.search_recent_tweets(
                query=query, 
                max_results=10, 
                tweet_fields=["created_at", "text", "author_id", "public_metrics"], 
                since_id=last_tweet_id
            )

            if tweets.data:
                tweet_data = []
                for tweet in tweets.data:
                    tweet_info = {
                        "tweet_id": tweet.id,
                        "user_id": tweet.author_id,
                        "text": tweet.text,
                        "timestamp": str(tweet.created_at),
                        "like_count": tweet.public_metrics["like_count"],
                        "retweet_count": tweet.public_metrics["retweet_count"]
                    }

                    # Check if tweet already exists in the database
                    if not collection.find_one({"tweet_id": tweet.id}):
                        collection.insert_one(tweet_info)
                        tweet_data.append(tweet_info)

                last_tweet_id = tweets.data[0].id  # Update last tweet ID
                tweets_fetched += len(tweet_data)
                print(f"Fetched {len(tweet_data)} tweets. Total: {tweets_fetched}")

            time.sleep(2)  # Short delay to avoid hitting per-minute limits

        except tweepy.TooManyRequests:
            print("Rate limit reached prematurely. Sleeping for 15 minutes...")
            time.sleep(900)
            break  # Stop loop if rate limit is hit early

    print("Rate limit reached. Sleeping for 15 minutes...")
    time.sleep(900)  # Sleep after 450 requests

# Continuous fetching loop
while True:
    fetch_tweets()
