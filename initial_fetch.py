import tweepy
import json
import time

# Twitter API v2 credentials
bearer_token = "AAAAAAAAAAAAAAAAAAAAABzTzgEAAAAAZHHPdG4RJxW3TJss4kY5HpRHt7Q%3DoAaIsBYUxpBGPuGqeSaBVPOqSTdfD7vphAl1HtgxBERSJ67HvQ"

# Authenticate with Twitter client
client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=False)  # Disable auto-wait

last_tweet_id = None

def fetch_tweets():
    global last_tweet_id
    query = "#CITYHamirpur -is:retweet"

    try:
        tweets = client.search_recent_tweets(
            query=query, max_results=10, tweet_fields=["created_at", "text", "author_id"], since_id=last_tweet_id
        )

        tweet_data = []
        if tweets.data:
            for tweet in tweets.data:
                tweet_info = {
                    "tweet_id": tweet.id,
                    "user_id": tweet.author_id,
                    "text": tweet.text,
                    "timestamp": str(tweet.created_at)
                }
                tweet_data.append(tweet_info)

            last_tweet_id = tweets.data[0].id  # Update last tweet ID

        with open("tweets.json", "w") as file:
            json.dump(tweet_data, file, indent=4)

        print("Fetched and stored", len(tweet_data), "tweets.")

    except tweepy.TooManyRequests:
        print("Rate limit reached. Sleeping for 15 minutes...")
        time.sleep(900)  # Sleep for 15 minutes if rate limit is hit

while True:
    fetch_tweets()
    time.sleep(90)  # Adjusted delay to prevent hitting the rate limit
