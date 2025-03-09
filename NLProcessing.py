import time
import torch
import numpy as np
from pymongo import MongoClient
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW

# GPU availability, local system processing
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# NLTK Resource check
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# MongoDB Atlas connection
mongo_uri = "mongodb+srv://AkshatDimri:X7G0tooefK1Xyo1y@postdata.1mc5t.mongodb.net/"
client_db = MongoClient(mongo_uri)

# source and target databases and collections
source_db = client_db["civic_complaints"]
source_collection = source_db["tweets"]

target_db = client_db["PostNLP_Data"]
target_collection = target_db["analyzed_tweets"]

# sentiment analyzer initizlizing
sid = SentimentIntensityAnalyzer()

# Temporary issue categories
ISSUE_CATEGORIES = ['Water', 'Electricity', 'Roads', 'Sanitation', 'Parks', 'Other']

# Initialize BERT model for advanced sentiment analysis - will work with enough data, used as a preemptive measure
def initialize_bert_model():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(ISSUE_CATEGORIES))
    model.to(device)
    return tokenizer, model

# Custom dataset for PyTorch
class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(ISSUE_CATEGORIES.index(label))
        }

def calculate_severity(sentiment_score, like_count, retweet_count):
    """
    Calculate severity score based on sentiment and engagement metrics.
    
    Severity increases with:
    - More negative sentiment
    - Higher engagement (likes and retweets)
    
    Scale: 0-10 where 10 is most severe
    """
    # Base severity from sentiment (-1 to 1 scale)
    # Convert to 0-10 scale and invert (more negative = more severe)
    base_severity = (1 - sentiment_score) * 5
    
    # Engagement factor - more engagement increases severity
    # Assuming max realistic engagement for local civic complaints is ~1000
    engagement = (like_count + retweet_count * 2) / 1000  # Weight retweets higher
    engagement_factor = min(engagement, 1) * 3  # Cap at 3 points of severity
    
    # Final severity score (capped at 10)
    severity = min(base_severity + engagement_factor, 10)
    
    return round(severity, 1)  # Round to 1 decimal place

def classify_issue(text):
    """
    Simple keyword-based classification of civic issues.
    In a production system, you would use a trained classifier.
    """
    text = text.lower()
    
    if any(word in text for word in ['water', 'leak', 'pipe', 'drainage', 'sewage','flooding']):
        return 'Water'
    elif any(word in text for word in ['electricity', 'power', 'outage', 'blackout','electric']):
        return 'Electricity'
    elif any(word in text for word in ['road', 'pothole', 'traffic', 'signal', 'jam']):
        return 'Roads'
    elif any(word in text for word in ['garbage', 'waste', 'trash', 'litter', 'dump']):
        return 'Sanitation'
    elif any(word in text for word in ['park', 'garden', 'tree', 'playground']):
        return 'Parks'
    else:
        return 'Other'

def process_tweets_gpu_batch(tweets, bert_model=None, tokenizer=None, batch_size=16):
    """
    Processing a batch of tweets using GPU acceleration if available
    """
    texts = [tweet.get("text", "") for tweet in tweets]
    
    # If we have a trained BERT model, use it for classification
    if bert_model is not None and tokenizer is not None:
        # Tokenize all texts in the batch
        encodings = tokenizer(texts, truncation=True, padding=True, return_tensors="pt").to(device)
        
        # Process in mini-batches to avoid CUDA out of memory
        all_predictions = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_encodings = {k: v[i:i+batch_size] for k, v in encodings.items()}
                outputs = bert_model(**batch_encodings)
                predictions = torch.argmax(outputs.logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                
        # Convert numerical predictions to category labels
        issue_categories = [ISSUE_CATEGORIES[pred] for pred in all_predictions]
    else:
        # Use keyword-based classification as fallback
        issue_categories = [classify_issue(text) for text in texts]
    
    processed_tweets = []
    for i, tweet in enumerate(tweets):
        tweet_text = texts[i]
        
        # Perform sentiment analysis with VADER (CPU-based, but fast)
        sentiment_scores = sid.polarity_scores(tweet_text)
        compound_score = sentiment_scores['compound']
        
        # Calculate severity score
        like_count = tweet.get("like_count", 0)
        retweet_count = tweet.get("retweet_count", 0)
        severity = calculate_severity(compound_score, like_count, retweet_count)
        
        # Create analyzed data document
        analyzed_tweet = {
            "tweet_id": tweet.get("tweet_id"),
            "user_id": tweet.get("user_id"),
            "text": tweet_text,
            "timestamp": tweet.get("timestamp"),
            "like_count": like_count,
            "retweet_count": retweet_count,
            "sentiment": {
                "compound": compound_score,
                "positive": sentiment_scores['pos'],
                "negative": sentiment_scores['neg'],
                "neutral": sentiment_scores['neu']
            },
            "severity_score": severity,
            "issue_category": issue_categories[i],
            "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        processed_tweets.append(analyzed_tweet)
    
    return processed_tweets

def process_tweets(batch_size=100, bert_model=None, tokenizer=None):
    """
    Process tweets from source collection, perform sentiment analysis,
    calculate severity, and save to target collection
    """
    # Find tweets that haven't been analyzed yet
    unprocessed_tweets = list(source_collection.find(
        {"processed": {"$ne": True}}
    ).limit(batch_size))
    
    if not unprocessed_tweets:
        print("No new tweets to process")
        return 0
    
    # Process tweets in GPU-accelerated batch
    processed_tweets = process_tweets_gpu_batch(
        unprocessed_tweets, 
        bert_model=bert_model, 
        tokenizer=tokenizer
    )
    
    # Bulk insert to target collection
    if processed_tweets:
        target_collection.insert_many(processed_tweets)
        
        # Mark tweets as processed in source collection
        for tweet in unprocessed_tweets:
            source_collection.update_one(
                {"_id": tweet["_id"]},
                {"$set": {"processed": True}}
            )
        
        print(f"Processed and stored {len(processed_tweets)} tweets")
    
    return len(processed_tweets)

def train_advanced_classifier_gpu():
    """
    Train a BERT-based classifier on GPU for issue categorization
    """
    # Fetch historical analyzed data
    historical_data = list(target_collection.find({}, {"text": 1, "issue_category": 1}))
    
    if len(historical_data) < 100:
        print("Not enough historical data for advanced classifier")
        return None, None
    
    print(f"Training with {len(historical_data)} samples on {device}")
    
    # Convert to DataFrame and prepare data
    df = pd.DataFrame(historical_data)
    texts = df['text'].tolist()
    labels = df['issue_category'].tolist()
    
    # Initialize tokenizer and model
    tokenizer, model = initialize_bert_model()
    
    # Create dataset and dataloader
    train_dataset = TweetDataset(texts, labels, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # Training parameters
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 3
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_dataloader:
            # Move batch to GPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_dataloader)}")
    
    print("Training completed")
    return tokenizer, model

def create_indexes():
    """Create indexes for better query performance"""
    target_collection.create_index("tweet_id", unique=True)
    target_collection.create_index("issue_category")
    target_collection.create_index("severity_score")
    
    # Add processed flag index to source collection
    source_collection.create_index("processed")
    
    print("Indexes created successfully")

def main():
    # Create necessary indexes
    create_indexes()
    
    print("Starting tweet analysis process with GPU acceleration...")
    
    # Variables to store BERT model and tokenizer
    bert_tokenizer = None
    bert_model = None
    
    # Main processing loop
    while True:
        try:
            # Process tweets with the current model (if available)
            processed_count = process_tweets(batch_size=100, bert_model=bert_model, tokenizer=bert_tokenizer)
            
            # Train/update model if we have enough data and no model yet
            if (bert_model is None and target_collection.count_documents({}) > 500) or \
               (processed_count > 0 and target_collection.count_documents({}) % 1000 == 0):
                print("Training advanced classifier with historical data...")
                bert_tokenizer, bert_model = train_advanced_classifier_gpu()
            
            # Sleep for a while before the next batch
            print("Waiting for new tweets...")
            time.sleep(60)  # Check for new tweets every minute
            
        except Exception as e:
            print(f"Error in processing: {e}")
            time.sleep(300)  # Sleep longer if there's an error

if __name__ == "__main__":
    main()

    #Using regex, for troll detection 

def detect_trolls(tweet_text):
    """
    Uses regex patterns to detect potential troll content in tweets.
    Returns a dictionary with detection results.
    
    Parameters:
    tweet_text (str): The text content of the tweet to analyze
    
    Returns:
    dict: Contains 'is_troll' boolean flag and 'troll_indicators' list of matched patterns
    """
    # Convert to lowercase for case-insensitive matching
    text = tweet_text.lower()
    
    # Regex patterns for different troll indicators
    patterns = {
        'repeated_characters': r'(\w)\1{4,}',  # Character repeated 5+ times (like "haaaaa", "!!!!!!")
        'excessive_punctuation': r'[!?$*()<>]{5,}',  # Three or more consecutive exclamation/question marks
        'fake_govt_claim': r'(?i)(official|govt|government).{0,20}(announcement|notice|alert)',  # Fake official claims
        'inflammatory_language': r'\b(idiots?|morons?|stupid|dumb|useless|incompetent)\b.{0,30}\b(government|officials?|mayor|council)\b',
        'excessive_hashtags': r'(#\w+){5,}',  # Five or more hashtags
        'spam_indicators': r'(?i)(click here|free money|lottery|prize|won|winner|discount code)',
        'bot_patterns': r'(?i)(auto-generated|bot response|automated message)',
        'offensive_slurs': r'\b(f+[\W_]*u+[\W_]*c+[\W_]*k+|b+[\W_]*i+[\W_]*t+[\W_]*c+[\W_]*h+|sh+[\W_]*i+[\W_]*t+)\b'
    }
    
    # Detect matches for each pattern
    matches = {}
    for pattern_name, regex in patterns.items():
        matches[pattern_name] = bool(re.search(regex, tweet_text))
    
    # Determine if the tweet is likely from a troll
    # A tweet is considered a troll if it matches at least 2 patterns
    troll_indicators = [name for name, matched in matches.items() if matched]
    is_troll = len(troll_indicators) >= 2
    
    return {
        'is_troll': is_troll,
        'troll_indicators': troll_indicators,
        'troll_score': len(troll_indicators) / len(patterns) * 10  # Score from 0-10
    }

# Modified version of process_tweets_gpu_batch that includes troll detection
def process_tweets_gpu_batch_with_trolls(tweets, bert_model=None, tokenizer=None, batch_size=16):
    """
    Processing a batch of tweets with troll detection while preserving original functionality
    """
    # First get the original processed tweets
    processed_tweets = process_tweets_gpu_batch(tweets, bert_model, tokenizer, batch_size)
    
    # Now add troll analysis to each processed tweet
    for processed_tweet in processed_tweets:
        tweet_text = processed_tweet.get("text", "")
        troll_analysis = detect_trolls(tweet_text)
        processed_tweet["troll_analysis"] = troll_analysis
    
    return processed_tweets

# Function to filter out troll tweets (can be called separately without modifying original pipeline)
def filter_troll_tweets():
    """
    Identifies troll tweets in the analyzed_tweets collection and marks them
    """
    analyzed_tweets = list(target_collection.find({"troll_analysis": {"$exists": False}}))
    
    for tweet in analyzed_tweets:
        tweet_text = tweet.get("text", "")
        troll_analysis = detect_trolls(tweet_text)
        
        # Update the tweet with troll analysis
        target_collection.update_one(
            {"_id": tweet["_id"]},
            {"$set": {"troll_analysis": troll_analysis}}
        )
        
        # If it's a troll, also mark it in the source collection if needed
        if troll_analysis['is_troll']:
            tweet_id = tweet.get("tweet_id")
            if tweet_id:
                source_collection.update_one(
                    {"tweet_id": tweet_id},
                    {"$set": {"is_troll": True, "troll_indicators": troll_analysis['troll_indicators']}}
                )
    
    return len(analyzed_tweets)

# Alternative process_tweets function that incorporates troll detection
def process_tweets_with_troll_detection(batch_size=100, bert_model=None, tokenizer=None):
    """
    Process tweets with troll detection while preserving the original process_tweets function
    """
    # Find tweets that haven't been analyzed yet
    unprocessed_tweets = list(source_collection.find(
        {"processed": {"$ne": True}}
    ).limit(batch_size))
    
    if not unprocessed_tweets:
        print("No new tweets to process")
        return 0
    
    # Process tweets with the enhanced function that includes troll detection
    processed_tweets = process_tweets_gpu_batch_with_trolls(
        unprocessed_tweets, 
        bert_model=bert_model, 
        tokenizer=tokenizer
    )
    
    # Bulk insert to target collection
    if processed_tweets:
        target_collection.insert_many(processed_tweets)
        
        # Mark tweets as processed in source collection
        for tweet in unprocessed_tweets:
            source_collection.update_one(
                {"_id": tweet["_id"]},
                {"$set": {"processed": True}}
            )
        
        print(f"Processed and stored {len(processed_tweets)} tweets with troll detection")
    
    return len(processed_tweets)

# Create a separate collection for trolls
def setup_troll_collection():
    """Creates a separate collection for tracking troll tweets"""
    troll_db = client_db["PostNLP_Data"]
    troll_collection = troll_db["troll_tweets"]
    
    # Create indexes
    troll_collection.create_index("tweet_id", unique=True)
    troll_collection.create_index("troll_score")
    
    print("Troll collection setup complete")
    return troll_collection

# Function to export only troll tweets to a separate collection
def export_troll_tweets():
    """
    Finds all tweets marked as trolls and exports them to the troll collection
    """
    troll_collection = setup_troll_collection()
    
    # Find all tweets with troll analysis where is_troll is true
    troll_tweets = target_collection.find(
        {"troll_analysis.is_troll": True}
    )
    
    count = 0
    for tweet in troll_tweets:
        # Create a document for the troll collection
        troll_doc = {
            "tweet_id": tweet.get("tweet_id"),
            "user_id": tweet.get("user_id"),
            "text": tweet.get("text"),
            "timestamp": tweet.get("timestamp"),
            "troll_analysis": tweet.get("troll_analysis"),
            "severity_score": tweet.get("severity_score"),
            "issue_category": tweet.get("issue_category"),
            "export_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Insert or update in troll collection
        troll_collection.update_one(
            {"tweet_id": tweet.get("tweet_id")},
            {"$set": troll_doc},
            upsert=True
        )
        count += 1
    
    print(f"Exported {count} troll tweets")
    return count

# Example of how to use these functions (can be called from a separate script)
def troll_analysis_workflow():
    """
    Complete workflow for troll analysis without modifying the original pipeline
    """
    print("Starting troll analysis workflow...")
    
    # Setup the troll collection
    setup_troll_collection()
    
    # Run either approach:
    
    # Option 1: Process new tweets with troll detection
    # Uncomment to use this approach:
    # processed_count = process_tweets_with_troll_detection(batch_size=100)
    # print(f"Processed {processed_count} new tweets with troll detection")
    
    # Option 2: Process existing analyzed tweets
    analyzed_count = filter_troll_tweets()
    print(f"Added troll analysis to {analyzed_count} existing tweets")
    
    # Export troll tweets to dedicated collection
    export_count = export_troll_tweets()
    print(f"Exported {export_count} troll tweets to dedicated collection")
    
    print("Troll analysis workflow completed!")