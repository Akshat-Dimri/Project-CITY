# C.I.T.Y. – Civic Issues Tracking for You

**An AI-powered system for real-time civic complaint tracking & analysis.**

## 🚀 Project Overview
C.I.T.Y. is a civic issue tracking system that fetches complaints from **Twitter** using APIs, processes them using **NLP & sentiment analysis**, and prioritizes them based on severity.

## 📌 Features
- Fetches real-time complaints from **Twitter** using hashtags (e.g., `#CITYHamirpur`).
- Stores data in **MongoDB** for NLP processing.
- Analyzes complaint severity using **VADER sentiment analysis and keyword classification**.
- Provides a **web-based dashboard** where users can upvote/downvote issues.
- Automatically prioritizes complaints based on severity & user engagement.

## 🛠️ Installation
To run the project locally, install the required dependencies:

### 1. Clone the Repository
```sh
git clone https://github.com/Akshat-Dimri/Project-CITY.git

```

### 2. Install Backend Dependencies
Ensure you have Python installed, then install the required libraries:
```sh
pip install -r requirements.txt
```

### 3. Install Frontend & Server Dependencies
Install Node.js dependencies:
```sh
npm install
```

### 4. Set Up Environment Variables
Create a `.env` file in the project root and add:
```
TWITTER_BEARER_TOKEN=your_twitter_api_key
MONGO_URI=your_mongodb_connection_string
```

### 5. Run Backend Services
Start the tweet fetching & NLP processing services:
```sh
python initial_fetch.py  # Fetch tweets from Twitter
python NLProcessing.py  # Perform NLP analysis
```

### 6. Run the Server
Start the Express.js server for the web dashboard:
```sh
node server.js
```

## 🔮 Tech Stack
- **Backend:** Python (Tweepy, NLTK, PyTorch)
- **Database:** MongoDB (Atlas)
- **NLP:** VADER Sentiment Analysis, Keyword-based classification
- **Frontend:** React/Next.js
- **Server:** Node.js, Express

## 🔮 Future Prospects
- **Blockchain integration** to ensure complaint immutability.
- **Advanced NLP models (BERT-based)** for better classification.
- **Additional complaint sources** such as WhatsApp & Telegram.
- **Multilingual support** for wider accessibility.

## 📢 Contact
For queries or contributions, reach out to the team via GitHub issues.

