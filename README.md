# C.I.T.Y. – Civic Issues Tracking for You
*An AI-powered system for real-time civic complaint tracking & analysis with blockchain integration.*

## 🚀 Project Overview
C.I.T.Y. is a civic issue tracking system that fetches complaints from *Twitter* using APIs, processes them using *NLP & sentiment analysis*, prioritizes them based on severity, and stores records immutably on blockchain.

## 📌 Features
- Fetches real-time complaints from *Twitter* using hashtags (e.g., #CITYHamirpur).
- Stores data in *MongoDB* for NLP processing.
- Analyzes complaint severity using *VADER sentiment analysis and keyword classification*.
- Provides a *web-based dashboard* where users can upvote/downvote issues.
- Automatically prioritizes complaints based on severity & user engagement.
- *Blockchain integration* for transparent, immutable record-keeping of complaints.

## 🛠 Installation
To run the project locally, install the required dependencies:

### 1. Clone the Repository
sh
git clone https://github.com/Akshat-Dimri/Project-CITY.git


### 2. Install Backend Dependencies
Ensure you have Python installed, then install the required libraries:
sh
pip install -r requirements.txt


### 3. Install Frontend & Server Dependencies
Install Node.js dependencies:
sh
npm install


### 4. Set Up Environment Variables
Create a .env file in the project root and add:

TWITTER_BEARER_TOKEN=your_twitter_api_key
MONGO_URI=your_mongodb_connection_string
WEB3_PROVIDER_URL=your_ethereum_node_url
WALLET_PRIVATE_KEY=your_ethereum_wallet_private_key
CONTRACT_ADDRESS=your_deployed_contract_address


### 5. Install Blockchain Dependencies
sh
npm install web3 @truffle/contract @openzeppelin/contracts


### 6. Deploy Smart Contract
sh
# Install Truffle globally
npm install -g truffle

# Initialize Truffle in the blockchain directory
cd blockchain
truffle init

# Deploy the smart contract to your network of choice
truffle migrate --network <network_name>


### 7. Configure Blockchain Settings
Update the blockchain/config.js file with your contract details:
js
module.exports = {
  CONTRACT_ADDRESS: process.env.CONTRACT_ADDRESS,
  WEB3_PROVIDER_URL: process.env.WEB3_PROVIDER_URL
}


### 8. Run Backend Services
Start the tweet fetching & NLP processing services:
sh
python initial_fetch.py  # Fetch tweets from Twitter
python NLProcessing.py  # Perform NLP analysis
python blockchain_sync.py  # Sync complaints to blockchain


### 9. Run the Server
Start the Express.js server for the web dashboard:
sh
node server.js


## 🔮 Tech Stack
- *Backend:* Python (Tweepy, NLTK, PyTorch)
- *Database:* MongoDB (Atlas)
- *NLP:* VADER Sentiment Analysis, Keyword-based classification
- *Frontend:* React/HTML
- *Server:* Node.js, Express
- *Blockchain:* Ethereum, Solidity, Web3.js, Truffle

## 🔮 Future Prospects
- *Advanced NLP models (BERT-based)* for better classification.
- *Additional complaint sources* such as WhatsApp & Telegram.
- *Multilingual support* for wider accessibility.
- *DAO governance* for community-led issue prioritization.

## 📢 Contact
For queries or contributions, reach out to the team via GitHub issues.