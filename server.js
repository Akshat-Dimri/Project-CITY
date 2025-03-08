// server.js
const express = require('express');
const { MongoClient, ObjectId } = require('mongodb');
const cors = require('cors');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// MongoDB connection string
const uri = "mongodb+srv://AkshatDimri:X7G0tooefK1Xyo1y@postdata.1mc5t.mongodb.net/";
const client = new MongoClient(uri);
const dbName = "PostNLP_Data";
const collectionName = "analyzed_tweets";

// Connect to MongoDB
async function connectToMongo() {
  try {
    await client.connect();
    console.log("Connected to MongoDB Atlas");
  } catch (error) {
    console.error("Error connecting to MongoDB:", error);
  }
}

// Get all issues sorted by severity score
app.get('/api/issues', async (req, res) => {
  try {
    const db = client.db(dbName);
    const collection = db.collection(collectionName);
    
    const issues = await collection.find({}).toArray();
    
    // Calculate effective severity score (original + upvotes - downvotes)
    const issuesWithEffectiveScore = issues.map(issue => {
      const upvotes = issue.upvotes || 0;
      const downvotes = issue.downvotes || 0;
      const effectiveSeverity = issue.severity_score + upvotes - downvotes;
      
      return {
        ...issue,
        effective_severity: effectiveSeverity
      };
    });
    
    // Sort by effective severity score
    issuesWithEffectiveScore.sort((a, b) => b.effective_severity - a.effective_severity);
    
    res.json(issuesWithEffectiveScore);
  } catch (error) {
    console.error("Error fetching issues:", error);
    res.status(500).json({ error: "Failed to fetch issues" });
  }
});

// Update vote count (upvote or downvote)
app.post('/api/issues/:id/vote', async (req, res) => {
  try {
    const { id } = req.params;
    const { voteType } = req.body; // 'upvote' or 'downvote'
    
    const db = client.db(dbName);
    const collection = db.collection(collectionName);
    
    const issue = await collection.findOne({ _id: new ObjectId(id) });
    
    if (!issue) {
      return res.status(404).json({ error: "Issue not found" });
    }
    
    // Initialize vote counts if they don't exist
    if (!issue.upvotes) issue.upvotes = 0;
    if (!issue.downvotes) issue.downvotes = 0;
    
    // Update the vote count
    if (voteType === 'upvote') {
      await collection.updateOne(
        { _id: new ObjectId(id) },
        { $inc: { upvotes: 1 } }
      );
    } else if (voteType === 'downvote') {
      await collection.updateOne(
        { _id: new ObjectId(id) },
        { $inc: { downvotes: 1 } }
      );
    }
    
    res.json({ success: true });
  } catch (error) {
    console.error("Error updating vote:", error);
    res.status(500).json({ error: "Failed to update vote" });
  }
});

// Start server
app.listen(PORT, async () => {
  await connectToMongo();
  console.log(`Server running on port ${PORT}`);
  
  // Set up periodic data refresh (every 10 minutes)
  setInterval(async () => {
    console.log("Refreshing data from MongoDB...");
    // This will ensure data is refreshed in subsequent API calls
  }, 10 * 60 * 1000); // 10 minutes in milliseconds
});