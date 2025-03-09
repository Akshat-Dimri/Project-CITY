require('dotenv').config();
const { MongoClient } = require('mongodb');
const Web3 = require('web3');
const fs = require('fs');
const crypto = require('crypto');
const HDWalletProvider = require('@truffle/hdwallet-provider');

console.log("🔹 INFURA_PROJECT_ID:", process.env.INFURA_PROJECT_ID);

// MongoDB Configuration
const mongoURI = process.env.MONGODB_URI;
const dbName = process.env.MONGODB_DB;
const collectionName = process.env.MONGODB_COLLECTION;

// Blockchain Configuration using HDWalletProvider
const provider = new HDWalletProvider({
  mnemonic: process.env.MNEMONIC,
  providerOrUrl: `https://sepolia.infura.io/v3/${process.env.INFURA_PROJECT_ID}`,
  pollingInterval: 8000, // Helps prevent rate limit errors
});

const web3 = new Web3(provider);
const contractJSON = require('./build/contracts/ComplaintRegistry.json');

// Batch size for blockchain uploads
const BATCH_SIZE = 2;

// Function to hash complaint data
function hashComplaint(complaint) {
  return crypto
    .createHash('sha256')
    .update(JSON.stringify(complaint))
    .digest('hex');
}

// Function to store complaints on blockchain
async function storeOnBlockchain(complaints) {
  try {
    // Get the deployed contract
    const networkId = await web3.eth.net.getId();
    console.log(`🔹 Detected network ID: ${networkId}`);

    const deployedNetwork = contractJSON.networks[networkId];
    if (!deployedNetwork) {
      console.error(`❌ Contract not deployed on network ${networkId}`);
      return;
    }

    const contract = new web3.eth.Contract(
      contractJSON.abi,
      deployedNetwork.address
    );

    // Get accounts from HDWalletProvider
    const accounts = await web3.eth.getAccounts();
    console.log(`🔹 Using account: ${accounts[0]}`);

    // Prepare batch data
    const complaintIds = complaints.map(c => c._id.toString());
    const complaintHashes = complaints.map(hashComplaint);

    console.log(`📌 Registering ${complaintIds.length} complaints on blockchain...`);
    
    await contract.methods
      .batchRegisterComplaints(complaintIds, complaintHashes)
      .send({ from: accounts[0], gas: 3000000 });

    console.log('✅ Complaints registered successfully!');

    // Update MongoDB to mark as registered
    const mongoClient = new MongoClient(mongoURI);
    await mongoClient.connect();
    const db = mongoClient.db(dbName);
    const collection = db.collection(collectionName);

    for (const complaint of complaints) {
      await collection.updateOne(
        { _id: complaint._id },
        { $set: { registeredOnBlockchain: true } }
      );
    }

    await mongoClient.close();
    console.log('✅ MongoDB records updated successfully!');

  } catch (error) {
    console.error('❌ Error storing complaints on blockchain:', error);
  }
}

// Function to monitor MongoDB and store complaints on blockchain
async function monitorAndStore() {
  const mongoClient = new MongoClient(mongoURI);

  try {
    await mongoClient.connect();
    console.log('✅ Connected to MongoDB');

    const db = mongoClient.db(dbName);
    const collection = db.collection(collectionName);

    // Initial sync - process existing complaints not yet registered on blockchain
    const existingComplaints = await collection
      .find({ registeredOnBlockchain: { $ne: true } })
      .toArray();

    console.log(`📌 Found ${existingComplaints.length} unregistered complaints`);

    // Process in batches
    for (let i = 0; i < existingComplaints.length; i += BATCH_SIZE) {
      const batch = existingComplaints.slice(i, i + BATCH_SIZE);
      await storeOnBlockchain(batch);
    }

    // Set up change stream to monitor new complaints
    const changeStream = collection.watch();

    console.log('🔍 Monitoring MongoDB for new complaints...');

    changeStream.on('change', async change => {
      try {
        if (change.operationType === 'insert') {
          const newComplaint = await collection.findOne({ _id: change.documentKey._id });

          if (newComplaint && !newComplaint.registeredOnBlockchain) {
            console.log(`📌 New complaint detected: ${newComplaint._id}`);
            await storeOnBlockchain([newComplaint]);
          }
        }
      } catch (error) {
        console.error('❌ Error processing new complaint:', error);
      }
    });

  } catch (error) {
    console.error('❌ Error connecting to MongoDB:', error);
    await mongoClient.close();
    process.exit(1);
  }
}

// Start the monitoring process
monitorAndStore()
  .catch(console.error)
  .finally(() => {
    console.log("🔻 Closing provider connection...");
    provider.engine.stop(); // Cleanup HDWalletProvider
  });
