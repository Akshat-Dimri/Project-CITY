require('dotenv').config();
const { MongoClient } = require('mongodb');
const Web3 = require('web3');
const crypto = require('crypto');
const HDWalletProvider = require('@truffle/hdwallet-provider');

// Contract ABI and address
const contractJSON = require('./build/contracts/ComplaintRegistry.json');

// MongoDB Configuration
const mongoURI = process.env.MONGODB_URI;
const dbName = process.env.MONGODB_DB;
const collectionName = process.env.MONGODB_COLLECTION;

// Blockchain Configuration
const provider = new HDWalletProvider(
  process.env.MNEMONIC,
  `https://sepolia.infura.io/v3/${process.env.INFURA_PROJECT_ID}`
);
const web3 = new Web3(provider);

// Function to hash complaint data
function hashComplaint(complaint) {
  return crypto
    .createHash('sha256')
    .update(JSON.stringify(complaint))
    .digest('hex');
}

// Function to verify a complaint on the blockchain
async function verifyComplaint(complaintId) {
  try {
    // Connect to MongoDB
    const mongoClient = new MongoClient(mongoURI);
    await mongoClient.connect();
    const db = mongoClient.db(dbName);
    const collection = db.collection(collectionName);
    
    // Get the complaint from MongoDB
    const complaint = await collection.findOne({ _id: MongoDB.ObjectId(complaintId) });
    
    if (!complaint) {
      console.log(`Complaint with ID ${complaintId} not found in MongoDB`);
      await mongoClient.close();
      return false;
    }
    
    // Get the deployed contract
    const networkId = await web3.eth.net.getId();
    const deployedNetwork = contractJSON.networks[networkId];
    
    if (!deployedNetwork) {
      console.error('Contract not deployed to detected network');
      await mongoClient.close();
      return false;
    }
    
    const contract = new web3.eth.Contract(
      contractJSON.abi,
      deployedNetwork.address
    );
    
    // Calculate the hash of the complaint
    const complaintHash = hashComplaint(complaint);
    
    // Verify on blockchain
    const isVerified = await contract.methods
      .verifyComplaint(complaintId, complaintHash)
      .call();
    
    console.log(`Verification result for complaint ${complaintId}: ${isVerified}`);
    
    await mongoClient.close();
    return isVerified;
    
  } catch (error) {
    console.error('Error verifying complaint:', error);
    return false;
  }
}

// Example usage
if (process.argv.length >= 3) {
  const complaintId = process.argv[2];
  verifyComplaint(complaintId)
    .then(result => {
      console.log(`Verification result: ${result}`);
      process.exit(0);
    })
    .catch(error => {
      console.error(error);
      process.exit(1);
    });
} else {
  console.log('Please provide a complaint ID to verify');
  process.exit(1);
}