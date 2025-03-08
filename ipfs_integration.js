// 1. IPFS INTEGRATION
// Install required packages:
// npm install ipfs-http-client ethers

const { create } = require('ipfs-http-client');
const { ethers } = require('ethers');
const fs = require('fs');

// Connect to IPFS (local or infura gateway)
const ipfs = create({
  host: 'ipfs.infura.io',
  port: 5001,
  protocol: 'https'
});

// 2. SMART CONTRACT (Ethereum Example)
// This is the ABI for a simple smart contract to store IPFS hashes
const contractABI = [
  "function storeComplaint(string memory complaintId, string memory ipfsHash, uint256 timestamp, string memory sentimentScore) public",
  "function getComplaintHash(string memory complaintId) public view returns (string memory ipfsHash, uint256 timestamp, string memory sentimentScore)"
];

// Contract address after deployment
const contractAddress = "0xYourContractAddressHere";

// 3. INTEGRATION WITH YOUR EXISTING PIPELINE
async function processAndStoreComplaint(complaintData) {
  try {
    // Assuming complaintData has already been processed by your NLP/BERT system
    // and contains the sentiment analysis results
    
    // Create a JSON object with the complaint and analysis data
    const complaintRecord = {
      id: complaintData._id.toString(),
      text: complaintData.text,
      timestamp: new Date().toISOString(),
      sentiment: {
        score: complaintData.sentimentScore,
        magnitude: complaintData.sentimentMagnitude,
        category: complaintData.sentimentCategory,
      },
      entities: complaintData.entities || [],
      nlpMetadata: complaintData.nlpMetadata || {}
    };
    
    // Convert the data to a Buffer for IPFS
    const complaintBuffer = Buffer.from(JSON.stringify(complaintRecord));
    
    // Upload to IPFS
    const ipfsResult = await ipfs.add(complaintBuffer);
    const ipfsHash = ipfsResult.path;
    
    console.log(`Complaint stored on IPFS with hash: ${ipfsHash}`);
    
    // Connect to Ethereum network (using Infura, Alchemy, or your preferred provider)
    const provider = new ethers.providers.JsonRpcProvider(process.env.ETHEREUM_RPC_URL);
    const wallet = new ethers.Wallet(process.env.PRIVATE_KEY, provider);
    
    // Create contract instance
    const complaintContract = new ethers.Contract(contractAddress, contractABI, wallet);
    
    // Store the IPFS hash on the blockchain
    const tx = await complaintContract.storeComplaint(
      complaintData._id.toString(),
      ipfsHash,
      Math.floor(Date.now() / 1000),
      complaintData.sentimentScore.toString()
    );
    
    // Wait for transaction confirmation
    await tx.wait();
    
    console.log(`Complaint hash stored on blockchain: ${tx.hash}`);
    
    // Update MongoDB with blockchain reference
    await updateComplaintWithBlockchainRef(
      complaintData._id,
      ipfsHash,
      tx.hash
    );
    
    return {
      success: true,
      ipfsHash,
      transactionHash: tx.hash
    };
  } catch (error) {
    console.error("Error storing complaint on blockchain:", error);
    return {
      success: false,
      error: error.message
    };
  }
}

// Helper function to update MongoDB with blockchain references
async function updateComplaintWithBlockchainRef(complaintId, ipfsHash, transactionHash) {
  // Implement your MongoDB update logic here
  // Example using MongoDB driver:
  /*
  await db.collection('processedComplaints').updateOne(
    { _id: complaintId },
    { 
      $set: { 
        blockchain: {
          ipfsHash,
          transactionHash,
          timestamp: new Date()
        }
      }
    }
  );
  */
}

// 4. VERIFICATION FUNCTION
// This function allows you to verify a complaint's authenticity
async function verifyComplaint(complaintId) {
  try {
    // Connect to provider (read-only is fine for verification)
    const provider = new ethers.providers.JsonRpcProvider(process.env.ETHEREUM_RPC_URL);
    
    // Create contract instance (read-only)
    const complaintContract = new ethers.Contract(contractAddress, contractABI, provider);
    
    // Get the stored hash from blockchain
    const { ipfsHash, timestamp, sentimentScore } = await complaintContract.getComplaintHash(complaintId);
    
    // Fetch the data from IPFS
    const chunks = [];
    for await (const chunk of ipfs.cat(ipfsHash)) {
      chunks.push(chunk);
    }
    
    const complaintData = JSON.parse(Buffer.concat(chunks).toString());
    
    return {
      verified: true,
      blockchainData: {
        ipfsHash,
        timestamp: new Date(timestamp * 1000).toISOString(),
        sentimentScore
      },
      complaintData
    };
  } catch (error) {
    console.error("Error verifying complaint:", error);
    return {
      verified: false,
      error: error.message
    };
  }
}

// Export the functions
module.exports = {
  processAndStoreComplaint,
  verifyComplaint
};