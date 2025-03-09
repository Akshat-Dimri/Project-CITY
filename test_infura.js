const Web3 = require('web3');
require('dotenv').config();

const web3 = new Web3(`https://sepolia.infura.io/v3/${process.env.INFURA_PROJECT_ID}`);

web3.eth.getBlockNumber()
  .then(block => console.log("Latest block number:", block))
  .catch(err => console.error("Infura connection error:", err));

  