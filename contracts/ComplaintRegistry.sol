pragma solidity ^0.8.17;

contract ComplaintRegistry {
    event ComplaintRegistered(string complaintId, string complaintHash);
    
    // Mapping from complaint ID to its hash
    mapping(string => string) public complaints;
    
    // Owner of the contract
    address public owner;
    
    constructor() {
        owner = msg.sender;
    }
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }
    
    // Store a complaint hash with its ID
    function registerComplaint(string memory complaintId, string memory complaintHash) public onlyOwner {
        complaints[complaintId] = complaintHash;
        emit ComplaintRegistered(complaintId, complaintHash);
    }
    
    // Batch registration for gas optimization
    function batchRegisterComplaints(
        string[] memory complaintIds, 
        string[] memory complaintHashes
    ) public onlyOwner {
        require(complaintIds.length == complaintHashes.length, "Arrays must be of same length");
        
        for (uint i = 0; i < complaintIds.length; i++) {
            complaints[complaintIds[i]] = complaintHashes[i];
            emit ComplaintRegistered(complaintIds[i], complaintHashes[i]);
        }
    }
    
    // Verify if a complaint matches what's stored on the blockchain
    function verifyComplaint(string memory complaintId, string memory complaintHash) 
        public view returns (bool) {
        return keccak256(abi.encodePacked(complaints[complaintId])) == 
               keccak256(abi.encodePacked(complaintHash));
    }
}