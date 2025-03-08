// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

/**
 * @title ComplaintRegistry
 * @dev Stores IPFS hashes of processed complaint data
 */
contract ComplaintRegistry {
    struct Complaint {
        string ipfsHash;
        uint256 timestamp;
        string sentimentScore;
        bool exists;
    }
    
    // Mapping from complaint ID to Complaint struct
    mapping(string => Complaint) private complaints;
    
    // Array to keep track of all complaint IDs
    string[] private complaintIds;
    
    // Event emitted when a new complaint is stored
    event ComplaintStored(
        string indexed complaintId,
        string ipfsHash,
        uint256 timestamp,
        string sentimentScore
    );
    
    /**
     * @dev Store a new complaint record
     * @param complaintId Unique identifier for the complaint
     * @param ipfsHash IPFS hash where the full complaint data is stored
     * @param timestamp Time when the complaint was processed
     * @param sentimentScore Sentiment analysis score as a string
     */
    function storeComplaint(
        string memory complaintId,
        string memory ipfsHash,
        uint256 timestamp,
        string memory sentimentScore
    ) public {
        // Check if complaint already exists
        require(!complaints[complaintId].exists, "Complaint already exists");
        
        // Store complaint data
        complaints[complaintId] = Complaint({
            ipfsHash: ipfsHash,
            timestamp: timestamp,
            sentimentScore: sentimentScore,
            exists: true
        });
        
        // Add to the list of complaints
        complaintIds.push(complaintId);
        
        // Emit event
        emit ComplaintStored(complaintId, ipfsHash, timestamp, sentimentScore);
    }
    
    /**
     * @dev Retrieve complaint information
     * @param complaintId Unique identifier for the complaint
     * @return ipfsHash IPFS hash where the full complaint data is stored
     * @return timestamp Time when the complaint was processed
     * @return sentimentScore Sentiment analysis score
     */
    function getComplaintHash(string memory complaintId) 
        public 
        view 
        returns (string memory ipfsHash, uint256 timestamp, string memory sentimentScore) 
    {
        // Check if complaint exists
        require(complaints[complaintId].exists, "Complaint does not exist");
        
        Complaint memory complaint = complaints[complaintId];
        return (complaint.ipfsHash, complaint.timestamp, complaint.sentimentScore);
    }
    
    /**
     * @dev Get the total number of complaints
     * @return Total number of complaints stored
     */
    function getTotalComplaints() public view returns (uint256) {
        return complaintIds.length;
    }
    
    /**
     * @dev Get complaint ID by index
     * @param index Index in the complaints array
     * @return Complaint ID at the specified index
     */
    function getComplaintIdByIndex(uint256 index) public view returns (string memory) {
        require(index < complaintIds.length, "Index out of bounds");
        return complaintIds[index];
    }
}