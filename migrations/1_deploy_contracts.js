const ComplaintRegistry = artifacts.require("ComplaintRegistry");

module.exports = function(deployer) {
  deployer.deploy(ComplaintRegistry);
};
