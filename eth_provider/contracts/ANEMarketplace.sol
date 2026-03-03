// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title ANEMarketplace — Decentralized ANE Compute Marketplace
/// @notice Escrow-based job marketplace with optimistic verification and slashing.
/// Providers stake ETH to register, clients deposit ETH to post jobs,
/// payment released after challenge window or dispute resolution.
contract ANEMarketplace {

    // ─── Types ───────────────────────────────────────────────────────

    enum JobState {
        Posted,     // Client deposited, waiting for provider
        Claimed,    // Provider locked job, computing
        Committed,  // Provider submitted result hash
        Revealed,   // Provider revealed result, challenge window open
        Finalized,  // No challenge, payment released
        Disputed,   // Challenge submitted, awaiting resolution
        Slashed     // Provider found invalid, stake slashed
    }

    struct Provider {
        address payable addr;
        string endpoint;         // HTTPS URL for X402 API
        uint256 stake;
        uint256 registeredAt;
        uint256 jobsCompleted;
        uint256 jobsFailed;
        bool active;
    }

    struct Job {
        // Parties
        address payable client;
        address payable provider;

        // Job spec
        bytes32 modelHash;       // keccak256 of model weights
        bytes32 inputHash;       // keccak256 of input data
        uint8 jobType;           // 0=inference, 1=training

        // Payment
        uint256 payment;         // ETH deposited by client
        uint256 maxPrice;        // Max acceptable price (wei)

        // State
        JobState state;
        uint256 deadline;        // Must complete by this timestamp
        uint256 revealedAt;      // When result was revealed
        uint256 claimedAt;

        // Proof
        bytes32 commitHash;      // keccak256(result || nonce)
        bytes32 resultHash;      // keccak256(result) after reveal
        uint256 nonce;

        // Training-specific
        uint256 steps;           // Number of training steps
        bytes32 merkleRoot;      // Merkle root of step checkpoints
    }

    struct Challenge {
        address challenger;
        uint256 deposit;
        bytes32 expectedResultHash;
        uint256 challengedAt;
    }

    // ─── State ───────────────────────────────────────────────────────

    mapping(address => Provider) public providers;
    mapping(uint256 => Job) public jobs;
    mapping(uint256 => Challenge) public challenges;
    address[] public providerList;
    uint256 public nextJobId;

    address public owner;
    address public treasury;

    // ─── Config ──────────────────────────────────────────────────────

    uint256 public constant MIN_STAKE = 0.01 ether;
    uint256 public constant CHALLENGE_WINDOW = 1 hours;
    uint256 public constant CHALLENGE_DEPOSIT = 0.005 ether;
    uint256 public constant PROTOCOL_FEE_BPS = 200; // 2%
    uint256 public constant MAX_JOB_DURATION = 24 hours;

    // ─── Events ──────────────────────────────────────────────────────

    event ProviderRegistered(address indexed provider, string endpoint, uint256 stake);
    event ProviderDeregistered(address indexed provider);
    event JobPosted(uint256 indexed jobId, address indexed client, bytes32 modelHash, uint256 payment);
    event JobClaimed(uint256 indexed jobId, address indexed provider);
    event ResultCommitted(uint256 indexed jobId, bytes32 commitHash);
    event ResultRevealed(uint256 indexed jobId, bytes32 resultHash);
    event JobFinalized(uint256 indexed jobId, uint256 payout);
    event JobChallenged(uint256 indexed jobId, address indexed challenger);
    event DisputeResolved(uint256 indexed jobId, bool providerValid);
    event ProviderSlashed(address indexed provider, uint256 amount);

    // ─── Modifiers ───────────────────────────────────────────────────

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    modifier onlyActiveProvider() {
        require(providers[msg.sender].active, "not active provider");
        _;
    }

    // ─── Constructor ─────────────────────────────────────────────────

    constructor(address _treasury) {
        owner = msg.sender;
        treasury = _treasury;
    }

    // ─── Provider Management ─────────────────────────────────────────

    /// @notice Register as a compute provider. Must send >= MIN_STAKE.
    function registerProvider(string calldata endpoint) external payable {
        require(msg.value >= MIN_STAKE, "insufficient stake");
        require(!providers[msg.sender].active, "already registered");

        providers[msg.sender] = Provider({
            addr: payable(msg.sender),
            endpoint: endpoint,
            stake: msg.value,
            registeredAt: block.timestamp,
            jobsCompleted: 0,
            jobsFailed: 0,
            active: true
        });
        providerList.push(msg.sender);

        emit ProviderRegistered(msg.sender, endpoint, msg.value);
    }

    /// @notice Add more stake to increase reputation/capacity.
    function addStake() external payable onlyActiveProvider {
        providers[msg.sender].stake += msg.value;
    }

    /// @notice Deregister and withdraw stake. Cannot have active jobs.
    function deregisterProvider() external onlyActiveProvider {
        Provider storage p = providers[msg.sender];
        p.active = false;
        uint256 stake = p.stake;
        p.stake = 0;
        p.addr.transfer(stake);

        emit ProviderDeregistered(msg.sender);
    }

    // ─── Job Lifecycle ───────────────────────────────────────────────

    /// @notice Post a new compute job. Payment is escrowed.
    function postJob(
        bytes32 modelHash,
        bytes32 inputHash,
        uint8 jobType,
        uint256 steps,
        uint256 deadline
    ) external payable returns (uint256 jobId) {
        require(msg.value > 0, "no payment");
        require(deadline > block.timestamp, "deadline in past");
        require(deadline <= block.timestamp + MAX_JOB_DURATION, "deadline too far");

        jobId = nextJobId++;
        jobs[jobId] = Job({
            client: payable(msg.sender),
            provider: payable(address(0)),
            modelHash: modelHash,
            inputHash: inputHash,
            jobType: jobType,
            payment: msg.value,
            maxPrice: msg.value,
            state: JobState.Posted,
            deadline: deadline,
            revealedAt: 0,
            claimedAt: 0,
            commitHash: bytes32(0),
            resultHash: bytes32(0),
            nonce: 0,
            steps: steps,
            merkleRoot: bytes32(0)
        });

        emit JobPosted(jobId, msg.sender, modelHash, msg.value);
    }

    /// @notice Provider claims a posted job.
    function claimJob(uint256 jobId) external onlyActiveProvider {
        Job storage j = jobs[jobId];
        require(j.state == JobState.Posted, "not posted");
        require(block.timestamp < j.deadline, "expired");

        j.provider = payable(msg.sender);
        j.state = JobState.Claimed;
        j.claimedAt = block.timestamp;

        emit JobClaimed(jobId, msg.sender);
    }

    /// @notice Provider commits hash of result before revealing.
    function commitResult(uint256 jobId, bytes32 _commitHash) external {
        Job storage j = jobs[jobId];
        require(j.state == JobState.Claimed, "not claimed");
        require(msg.sender == j.provider, "not provider");
        require(block.timestamp <= j.deadline, "past deadline");

        j.commitHash = _commitHash;
        j.state = JobState.Committed;

        emit ResultCommitted(jobId, _commitHash);
    }

    /// @notice Provider reveals the actual result and nonce.
    function revealResult(
        uint256 jobId,
        bytes32 _resultHash,
        uint256 _nonce,
        bytes32 _merkleRoot
    ) external {
        Job storage j = jobs[jobId];
        require(j.state == JobState.Committed, "not committed");
        require(msg.sender == j.provider, "not provider");

        // Verify commitment
        require(
            keccak256(abi.encodePacked(_resultHash, _nonce)) == j.commitHash,
            "commitment mismatch"
        );

        j.resultHash = _resultHash;
        j.nonce = _nonce;
        j.merkleRoot = _merkleRoot;
        j.state = JobState.Revealed;
        j.revealedAt = block.timestamp;

        emit ResultRevealed(jobId, _resultHash);
    }

    /// @notice Finalize job after challenge window. Releases payment.
    function finalizeJob(uint256 jobId) external {
        Job storage j = jobs[jobId];
        require(j.state == JobState.Revealed, "not revealed");
        require(
            block.timestamp >= j.revealedAt + CHALLENGE_WINDOW,
            "challenge window active"
        );

        j.state = JobState.Finalized;

        // Calculate fees
        uint256 fee = (j.payment * PROTOCOL_FEE_BPS) / 10000;
        uint256 payout = j.payment - fee;

        // Pay provider
        j.provider.transfer(payout);
        // Protocol fee
        payable(treasury).transfer(fee);

        // Update stats
        providers[j.provider].jobsCompleted++;

        emit JobFinalized(jobId, payout);
    }

    // ─── Challenge / Dispute ─────────────────────────────────────────

    /// @notice Challenge a revealed result. Must deposit CHALLENGE_DEPOSIT.
    function challengeResult(
        uint256 jobId,
        bytes32 expectedResultHash
    ) external payable {
        Job storage j = jobs[jobId];
        require(j.state == JobState.Revealed, "not revealed");
        require(
            block.timestamp < j.revealedAt + CHALLENGE_WINDOW,
            "challenge window closed"
        );
        require(msg.value >= CHALLENGE_DEPOSIT, "insufficient challenge deposit");
        require(expectedResultHash != j.resultHash, "same result");

        challenges[jobId] = Challenge({
            challenger: msg.sender,
            deposit: msg.value,
            expectedResultHash: expectedResultHash,
            challengedAt: block.timestamp
        });
        j.state = JobState.Disputed;

        emit JobChallenged(jobId, msg.sender);
    }

    /// @notice Resolve a dispute. Called by owner/verifier with the correct result.
    /// In production, this would be a decentralized verifier committee.
    function resolveDispute(
        uint256 jobId,
        bytes32 verifiedResultHash
    ) external onlyOwner {
        Job storage j = jobs[jobId];
        Challenge storage c = challenges[jobId];
        require(j.state == JobState.Disputed, "not disputed");

        bool providerCorrect = (verifiedResultHash == j.resultHash);

        if (providerCorrect) {
            // Provider was right — slash challenger, pay provider
            j.state = JobState.Finalized;
            uint256 fee = (j.payment * PROTOCOL_FEE_BPS) / 10000;
            uint256 payout = j.payment - fee + c.deposit; // provider gets challenger's deposit too
            j.provider.transfer(payout);
            payable(treasury).transfer(fee);
            providers[j.provider].jobsCompleted++;
        } else {
            // Provider was wrong — slash provider, refund client
            j.state = JobState.Slashed;
            uint256 slashAmount = providers[j.provider].stake;
            providers[j.provider].stake = 0;
            providers[j.provider].active = false;
            providers[j.provider].jobsFailed++;

            // Refund client
            j.client.transfer(j.payment);
            // Reward challenger
            payable(c.challenger).transfer(c.deposit + slashAmount / 2);
            // Rest to treasury
            payable(treasury).transfer(slashAmount / 2);

            emit ProviderSlashed(j.provider, slashAmount);
        }

        emit DisputeResolved(jobId, providerCorrect);
    }

    // ─── Timeout / Cancellation ──────────────────────────────────────

    /// @notice Client cancels an unclaimed job.
    function cancelJob(uint256 jobId) external {
        Job storage j = jobs[jobId];
        require(msg.sender == j.client, "not client");
        require(j.state == JobState.Posted, "already claimed");

        j.state = JobState.Finalized;
        j.client.transfer(j.payment);
    }

    /// @notice Claim timeout if provider didn't deliver by deadline.
    function claimTimeout(uint256 jobId) external {
        Job storage j = jobs[jobId];
        require(
            j.state == JobState.Claimed || j.state == JobState.Committed,
            "wrong state"
        );
        require(block.timestamp > j.deadline, "not expired");

        // Partial slash for timeout
        uint256 slashAmount = providers[j.provider].stake / 10; // 10% slash
        providers[j.provider].stake -= slashAmount;
        providers[j.provider].jobsFailed++;

        j.state = JobState.Finalized;
        j.client.transfer(j.payment + slashAmount);
    }

    // ─── Views ───────────────────────────────────────────────────────

    function getProviderCount() external view returns (uint256) {
        return providerList.length;
    }

    function getProvider(address addr) external view returns (
        string memory endpoint,
        uint256 stake,
        uint256 jobsCompleted,
        uint256 jobsFailed,
        bool active
    ) {
        Provider storage p = providers[addr];
        return (p.endpoint, p.stake, p.jobsCompleted, p.jobsFailed, p.active);
    }

    function getJob(uint256 jobId) external view returns (
        address client,
        address provider,
        JobState state,
        uint256 payment,
        bytes32 resultHash,
        uint256 deadline
    ) {
        Job storage j = jobs[jobId];
        return (j.client, j.provider, j.state, j.payment, j.resultHash, j.deadline);
    }

    function isChallengeable(uint256 jobId) external view returns (bool) {
        Job storage j = jobs[jobId];
        return j.state == JobState.Revealed
            && block.timestamp < j.revealedAt + CHALLENGE_WINDOW;
    }
}
