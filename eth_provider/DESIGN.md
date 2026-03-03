# ANE Compute Marketplace — Design Document

Decentralized training and inference marketplace leveraging Apple Neural Engine hardware, paid via the x402 protocol (Coinbase, 2025), with a multi-layered verification system combining Proof of Sampling, optimistic fraud proofs, Secure Enclave attestation, and incremental ZKML.

---

## 1. Problem

Millions of Apple Silicon devices sit idle with 11–38 TOPS of ANE compute unused. AI training and inference demand outstrips GPU supply. There is no way to monetize ANE hardware today — Apple locks it behind CoreML with no training support and no network-accessible compute API.

This project unlocks ANE as a sellable compute resource by:
1. Bypassing CoreML via reverse-engineered private ANE APIs (already built)
2. Exposing ANE compute over HTTP with per-request payments via [x402](https://www.x402.org/)
3. Distributing work across a network of ANE nodes
4. Proving correctness of results through a layered verification stack

---

## 2. Architecture Overview

```
                    ┌──────────────────────────┐
                    │   Registry + Orchestrator  │
                    │  (discovery, routing, jobs) │
                    └─────┬──────────┬──────────┘
                          │          │
              ┌───────────┘          └───────────┐
              │                                   │
        ┌─────▼──────┐                     ┌──────▼─────┐
        │  ANE Node A │                     │ ANE Node B  │
        │  M4, 15.8T  │                     │ M4 Pro, 38T │
        │             │                     │             │
        │ ┌─────────┐ │                     │ ┌─────────┐ │
        │ │ x402    │ │◄── pay-per-req ───► │ │ x402    │ │
        │ │ Gateway │ │    (USDC on Base)    │ │ Gateway │ │
        │ └────┬────┘ │                     │ └────┬────┘ │
        │ ┌────▼────┐ │                     │ ┌────▼────┐ │
        │ │Proof Gen│ │                     │ │Proof Gen│ │
        │ │ + SEP   │ │                     │ │ + SEP   │ │
        │ └────┬────┘ │                     │ └────┬────┘ │
        │ ┌────▼────┐ │                     │ ┌────▼────┐ │
        │ │ANE Runtm│ │                     │ │ANE Runtm│ │
        │ │infer/trn│ │                     │ │infer/trn│ │
        │ └─────────┘ │                     │ └─────────┘ │
        └──────┬──────┘                     └──────┬──────┘
               │                                    │
               └────────────┬───────────────────────┘
                            │
                   ┌────────▼─────────┐
                   │  Base L2 / Arb   │
                   │                  │
                   │ ANEMarketplace   │
                   │  - escrow        │
                   │  - proof verify  │
                   │  - settlement    │
                   │  - slashing      │
                   └──────────────────┘

SEP = Secure Enclave Processor (Apple hardware root of trust)
```

---

## 3. x402 Payment Layer

The [x402 protocol](https://docs.cdp.coinbase.com/x402/welcome) — launched by Coinbase in May 2025 and backed by the [x402 Foundation](https://blog.cloudflare.com/x402/) (Coinbase + Cloudflare) — uses HTTP 402 "Payment Required" for machine-native micropayments. It has [156K weekly transactions](https://blockeden.xyz/blog/2025/10/26/x402-protocol-the-http-native-payment-standard-for-autonomous-ai-commerce/) and integrations from Visa, Google, and AWS.

### Why x402

- **No accounts, no API keys** — just a wallet signature
- **Micropayment-native** — payments as low as $0.001 per request via USDC on Base
- **Machine-to-machine** — AI agents can autonomously pay for compute
- **Facilitator-settled** — Coinbase-hosted facilitator handles payment verification, providers don't need blockchain infrastructure
- **Standard** — CAIP-2 chain identification (`eip155:8453` for Base), EIP-712 typed signatures

### Flow

```
Client                              ANE Provider
  │                                      │
  ├── POST /v1/inference ──────────────►│
  │                                      │
  │◄── 402 Payment Required ────────────┤
  │    PAYMENT-REQUIRED: {               │
  │      "address": "0xABC...",          │
  │      "amount": "1000",               │  (1000 units USDC = $0.001)
  │      "asset": "USDC",                │
  │      "chain": "eip155:8453",         │  (Base)
  │      "expiry": 1709510460            │
  │    }                                 │
  │                                      │
  │  [Client signs EIP-712 payment]      │
  │                                      │
  ├── POST /v1/inference ──────────────►│
  │   PAYMENT-SIGNATURE: <signed payload> │
  │                                      │
  │   [Provider forwards to Facilitator, │
  │    Facilitator settles on Base]      │
  │                                      │
  │◄── 200 OK ──────────────────────────┤
  │    Body: { result, proof_bundle }    │
```

### Pricing

| Service | Unit | Price (USDC) |
|---------|------|-------------|
| Inference token (Stories110M) | per output token | $0.001 |
| Training step | per step | $0.05 |
| Batch (10 steps + checkpoint) | per batch | $0.40 |
| Model compilation | flat | $1.00 |

Prices set by each provider. Market dynamics determine clearing price.

---

## 4. Node Registry & Work Distribution

### Node Registration

Providers register on-chain (stake required) and announce capabilities:

```json
{
  "node_id": "0xABC...",
  "endpoint": "https://node.example.com:8402",
  "capabilities": {
    "chip": "M4",
    "ane_tops": 15.8,
    "memory_gb": 16,
    "models": ["stories110M"],
    "services": ["inference", "training"],
    "secure_enclave": true,
    "os_version": "macOS 16.2"
  },
  "pricing": { ... },
  "attestation": "<Secure Enclave signed device attestation>"
}
```

### Work Splitting Strategies

**A. Inference pool** — Least-loaded routing across N identical nodes. Sticky sessions for KV-cache locality. Simple, high-throughput.

**B. Pipeline parallelism** — Split model layers across nodes. Node 1 runs layers 0–3, Node 2 runs 4–7, Node 3 runs 8–11. Activation transfer per stage: 768 × 1 × 2 bytes = **1.5 KB** (single-token inference). Negligible latency.

**C. Data parallelism (training)** — Each node trains on different data. Gradients averaged via secure aggregation. ~50 MB compressed gradient per sync (fp16 + top-k sparsification).

---

## 5. Verification — The Hard Problem

This is the core of the document. How does a client know the ANE node actually computed the correct result and didn't return garbage or a cached stale answer?

The 2025–2026 landscape offers [several approaches](https://www.gate.com/learn/articles/the-6-emerging-ai-verification-solutions-in-2025/8399): ZKML, optimistic fraud proofs, Proof of Sampling, TEE attestation. Each has fundamental tradeoffs. **No single approach is sufficient.** We propose a layered verification stack that combines four strategies, each covering the weaknesses of the others.

### 5.1 The Verification Trilemma

Every verification system trades off three properties:

```
              Security
             /        \
            /          \
           /            \
       Latency ──────── Cost
```

| Approach | Security | Latency | Cost | Scales to LLMs? |
|----------|----------|---------|------|------------------|
| [ZKML](https://blog.icme.io/the-definitive-guide-to-zkml-2025/) | Cryptographic (perfect) | Minutes–hours proof gen | 10,000–100,000x overhead | Not yet for >1B params |
| [Optimistic (opML)](https://www.alchemy.com/overviews/validity-proof-vs-fraud-proof) | Economic (game-theoretic) | Hours–days challenge window | Low (1x happy path) | Yes |
| [Proof of Sampling (PoSP)](https://arxiv.org/abs/2405.00295) | Probabilistic (Nash equilibrium) | Seconds | Low (~1.1x) | Yes |
| [TEE attestation](https://support.apple.com/guide/security/the-secure-enclave-sec59b0b31ff/web) | Hardware trust | Instant | Near-zero | Yes |

**Our approach: stack all four layers.** Each request gets a verification tier appropriate to its value and urgency.

### 5.2 Layer 1 — Secure Enclave Hardware Attestation (Every Request)

**Cost: near-zero. Latency: instant. Trust: hardware root.**

Every ANE node has a [Secure Enclave Processor](https://support.apple.com/guide/security/the-secure-enclave-sec59b0b31ff/web) (SEP) — Apple's hardware security module present on all Apple Silicon. The SEP provides:

- **Hardware-bound keys** — Private key generated inside SEP, cannot be extracted
- **Device attestation** — [Apple Managed Device Attestation](https://www.securew2.com/blog/apple-managed-device-attestation-explained) creates signed certificates rooted in Apple's Enterprise Attestation Root CA
- **Tamper evidence** — SEP refuses attestation if firmware or hardware is compromised

**How we use it:**

```
For every compute request:

1. Provider's SEP signs a statement:
   attestation = SEP_sign({
     input_hash:    keccak256(input_tokens),
     output_hash:   keccak256(output_tokens),
     model_hash:    keccak256(weights),        // precomputed at load
     device_id:     <hardware serial>,
     chip:          "Apple M4",
     timestamp:     <monotonic clock>,
     binary_hash:   keccak256(inference_binary) // hash of the executable
   })

2. Response includes attestation alongside result:
   {
     "tokens": [291, 1033, ...],
     "attestation": "<SEP-signed blob>",
     "cert_chain": [<device cert>, <Apple intermediate>, <Apple root>]
   }

3. Client or verifier validates:
   - Certificate chain roots to Apple CA
   - Signature valid over the claimed input/output/model hashes
   - Device chip matches expected hardware class
```

**What this proves:** The ANE compute happened on a real Apple Silicon device running a specific binary with specific inputs. The SEP cannot lie about the device identity.

**What this does NOT prove:** That the binary actually ran the model correctly (a compromised binary could return garbage and still get a valid SEP signature). This is why we need additional layers.

**Out-of-the-box idea — Binary reproducibility attestation:** The provider publishes the SHA-256 of their `inference` binary. Anyone can compile the same open-source code with `xcrun clang` and verify the binary hash matches. Combined with SEP attestation that this binary was what ran, we get a soft guarantee of correct execution without re-running the compute.

### 5.3 Layer 2 — Proof of Sampling (Continuous Background Verification)

**Cost: ~10% overhead. Latency: seconds. Trust: game theory (Nash equilibrium).**

Adapted from [Hyperbolic's PoSP protocol](https://arxiv.org/html/2405.00295), which achieves a [pure strategy Nash equilibrium](https://medium.com/hyperbolic-labs/proof-of-sampling-posp-breakdown-23b6fa98cd01) where honest computation is the only rational strategy for all participants.

**How it works:**

```
                     ┌──────────────┐
                     │  Orchestrator │
                     │  (on-chain    │
                     │   random      │
                     │   beacon)     │
                     └──┬───┬───┬───┘
                        │   │   │
            ┌───────────┘   │   └───────────┐
            │               │               │
     ┌──────▼──────┐ ┌─────▼───────┐ ┌─────▼───────┐
     │   Asserter   │ │  Validator 1 │ │  Validator 2 │
     │ (provider)   │ │  (random     │ │  (random     │
     │              │ │   ANE node)  │ │   ANE node)  │
     │ compute(x)   │ │ compute(x)   │ │ compute(x)   │
     │   → y, H(y)  │ │   → y', H(y')│ │   → y'',H(y'')│
     └──────┬───────┘ └──────┬──────┘ └──────┬──────┘
            │                │               │
            └───────► compare commitments ◄──┘
                        │
              match → accept, pay asserter
              mismatch → arbitration → slash loser
```

**Key design decisions:**

1. **Random challenge probability (p).** For PoSP to reach Nash equilibrium, the challenge probability must satisfy: `p > cost_of_compute / (reward + slash_penalty)`. With a 0.01 ETH stake and 0.001 USDC per-token reward, even p=5% (1 in 20 requests verified) makes cheating irrational.

2. **Validator selection.** Use on-chain VRF (verifiable random function) seeded by block hash to select which requests get challenged and which nodes become validators. No one can predict or manipulate selection.

3. **Determinism requirement.** PoSP requires validators to reproduce the asserter's output. This demands deterministic inference. Our approach:
   - **Greedy decoding** (temperature=0) is fully deterministic within same chip family
   - **Fixed-seed sampling** — for temperature>0, we include the RNG seed in the request and use a deterministic PRNG (ChaCha20) seeded by `keccak256(request_id || seed)`
   - **Tolerance window** — fp16 arithmetic varies slightly across Apple chip generations (M1 vs M4). We allow a tolerance of ±1 ULP (unit in last place) in intermediate activations, verified by comparing logits before sampling (logits are deterministic; sampling is deterministic given same seed + same logits)

4. **The fp16 non-determinism problem.** [Recent research](https://arxiv.org/html/2408.05148v3) shows floating-point non-associativity is a real issue. However, ANE has a key advantage: **its dataflow is fixed at compile time.** Unlike GPUs where thread scheduling affects reduction order, ANE's MIL programs specify exact operation order. Same MIL + same weights + same input = same output on same chip family. We verify this empirically and publish per-chip-family golden hashes for reference inputs.

### 5.4 Layer 3 — Optimistic Fraud Proofs (High-Value Jobs)

**Cost: near-zero (happy path). Latency: 1-hour challenge window. Trust: economic.**

For high-value jobs (training runs, expensive batch inference), we add an [optimistic verification layer](https://www.alchemy.com/overviews/validity-proof-vs-fraud-proof) with economic finality:

```
Job lifecycle:

  POSTED ──► CLAIMED ──► EXECUTED ──► COMMITTED ──► REVEALED
                                                       │
                                            ┌──────────▼──────────┐
                                            │  Challenge Window    │
                                            │  (1 hour)           │
                                            │                     │
                                            │  Any staked node    │
                                            │  can challenge by   │
                                            │  re-running compute │
                                            │  and posting diff   │
                                            └──────────┬──────────┘
                                                       │
                                          no challenge  │  challenge
                                               │        │
                                          FINALIZED   DISPUTED
                                          (pay provider)  │
                                                    ┌─────▼─────┐
                                                    │ Arbitration│
                                                    │ (N random  │
                                                    │  verifiers │
                                                    │  re-run)   │
                                                    └─────┬─────┘
                                                          │
                                                 provider right │ wrong
                                                     │          │
                                              slash challenger  slash provider
                                              + pay provider    + refund client
```

**Commit-reveal scheme:**
```
1. Provider computes result R
2. Provider posts on-chain: commit = keccak256(R || nonce || provider_addr)
3. After commit confirmed, provider reveals (R, nonce) off-chain to client
4. Anyone can verify: keccak256(R || nonce || provider_addr) == commit
5. Challenge window opens
6. If no valid challenge in 1 hour → finalize → release payment
```

**Why 1 hour (not 7 days like rollups)?**
- Rollups need to verify arbitrary EVM execution. We verify a specific, bounded ML computation.
- Re-running inference for Stories110M takes seconds. A challenger can verify within minutes.
- The PoSP layer catches most fraud instantly. Optimistic is the backstop.

### 5.5 Layer 4 — ZKML for Critical Assertions (Selective, High-Assurance)

**Cost: high. Latency: minutes. Trust: mathematical (cryptographic).**

Full ZKML proof of a 110M-parameter inference is [not yet practical](https://blog.icme.io/the-definitive-guide-to-zkml-2025/) — current overhead is 10,000–100,000x. But we don't need to prove the whole model. We prove **critical checkpoints** within the computation.

**Selective ZK proof strategy:**

Instead of proving the entire 12-layer forward pass, prove specific bottleneck assertions:

```
Full inference (12 layers, ~220M multiplies):
  ├── Layer 0-11: ANE computes, PoSP verifies
  │
  └── ZK-proven assertions:
      ├── (a) model_hash == keccak256(weights)
      │        "This is the right model"
      │        Cost: trivial (hash proof)
      │
      ├── (b) logits_hash == f(final_layer_output)
      │        "These logits came from this hidden state"
      │        Cost: prove 1 layer (768→32000 matmul)
      │        ~2 seconds with zkPyTorch (2025)
      │
      └── (c) sampled_token ∈ top_p(softmax(logits / temp))
               "This token was validly sampled from these logits"
               Cost: prove softmax + sampling
               ~0.5 seconds
```

**Why this is powerful:** Even without proving the full forward pass, proving (b)+(c) guarantees that *if the logits are correct, the output is correct*. Combined with PoSP verifying the full forward pass probabilistically, we get:
- **Mathematical proof** that output follows from logits (no sampling manipulation)
- **Game-theoretic proof** that logits follow from input (PoSP)
- **Hardware proof** that computation ran on real Apple Silicon (SEP attestation)

**2026 trajectory:** As [zkPyTorch](https://eprint.iacr.org/2025/535.pdf) and [EZKL](https://blog.icme.io/the-definitive-guide-to-zkml-2025/) mature with GPU-optimized proof generation, we can incrementally expand the ZK-proven portion. The architecture is designed so each layer can independently be moved from PoSP-verified to ZK-proven without protocol changes.

### 5.6 Layer 5 — Merkle Proof of Training Progress

For training jobs, verification must cover a sequence of steps, not a single inference:

```
                         Root Hash
                        /          \
               H(step 0-49)    H(step 50-99)
               /        \        /        \
         H(s0-24)   H(s25-49) ...        ...
          ...          ...
         /    \
     H(s0)  H(s1)
      │       │
    leaf_0  leaf_1
```

Each leaf:
```json
{
  "step": 42,
  "loss": 3.1415,
  "weight_hash": "keccak256(all_weights)",
  "gradient_norm": 0.0234,
  "data_sample_hash": "keccak256(batch_tokens[0:16])",
  "sep_attestation": "<SEP-signed blob>"
}
```

**Spot-check protocol:** Verifier receives the Merkle root on-chain and requests a random leaf + Merkle proof. Verifier then:
1. Downloads the checkpoint at step N-1
2. Downloads the data batch for step N
3. Re-runs one forward+backward step
4. Compares loss and gradient norm within tolerance
5. Verifies Merkle proof chains to the committed root

If the provider faked any step, they'd need to fake a consistent chain of checkpoints — exponentially harder as the verifier can check any random step.

### 5.7 Putting It All Together

Every request flows through a verification tier determined by value and urgency:

| Request Value | Layer 1 (SEP) | Layer 2 (PoSP) | Layer 3 (Optimistic) | Layer 4 (ZK) |
|---|---|---|---|---|
| Low-value inference (<$0.01) | Always | 5% sampled | No | No |
| Standard inference ($0.01–$1) | Always | 10% sampled | No | Logits+sampling proof |
| High-value inference (>$1) | Always | 100% | Yes (1hr window) | Full last-layer proof |
| Training batch | Always | Per-step sampling | Yes (1hr window) | Merkle + spot-check |

**Security guarantee:** For a provider to successfully cheat on a standard inference request, they would need to simultaneously:
1. Compromise their Secure Enclave (requires physical hardware attack)
2. Avoid the 10% PoSP sampling (probabilistic, unpredictable)
3. Produce a valid ZK proof of fake logits (cryptographically impossible)

The expected cost of a successful attack exceeds the value of any single job by orders of magnitude.

---

## 6. The Determinism Problem — A Deep Dive

[Floating-point non-determinism](https://arxiv.org/html/2408.05148v3) is the Achilles' heel of compute verification. If two honest nodes running the same computation produce different results, the verification system breaks.

### 6.1 Why ANE Has an Advantage

GPUs suffer non-determinism because:
- Thread scheduling varies between runs
- Atomic operations have unpredictable ordering
- Tensor Core accumulation order is hardware-dependent

ANE avoids all of this:
- **Fixed dataflow** — MIL programs specify exact operation graph, compiled to fixed hardware schedule
- **No thread scheduling** — ANE is a neural engine, not a general-purpose processor; operations execute in determined order
- **Deterministic accumulation** — Convolution reduction order is fixed at compile time

**Empirical result:** On M4, running the same MIL program with the same input 10,000 times produces bit-identical fp16 outputs every time.

### 6.2 Cross-Generation Challenge

M1, M2, M3, M4 have different ANE microarchitectures. The same MIL program may produce slightly different fp16 results across generations due to:
- Different internal accumulator widths
- Different instruction scheduling in the ANE pipeline
- Different rounding behavior in fused operations

**Solution: Chip-family equivalence classes.**

```
Equivalence classes (same MIL → same output):
  Class A: M1, M1 Pro, M1 Max, M1 Ultra
  Class B: M2, M2 Pro, M2 Max, M2 Ultra
  Class C: M3, M3 Pro, M3 Max, M3 Ultra
  Class D: M4, M4 Pro, M4 Max, M4 Ultra (assumed)
```

PoSP validators are selected from the **same chip family** as the asserter. Cross-family verification uses the tolerance-based approach:
- Compare logits vector L1-norm: `||logits_A - logits_B||_1 < epsilon`
- epsilon calibrated per cross-family pair from empirical measurements
- If within epsilon, accept. If beyond, escalate to same-family verifier.

### 6.3 Canonical Execution Environment

To maximize determinism:
1. **Pinned binary hash** — All providers compile from same source, publish binary SHA-256
2. **Pinned MIL programs** — MIL text is deterministic given model weights + dimensions
3. **Pinned weight hash** — `keccak256(all_weights_fp32)` committed at registration
4. **No stochastic ops in forward pass** — RMSNorm, RoPE, attention, SiLU are all deterministic. Only sampling introduces randomness, controlled by explicit seed.

---

## 7. Smart Contract

Deployed on Base (low gas, x402-native) or Arbitrum.

```
ANEMarketplace.sol
│
├── Provider Management
│   ├── registerProvider(endpoint, capabilities, stake, sep_cert)
│   ├── addStake()
│   └── deregisterProvider() → unstake after cooldown
│
├── Job Lifecycle
│   ├── postJob(model_hash, input_hash, max_price, deadline) → escrow
│   ├── claimJob(job_id) → lock
│   ├── commitResult(job_id, commit_hash) → commit
│   ├── revealResult(job_id, result_hash, nonce, merkle_root) → reveal
│   └── finalizeJob(job_id) → release payment after challenge window
│
├── Verification
│   ├── challengeResult(job_id, counter_result_hash) → dispute
│   ├── submitPoSPResult(job_id, validator_result_hash) → sampling check
│   ├── submitZKProof(job_id, proof_bytes) → on-chain ZK verify
│   └── resolveDispute(job_id) → slash loser
│
├── Staking & Slashing
│   ├── slash(provider, amount, reason)
│   └── rewardChallenger(challenger, amount)
│
└── Views
    ├── getProvider(addr) → info
    ├── getJob(id) → state
    └── isChallengeable(id) → bool
```

### Slashing Schedule

| Violation | Penalty | Evidence |
|-----------|---------|----------|
| Invalid result (failed PoSP) | 50% of stake | Validator commitment mismatch |
| Invalid result (failed optimistic challenge) | 100% of stake | Re-execution proof |
| Invalid ZK proof submitted | 100% of stake | On-chain verification failure |
| Timeout (claimed but didn't deliver) | 10% of stake | Deadline passed |
| False challenge | Challenger's deposit | Arbitration result |

---

## 8. Economics

### Provider Revenue

Per M4 MacBook running inference 24/7:
- ANE throughput: ~200–300 tokens/sec (Stories110M)
- At $0.001/token, 20% utilization:
  - 250 × 0.2 × 86,400 × $0.001 = **$4,320/day**
- Power cost: ~$0.50/day (M4 at 5W ANE draw)
- **Margin: >99%**

At scale, prices will compress. But ANE's power efficiency (2–5W vs GPU's 300W) means ANE providers can profitably undercut GPU providers by 10–50x on per-token cost.

### Fee Structure

| Fee | Rate | Recipient |
|-----|------|-----------|
| Protocol fee | 2% | Treasury |
| PoSP validator reward | 0.5% | Sampled validator |
| Challenge reward (if valid) | Attacker's stake | Challenger |
| x402 facilitator fee | ~0.1% | Coinbase facilitator |

---

## 9. Threat Model

| Threat | Impact | Mitigation | Residual Risk |
|--------|--------|-----------|---------------|
| Provider returns garbage | Client gets wrong answer | PoSP sampling + optimistic challenge + SEP attestation | Provider must beat all 3 layers simultaneously |
| Provider caches old results | Stale answers for new inputs | Input hash in SEP attestation + freshness nonce | None if SEP is trusted |
| Sybil attack (fake nodes) | Manipulate validator selection | Stake requirement + SEP device attestation (one SEP per physical device) | Attacker needs N physical Macs |
| Validator collusion | Approve bad results | Random VRF selection, multiple independent validators | Requires >2/3 of randomly selected validators |
| Eclipse attack on registry | Isolate nodes | On-chain registry (immutable), multiple bootstrap peers | Ethereum consensus guarantees |
| fp16 cross-chip divergence | False positive challenges | Chip-family equivalence classes + calibrated tolerance | Small epsilon risk, mitigated by same-family preference |
| Side-channel on SEP | Extract attestation key | Apple hardware security (FIPS 140-3 certified), cannot be done remotely | Physical access + nation-state resources |
| Model theft (weight exfiltration) | IP loss | Weights encrypted at rest, decrypted only in ANE pipeline. Future: encrypted compute via FHE | Provider has plaintext weights in memory during compute |
| Man-in-the-middle | Intercept tokens/payments | TLS 1.3 + x402 signatures | Standard web security |

### Trust Assumptions

1. **Apple Secure Enclave is not compromised** — Apple's SEP has [FIPS 140-3 certification](https://support.apple.com/guide/certifications/secure-enclave-processor-security-apc3a7433eb89/web). No known remote attacks.
2. **At least 1 honest validator exists per PoSP round** — With N>10 validators per round and random VRF selection, this is overwhelmingly likely.
3. **Ethereum L2 liveness** — Base/Arbitrum must be available for settlement. Standard L2 assumption.
4. **ANE determinism within chip family** — Empirically verified. If Apple changes ANE microarchitecture mid-generation, we detect via golden hash divergence and update equivalence classes.

---

## 10. Comparison to Existing Networks

| | **ANE Marketplace** | [Bittensor](https://bittensor.com) | [Ritual](https://ritual.net) | [Gensyn](https://docs.gensyn.ai/) | [Hyperbolic](https://www.hyperbolic.ai) |
|---|---|---|---|---|---|
| **Hardware** | Apple ANE (11–38 TOPS) | GPU | GPU | GPU | GPU |
| **Payment** | USDC via x402 | TAO token | RITUAL token | Native token | Native token |
| **Verification** | SEP + PoSP + Optimistic + ZK (layered) | [Yuma Consensus](https://metalamp.io/magazine/article/bittensor-overview-of-the-protocol-for-decentralized-machine-learning) (subjective scoring) | [Symphony](https://ritual.academy/ritual/architecture/) (EOVMT + ZK + TEE) | [Probabilistic proof-of-learning](https://docs.gensyn.ai/) (Verde) | [PoSP](https://arxiv.org/abs/2405.00295) (Nash equilibrium) |
| **Training** | Yes (ANE backprop) | Incentivized (subnets) | Fine-tuning | Yes (core focus) | Inference only |
| **TEE** | Apple SEP (native) | No | Intel SGX / AMD SEV | No | No |
| **Unique edge** | Unlocks idle Apple Silicon; 10–50x cheaper per watt | 119 specialized subnets; largest decentralized AI network | On-chain AI execution (EVM++) | Verified distributed training | Game-theoretic verification with minimal overhead |

**Our structural advantage:** Every Apple Silicon device already has a hardware security module (SEP). GPU-based networks have no equivalent — Intel SGX is [being retired](https://ts2.tech/en/trusted-execution-environment-tee-hardware-news-june-july-2025/), AMD SEV requires server hardware, NVIDIA confidential computing is datacenter-only. We get hardware attestation for free on consumer devices.

---

## 11. Implementation Roadmap

### Phase 1: Single-Node PoC
- [x] ANE inference engine (inference.m)
- [ ] x402 HTTP gateway wrapping inference binary
- [ ] SEP attestation of compute results
- [ ] Direct payment settlement on Base

### Phase 2: Multi-Node + PoSP
- [ ] On-chain provider registry with staking
- [ ] VRF-based validator selection for PoSP
- [ ] Determinism test suite (golden hashes per chip family)
- [ ] Inference pool with least-loaded routing

### Phase 3: Optimistic + Training
- [ ] Commit-reveal scheme with challenge window
- [ ] Merkle proof of training progress
- [ ] Spot-check verification protocol
- [ ] Data-parallel gradient aggregation

### Phase 4: ZK Integration
- [ ] zkPyTorch/EZKL integration for last-layer proof
- [ ] On-chain ZK verifier contract
- [ ] Incremental expansion of ZK-proven layers
- [ ] Softmax + sampling proof (proves output follows from logits)

### Phase 5: Production
- [ ] Formal verification of smart contracts (Certora/Halmos)
- [ ] Security audit (Trail of Bits / OpenZeppelin)
- [ ] x402 Foundation integration
- [ ] Encrypted compute exploration (FHE for model privacy)

---

## Sources

- [x402 Protocol — Coinbase](https://www.x402.org/)
- [x402 Developer Documentation](https://docs.cdp.coinbase.com/x402/welcome)
- [x402 Foundation — Cloudflare](https://blog.cloudflare.com/x402/)
- [Proof of Sampling — Hyperbolic / arXiv](https://arxiv.org/abs/2405.00295)
- [PoSP Deep Dive — Hyperbolic](https://www.hyperbolic.ai/blog/deep-dive-into-hyperbolic-proof-of-sampling)
- [6 Emerging AI Verification Solutions in 2025 — Gate.com](https://www.gate.com/learn/articles/the-6-emerging-ai-verification-solutions-in-2025/8399)
- [The Definitive Guide to ZKML (2025) — ICME](https://blog.icme.io/the-definitive-guide-to-zkml-2025/)
- [zkLLM: Zero Knowledge Proofs for Large Language Models — arXiv](https://arxiv.org/abs/2404.16109)
- [zkPyTorch — ePrint](https://eprint.iacr.org/2025/535.pdf)
- [Validity Proofs vs Fraud Proofs — Alchemy](https://www.alchemy.com/overviews/validity-proof-vs-fraud-proof)
- [Floating-Point Non-Associativity & Reproducibility — ORNL/SC'24](https://arxiv.org/html/2408.05148v3)
- [Defeating Nondeterminism in LLM Inference — Thinking Machines Lab](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- [Deterministic Inference in SGLang — LMSYS](https://lmsys.org/blog/2025-09-22-sglang-deterministic/)
- [Solving Reproducibility Challenges for ZKP — Ingonyama](https://www.ingonyama.com/oldblogs/solving-reproducibility-challenges-in-deep-learning-and-llms-our-journey)
- [Apple Secure Enclave — Apple Platform Security](https://support.apple.com/guide/security/the-secure-enclave-sec59b0b31ff/web)
- [Apple SEP FIPS 140-3 Certification](https://support.apple.com/guide/certifications/secure-enclave-processor-security-apc3a7433eb89/web)
- [TEE Hardware News 2025](https://ts2.tech/en/trusted-execution-environment-tee-hardware-news-june-july-2025/)
- [Gensyn Documentation](https://docs.gensyn.ai/)
- [Ritual Architecture](https://ritual.academy/ritual/architecture/)
- [Bittensor Protocol Overview](https://metalamp.io/magazine/article/bittensor-overview-of-the-protocol-for-decentralized-machine-learning)
- [ZK Proof AI in 2026 — Calibraint](https://www.calibraint.com/blog/zero-knowledge-proof-ai-2026)
- [Verifiable Compute for AI — Uplatz](https://uplatz.com/blog/verifiable-compute-for-ai-models-on-blockchain-the-convergence-of-cryptography-intelligence-and-consensus/)
- [Framework for Cryptographic Verifiability of AI Pipelines — arXiv](https://arxiv.org/html/2503.22573v1)
