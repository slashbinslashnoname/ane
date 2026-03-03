# ANE Compute Provider — Implementation Spec

MVP implementation: Secure Enclave attestation + Proof of Sampling + x402 payments. Model-agnostic, sandboxed execution, multi-tenant.

---

## 1. Design Principles

1. **Model-agnostic.** The provider accepts any llama2.c-format model. It does not hardcode dimensions, layer counts, or vocab sizes. Models are opaque weight blobs identified by hash.
2. **Isolated.** Each job runs in a sandboxed subprocess with its own memory, temp directory, and ANE kernel set. A crash or malicious model cannot affect other jobs or the host.
3. **Stateless between jobs.** No job state persists after completion. KV-cache, activations, compiled kernels — all freed. The provider is a pure function: `(model, input) → (output, proof)`.
4. **Minimal trust surface.** The provider binary is open-source and deterministically compiled. The Secure Enclave signs every result. The provider operator cannot tamper with outputs without invalidating the SEP attestation.

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      ANE Provider Node                        │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                    Gateway (Python)                      │ │
│  │                                                         │ │
│  │  ┌──────────┐  ┌──────────┐  ┌───────────────────────┐ │ │
│  │  │ x402     │  │ Job      │  │ Proof Assembler       │ │ │
│  │  │ Endpoint │─►│ Scheduler│─►│ (SEP sign + PoSP)     │ │ │
│  │  └──────────┘  └────┬─────┘  └───────────────────────┘ │ │
│  │                     │                                    │ │
│  └─────────────────────┼────────────────────────────────────┘ │
│                        │                                      │
│  ┌─────────────────────▼────────────────────────────────────┐ │
│  │               Sandbox Layer                               │ │
│  │                                                           │ │
│  │   ┌─────────┐  ┌─────────┐  ┌─────────┐                │ │
│  │   │ Job 1   │  │ Job 2   │  │ Job 3   │  ...           │ │
│  │   │         │  │         │  │         │                 │ │
│  │   │ model A │  │ model B │  │ model A │                 │ │
│  │   │ tmpdir/ │  │ tmpdir/ │  │ tmpdir/ │                 │ │
│  │   │ ANE krnl│  │ ANE krnl│  │ ANE krnl│                 │ │
│  │   └────┬────┘  └────┬────┘  └────┬────┘                │ │
│  │        │             │            │                      │ │
│  └────────┼─────────────┼────────────┼──────────────────────┘ │
│           │             │            │                         │
│  ┌────────▼─────────────▼────────────▼──────────────────────┐ │
│  │                ANE Hardware                               │ │
│  │   _ANEInMemoryModel compile / eval via private API        │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                               │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Secure Enclave Processor                                 │ │
│  │  (hardware root of trust, per-result signing)             │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                               │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Model Store                                              │ │
│  │  ~/.ane_provider/models/<hash>/                            │ │
│  │    weights.bin   (original weights, immutable)            │ │
│  │    manifest.json (dims, hash, format)                     │ │
│  └──────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. Model Abstraction Layer

The current codebase hardcodes Stories110M dimensions everywhere (`#define DIM 768`, etc.). The provider must be model-agnostic. Instead of rewriting the entire C codebase to be dynamic, we use a **compilation-per-model** strategy that plays to ANE's strengths.

### 3.1 Model Manifest

Every model in the store has a manifest:

```json
{
  "model_id": "sha256:<hash of weights.bin>",
  "format": "llama2c",
  "config": {
    "dim": 768,
    "hidden_dim": 2048,
    "n_layers": 12,
    "n_heads": 12,
    "n_kv_heads": 12,
    "vocab_size": 32000,
    "seq_len": 256
  },
  "size_bytes": 438000000,
  "uploaded_at": "2026-03-03T12:00:00Z"
}
```

### 3.2 Templated Compilation

Rather than making the C code dynamic (fragile, 50+ locations to change), we **generate a model-specific C source file from a template** at model registration time, compile it once, and cache the binary.

```
Model registration flow:

  1. Client uploads weights.bin
  2. Provider reads Llama2Config header from weights.bin
  3. Provider generates inference_<hash>.m from template:
     - sed s/%%DIM%%/768/g; s/%%HIDDEN%%/2048/g; ...
  4. Provider compiles: xcrun clang -O2 -o inference_<hash> inference_<hash>.m ...
  5. Provider stores binary in model store
  6. Provider hashes binary: binary_hash = sha256(inference_<hash>)
  7. Manifest updated with binary_hash
```

The template (`inference_template.m`) is identical to `inference.m` but with `%%PLACEHOLDER%%` tokens instead of hardcoded `#define` values:

```c
#define DIM %%DIM%%
#define HIDDEN %%HIDDEN%%
#define HEADS %%HEADS%%
#define HD (DIM/HEADS)
#define NLAYERS %%NLAYERS%%
#define VOCAB %%VOCAB%%
```

**Why this approach:**
- ANE kernels are compiled with dimensions baked in anyway (MIL text embeds tensor shapes). Dynamic dims don't help at the ANE level.
- A pre-compiled binary per model is faster than parsing config at runtime.
- The binary hash becomes part of the attestation — anyone can reproduce it from the same source + weights.
- No risk of buffer overflows from dynamic allocation mistakes.

### 3.3 Supported Model Formats

Phase 1 — **llama2.c binary**: The format used by stories110M and all karpathy/llama2.c models. Header is 7 ints (28 bytes) followed by contiguous weight arrays. Already implemented.

Phase 2 — **GGUF**: The format used by llama.cpp. Parse GGUF metadata for dimensions, extract F16/F32 weight tensors, repack into llama2.c layout. Covers Llama, Mistral, Phi, Gemma, Qwen — any architecture that's a stack of `{RMSNorm, QKV, Attn, FFN}` layers.

Phase 3 — **Safetensors/ONNX**: For arbitrary PyTorch models. Requires mapping layer names to the QKV/FFN structure.

### 3.4 Architecture Constraints

The ANE inference engine assumes a **Llama-family architecture**: RMSNorm → QKV → RoPE → Causal Attention → Wo → Residual → RMSNorm → SwiGLU FFN → Residual. This covers:

| Model Family | Compatible | Notes |
|---|---|---|
| Llama 1/2/3 | Yes | Native format |
| Mistral/Mixtral | Yes (dense layers) | MoE routing on CPU |
| Phi-1/2/3 | Yes | Same transformer block |
| Gemma | Yes | Minor norm differences |
| Qwen | Yes | Same architecture |
| GPT-2/NeoX | No | LayerNorm, not RMSNorm; different FFN |
| BERT/encoder | No | Bidirectional attention |

Non-Llama architectures require a separate inference template. The system supports multiple templates selected by architecture family.

---

## 4. Sandbox & Isolation

Every job runs in an isolated environment. A malicious model or input cannot escape.

### 4.1 Process Isolation

```
Gateway (parent)
  │
  ├── fork+exec: inference_<model_hash>
  │     │
  │     ├── Reads input from stdin (JSON)
  │     ├── Loads weights from read-only path
  │     ├── Compiles ANE kernels in private tmpdir
  │     ├── Runs inference
  │     ├── Writes output to stdout (JSON + proof data)
  │     └── Exits (all ANE kernels auto-unloaded)
  │
  └── Parent collects stdout, assembles proof, returns to client
```

### 4.2 Resource Limits

The gateway enforces per-job limits before exec:

| Resource | Limit | Enforcement |
|----------|-------|-------------|
| Wall time | 60s (inference), 600s (training batch) | `alarm()` / SIGKILL |
| Memory | 4 GB (inference), 16 GB (training) | `setrlimit(RLIMIT_AS)` |
| Disk (tmpdir) | 2 GB | Tmpfs mount with size cap |
| ANE compiles | 100 per process | Existing ANE limit (natural) |
| Network | None | `sandbox-exec` deny-network (macOS) |
| File access | Read-only model dir + private tmpdir | `sandbox-exec` profile |

### 4.3 macOS Sandbox Profile

```scheme
(version 1)
(deny default)

; Allow read from model store
(allow file-read*
  (subpath "/Users/*/ane_provider/models"))

; Allow read-write to job tmpdir
(allow file-read* file-write*
  (subpath "/private/tmp/ane_job_*"))

; Allow ANE framework access
(allow file-read*
  (subpath "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework"))

; Allow IOSurface
(allow iokit-open)
(allow mach-lookup
  (global-name "com.apple.ane.ioservice"))

; Deny network
(deny network*)

; Deny process control
(deny process-exec)
(deny signal)
```

### 4.4 Cleanup

On job completion (success or failure):
1. Kill subprocess if still running
2. `rm -rf /tmp/ane_job_<id>/`
3. ANE kernels auto-unload when process exits (OS reclaims)
4. Zero parent-process state change

---

## 5. Gateway Server

### 5.1 Endpoints

```
POST /v1/inference
  → x402 payment gated
  → Input: { model_id, tokens, max_tokens, temperature, seed, top_p }
  → Output: { tokens, proof_bundle }

POST /v1/training/step
  → x402 payment gated
  → Input: { model_id, data_url, steps, learning_rate }
  → Output: { losses, checkpoint_hash, merkle_root, proof_bundle }

POST /v1/models
  → Upload model weights (multipart/form-data)
  → Triggers template compilation
  → Output: { model_id, config, binary_hash }

GET /v1/models
  → List available models with configs and pricing

GET /v1/models/<id>
  → Model manifest + binary hash

GET /v1/health
  → { status, queue_depth, chip, ane_tops, models_loaded }

GET /v1/capabilities
  → { chip, memory, ane_tops, models, services, pricing }
```

### 5.2 x402 Integration

The gateway acts as an x402 resource server. Payment is verified before compute begins.

```python
@app.route('/v1/inference', methods=['POST'])
def inference():
    # Check for payment header
    payment = request.headers.get('X-PAYMENT')
    if not payment:
        # Return 402 with payment requirements
        return Response(
            status=402,
            headers={
                'X-PAYMENT-REQUIRED': json.dumps({
                    'address': PROVIDER_ADDRESS,
                    'amount': str(estimate_cost(request.json)),
                    'asset': 'USDC',
                    'chain': 'eip155:8453',  # Base
                    'expiry': int(time.time()) + 300,
                    'facilitator': 'https://x402.coinbase.com'
                })
            }
        )

    # Verify payment via facilitator
    if not verify_payment(payment, request.json):
        return Response(status=402, ...)

    # Payment valid — execute job
    result = execute_inference_job(request.json)
    return jsonify(result)
```

### 5.3 Job Queue

```
                  ┌──────────┐
   request ─────►│  Queue    │
                  │ (bounded) │
                  └────┬─────┘
                       │
          ┌────────────┼────────────┐
          │            │            │
     ┌────▼───┐  ┌────▼───┐  ┌────▼───┐
     │Worker 1│  │Worker 2│  │Worker 3│
     │(subproc│  │(subproc│  │(subproc│
     └────────┘  └────────┘  └────────┘
```

- **Queue depth** configurable (default: 8). Requests beyond capacity get 503.
- **Workers** = number of concurrent ANE jobs. Default: 2 (ANE can multiplex but throughput drops with >2 concurrent models).
- **Priority**: Higher-paying jobs dequeue first. Prevents low-value spam from blocking legitimate requests.

---

## 6. Proof Bundle

Every response includes a proof bundle — the evidence package that makes the result verifiable.

### 6.1 Structure

```json
{
  "proof_version": 1,

  "job": {
    "job_id": "uuid",
    "model_id": "sha256:abc...",
    "binary_hash": "sha256:def...",
    "input_hash": "keccak256(input_tokens)",
    "output_hash": "keccak256(output_tokens)",
    "seed": 42,
    "temperature": 0.0,
    "timestamp": 1709510400
  },

  "attestation": {
    "sep_signature": "<base64 Secure Enclave signature over job fields>",
    "cert_chain": ["<device cert>", "<intermediate>", "<Apple root>"],
    "device_id": "<hardware identifier>",
    "chip_family": "M4"
  },

  "reproducibility": {
    "logits_hash": "keccak256(final_logits_fp32)",
    "layer_checkpoints": {
      "6": "keccak256(residual_after_layer_6)"
    },
    "rng_state": "<ChaCha20 state after sampling>"
  }
}
```

### 6.2 What Each Part Proves

| Field | Proves | Verifiable by |
|-------|--------|---------------|
| `sep_signature` | A real Apple device produced this output | Anyone (verify cert chain) |
| `binary_hash` | The inference binary matches published source | Anyone (recompile and compare) |
| `input_hash` / `output_hash` | Commitment to specific I/O | Anyone (hash the data) |
| `logits_hash` | The raw logits before sampling | PoSP validator (re-run model, compare hash) |
| `layer_checkpoints` | Intermediate state at layer N | Spot-check verifier |
| `seed` + `rng_state` | Sampling was deterministic given seed | Anyone (replay sampling) |

### 6.3 SEP Signing Flow

The inference binary calls the Secure Enclave via the Keychain API:

```
1. At first boot, generate a SEP-bound EC key:
   SecKeyCreateRandomKey(kSecAttrTokenIDSecureEnclave, ...)

2. At end of each inference:
   payload = canonical_serialize(job_id, model_id, input_hash, output_hash, ...)
   signature = SecKeyCreateSignature(sep_key, kSecKeyAlgorithmECDSASignatureMessageX962SHA256, payload)

3. Return (signature, certificate_chain) as attestation
```

The private key never leaves the Secure Enclave. The signature proves this specific device produced this specific output.

---

## 7. Proof of Sampling (PoSP)

### 7.1 Roles

| Role | Who | Action |
|------|-----|--------|
| **Asserter** | The provider that computed the job | Submits output + proof bundle |
| **Orchestrator** | On-chain VRF contract | Decides which jobs get challenged, selects validators |
| **Validator** | Another ANE provider (randomly selected) | Re-runs computation, compares result |

### 7.2 Protocol

```
For each completed job:

1. Asserter posts on-chain:
   commit(job_id, output_hash, logits_hash, attestation)

2. Orchestrator draws VRF:
   challenge = VRF(block_hash, job_id) < challenge_threshold

   If not challenged (90-95% of jobs):
     → Finalize immediately, release payment

   If challenged (5-10% of jobs):
     → Select K validators from same chip_family
     → Validators receive (model_id, input_tokens, seed, temperature)

3. Each validator:
   a. Load same model (by model_id hash)
   b. Run inference with same input + seed
   c. Compute logits_hash and output_hash
   d. Submit: commit(job_id, validator_output_hash, validator_logits_hash)

4. Compare:
   If asserter.logits_hash == majority(validator.logits_hash):
     → Asserter correct. Finalize. Validators get reward.
   Else:
     → Asserter wrong. Slash asserter stake. Refund client.
```

### 7.3 Determinism Guarantees

For PoSP to work, the same input must produce the same output across nodes. We enforce this at multiple levels:

**Level 1 — Same binary.** All providers for a given model compile from the same template source. Binary hash published in manifest. Validators reject binary hash mismatch.

**Level 2 — Same MIL programs.** The MIL text is a deterministic function of (template_source, model_dimensions). Same dimensions → same MIL → same ANE program → same compute graph.

**Level 3 — Same chip family.** Validators selected from same chip family (M4 validates M4). Within a family, ANE produces bit-identical fp16 results because:
- Fixed dataflow (no thread scheduling variance)
- Deterministic reduction order (compiled into MIL)
- No atomic operations

**Level 4 — Deterministic sampling.** For temperature > 0, both asserter and validator use `ChaCha20(seed=keccak256(job_id))` as PRNG. Same seed → same random draws → same token sequence.

**Level 5 — Tolerance for cross-family.** When same-family validators aren't available, allow L1-norm tolerance on logits: `||logits_A - logits_B||_1 / vocab_size < 0.01`. Calibrated from empirical cross-family divergence measurements.

### 7.4 Nash Equilibrium Condition

From the [PoSP paper](https://arxiv.org/abs/2405.00295), honesty is the dominant strategy when:

```
p × (stake + reward) > cost_of_compute

Where:
  p = challenge probability (0.05 to 0.10)
  stake = provider's locked stake (e.g., 0.01 ETH)
  reward = job payment (e.g., 0.001 USDC)
  cost_of_compute = electricity + depreciation for one inference (~$0.0001)

Example:
  0.05 × (0.01 ETH + 0.001 USDC) > $0.0001
  0.05 × $30.001 > $0.0001
  $1.50 > $0.0001  ✓  (by a factor of 15,000)
```

Cheating is economically irrational by 4 orders of magnitude.

---

## 8. Model Store

### 8.1 Directory Layout

```
~/.ane_provider/
├── config.yaml                    # Provider configuration
├── keys/
│   └── sep_key_ref.pem            # Reference to SEP key (not the key itself)
├── models/
│   ├── sha256_a1b2c3.../
│   │   ├── weights.bin            # Original weights (read-only)
│   │   ├── manifest.json          # Config, hashes, metadata
│   │   ├── inference              # Compiled binary (model-specific)
│   │   └── tokenizer.bin          # Optional tokenizer
│   └── sha256_d4e5f6.../
│       └── ...
└── logs/
    └── provider.log
```

### 8.2 Model Lifecycle

```
Upload → Validate → Template → Compile → Hash → Register → Serve
  │         │          │          │        │        │         │
  │     read header  generate   xcrun   sha256   on-chain   ready
  │     check dims   .m file   clang   binary   announce
  │     check size
  │
  └─ reject if:
     - header invalid
     - dims unsupported (e.g., dim > 8192)
     - file too large (> 50 GB)
     - architecture not Llama-family
```

### 8.3 Cache & Eviction

- **Compiled binaries** cached indefinitely (small: ~200 KB each)
- **ANE kernel cache** managed by macOS (temp directories)
- **Weights** evicted LRU when disk exceeds configured limit (default: 100 GB)
- **Hot models** (served in last hour) never evicted

---

## 9. Configuration

```yaml
# ~/.ane_provider/config.yaml

provider:
  address: "0x..."                   # Ethereum address for payments
  private_key_path: "./keys/eth.key" # For signing on-chain txs
  stake: "0.01"                      # ETH staked on-chain

server:
  host: "0.0.0.0"
  port: 8402
  tls_cert: "./certs/cert.pem"
  tls_key: "./certs/key.pem"
  max_queue: 8
  max_workers: 2

x402:
  facilitator: "https://x402.coinbase.com"
  chain: "eip155:8453"               # Base
  asset: "USDC"

models:
  store_path: "~/.ane_provider/models"
  max_disk_gb: 100
  allowed_architectures: ["llama"]   # Restrict to known-safe architectures
  max_dim: 8192
  max_layers: 80
  max_vocab: 128000

sandbox:
  timeout_inference_s: 60
  timeout_training_s: 600
  max_memory_gb: 4
  network: deny

posp:
  registry_contract: "0x..."
  challenge_rate: 0.05               # 5% of jobs challenged
  validator_reward_bps: 50           # 0.5% of job value

logging:
  level: "info"
  file: "~/.ane_provider/logs/provider.log"
```

---

## 10. Wire Protocol

### 10.1 Inference Request

```
POST /v1/inference HTTP/1.1
Host: provider.example.com:8402
Content-Type: application/json
X-PAYMENT: <x402 payment payload>

{
  "model_id": "sha256:a1b2c3...",
  "tokens": [1, 4532, 817],
  "max_tokens": 256,
  "temperature": 0.0,
  "seed": 42,
  "top_p": 1.0,
  "stream": false,
  "include_logits": false
}
```

### 10.2 Inference Response

```json
{
  "tokens": [291, 1033, 445, 2],
  "text": "Once upon a time...",
  "finish_reason": "eos",
  "usage": {
    "prompt_tokens": 3,
    "completion_tokens": 4,
    "total_tokens": 7,
    "ane_time_ms": 12.4,
    "total_time_ms": 18.7
  },
  "proof": {
    "proof_version": 1,
    "job": { ... },
    "attestation": { ... },
    "reproducibility": { ... }
  }
}
```

### 10.3 IPC with Sandbox Subprocess

Gateway → subprocess communication via stdin/stdout:

```
Gateway writes to stdin:
{
  "command": "inference",
  "weights_path": "/path/to/weights.bin",
  "tokenizer_path": "/path/to/tokenizer.bin",
  "tokens": [1, 4532, 817],
  "max_tokens": 256,
  "temperature": 0.0,
  "seed": 42,
  "top_p": 1.0
}
<EOF>

Subprocess writes to stdout:
{
  "tokens": [291, 1033, 445, 2],
  "logits_hash": "abc...",
  "layer_checkpoints": {"6": "def..."},
  "ane_time_ms": 12.4
}
<EOF>
```

The subprocess reads JSON from stdin, runs inference, writes JSON to stdout, exits. No persistent connection. No shared state.

---

## 11. File Structure

```
eth_provider/
├── DESIGN.md                        # Strategic design document (verification theory)
├── IMPLEMENTATION.md                # This document
│
├── contracts/
│   └── ANEMarketplace.sol           # On-chain: escrow, staking, PoSP, slashing
│
├── gateway/
│   ├── server.py                    # Flask x402 server
│   ├── x402.py                      # x402 payment verification
│   ├── scheduler.py                 # Job queue + worker pool
│   ├── proof.py                     # Proof bundle assembly
│   └── posp.py                      # PoSP orchestrator client
│
├── sandbox/
│   ├── runner.py                    # Subprocess exec with sandbox-exec
│   ├── ane_sandbox.sb               # macOS sandbox profile
│   └── limits.py                    # Resource limit enforcement
│
├── models/
│   ├── store.py                     # Model store management (upload, compile, evict)
│   ├── template.py                  # C template generation from model config
│   └── templates/
│       └── inference_llama.m.tmpl   # Llama-family inference template
│
├── attestation/
│   ├── sep.py                       # Secure Enclave key management
│   └── verify.py                    # Attestation chain verification
│
├── config.py                        # Config loading (YAML)
├── requirements.txt
└── README.md                        # Quick start
```

---

## 12. Threat Model (Implementation-Specific)

| Threat | Mitigation |
|--------|-----------|
| Malicious model weights (crafted to exploit buffer overflow) | Sandbox: deny network, deny process-exec, memory limit. Template compilation validates all dimensions before generating C code. |
| Model too large for device memory | Pre-check: `weight_bytes < available_memory * 0.8`. Reject at upload. |
| Denial of service (flood requests) | Bounded queue (503 on overflow). x402 payment required (costs money to spam). Rate limit per wallet. |
| Provider returns cached stale result | Freshness nonce in every request. SEP attestation includes timestamp. PoSP validators use same nonce. |
| Provider runs weaker model (fewer layers) | Binary hash in attestation. Validators compile same template, compare hash. |
| Sandbox escape | macOS sandbox-exec is kernel-enforced. ANE access is read-only (no code execution on ANE). Process exits after each job. |
| Side-channel on model weights | Weights are read-only in sandbox. Network denied. No exfiltration path. |
| Compromised gateway (not subprocess) | Gateway never touches model weights or ANE. It only routes JSON. Attestation is signed by subprocess via SEP. |

---

## 13. Implementation Sequence

```
Week 1: Core plumbing
  ├── inference_llama.m.tmpl (parameterized inference template)
  ├── store.py (model upload, template gen, compile)
  ├── runner.py (subprocess exec with IPC)
  └── Verify: upload model → compile → run inference → get output

Week 2: Proof + attestation
  ├── Add logits_hash + layer_checkpoints to inference binary output
  ├── sep.py (SEP key gen + signing via Security.framework)
  ├── proof.py (assemble proof bundle)
  └── Verify: output includes valid SEP-signed proof

Week 3: x402 + server
  ├── server.py (Flask endpoints)
  ├── x402.py (payment verification via Coinbase facilitator)
  ├── scheduler.py (queue + workers)
  └── Verify: end-to-end paid inference request

Week 4: PoSP + on-chain
  ├── ANEMarketplace.sol (deploy to Base testnet)
  ├── posp.py (commit results, handle challenges)
  ├── Validator mode (re-run jobs on challenge)
  └── Verify: full loop — pay, compute, prove, challenge, settle
```
